import csv
import random
import time
from math import sqrt
from typing import List, Optional, Tuple

import numpy as np
import tkinter as tk
from tkinter import scrolledtext, Text
from concurrent.futures import ThreadPoolExecutor
import threading

# Constantes graphiques / environnement
LARGEUR = 800
HAUTEUR = 600
NB_LIEUX = 1000
RAYON_LIEU = 8
N_BEST = 5

class Lieu:
    """Représentation d'un lieu (x,y) avec un nom."""

    def __init__(self, x: float, y: float, name: Optional[str] = None):
        self.x = float(x)
        self.y = float(y)
        self.name = str(name) if name is not None else ""

    def distance(self, other: "Lieu") -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        return sqrt(dx * dx + dy * dy)

    def __repr__(self):
        return f"Lieu(name={self.name!r}, x={self.x:.2f}, y={self.y:.2f})"


class Graph:
    """Graphe de lieux."""
    def __init__(self, nb_lieux: int = NB_LIEUX, largeur: int = LARGEUR, hauteur: int = HAUTEUR, csv_file: Optional[str] = None):
        self.largeur = largeur
        self.hauteur = hauteur
        self.nb_lieux = int(nb_lieux)
        self.liste_lieux: List[Lieu] = []
        self.matrice_od: Optional[np.ndarray] = None

        # --- NOUVEAU: sparse matrix ---
        self.is_sparse = False
        self.k_voisins = 20           # nombre de voisins à conserver
        self.matrice_voisins: Optional[Tuple[np.ndarray, np.ndarray]] = None  # indices + distances

        if csv_file is not None:
            print(f"Chargement des lieux depuis le fichier CSV : {csv_file}")
            self.charger_lieux_depuis_csv(csv_file)
        else:
            print(f"Génération de {nb_lieux} lieux aléatoires")
            self.generer_lieux_aleatoires(self.nb_lieux)

        if csv_file is not None:
            print(f"Chargement des lieux depuis le fichier CSV : {csv_file}")
            self.charger_lieux_depuis_csv(csv_file)
        else:
            print(f"Génération de {nb_lieux} lieux aléatoires")
            self.generer_lieux_aleatoires(self.nb_lieux)

    def charger_lieux_depuis_csv(self, chemin_fichier: str):
        """Charge les lieux depuis un fichier CSV avec en-tête x,y"""
        self.liste_lieux = []
        with open(chemin_fichier, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # sauter l'en-tête
            for i, row in enumerate(reader):
                x = float(row[0])
                y = float(row[1])
                self.liste_lieux.append(Lieu(x, y, name=str(i)))
        self.nb_lieux = len(self.liste_lieux)
        self.matrice_od = None


    def generer_lieux_aleatoires(self, nb: int):
        self.liste_lieux = []
        margin = 20
        for i in range(nb):
            x = random.uniform(margin, self.largeur - margin)
            y = random.uniform(margin, self.hauteur - margin)
            self.liste_lieux.append(Lieu(x, y, name=str(i)))
        self.nb_lieux = len(self.liste_lieux)
        self.matrice_od = None

    def calcul_matrice_cout_od(self):
        """Calcule la matrice symétrique des distances euclidiennes entre tous les lieux (numpy optimisé)."""
        print("entrée matrice optimisé")
        import time
        start_time = time.time()

        n = self.nb_lieux
        coords = np.array([[l.x, l.y] for l in self.liste_lieux], dtype=float)

        # Matrice vide
        mat = np.zeros((n, n), dtype=float)

        # On calcule seulement la moitié supérieure
        for i in range(n):
            diff = coords[i+1:] - coords[i]      # vecteurs vers les points suivants
            dists = np.sqrt(np.sum(diff**2, axis=1))
            mat[i, i+1:] = dists
            mat[i+1:, i] = dists                 # symétrie

        self.matrice_od = mat

        end_time = time.time()
        print(f"Temps calcul matrice optimisé : {end_time - start_time:.6f} s")
        print("sortie matrice optimisé")
        return mat
    
    def plus_proche_voisin(self, index: int, remaining: Optional[set] = None) -> int:
        """
        Retourne l'indice du plus proche voisin du lieu `index` dans remaining.
        Si remaining est None, on considère tous les lieux sauf index.
        """
        if self.matrice_od is None:
            self.calcul_matrice_cout_od()
        
        print(f"Points restants : {len(remaining)}")            

        row = self.matrice_od[index]

        # conversion en array numpy pour vectorisation
        rem_list = np.array(list(remaining))
        distances = row[rem_list]
        best_idx = rem_list[np.argmin(distances)]
        return int(best_idx)
    
    # ================================
    # --- NOUVELLES FONCTIONS SPARSE ---
    # ================================



    def calcul_matrice_sparse_grille(self, k_voisins: int = 20, grid_size: int = 50):
        """
        Calcule une matrice sparse en utilisant une grille spatiale pour limiter les voisins candidats.
        Retourne (indices_voisins, distances_voisins)
        """
        print("Calcul matrice SPARSE avec grille...")
        self.is_sparse = True
        self.k_voisins = k_voisins

        n = self.nb_lieux
        coords = np.array([[l.x, l.y] for l in self.liste_lieux], dtype=float)

        # --- Construction de la grille ---
        indices_grid_x = (coords[:,0] // grid_size).astype(int)
        indices_grid_y = (coords[:,1] // grid_size).astype(int)
        grid = {}
        for idx in range(n):
            key = (indices_grid_x[idx], indices_grid_y[idx])
            grid.setdefault(key, []).append(idx)

        # --- Recherche des k voisins pour chaque point ---
        indices_voisins = np.zeros((n, k_voisins), dtype=int)
        dists_voisins = np.zeros((n, k_voisins), dtype=float)
        offsets = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,0), (0,1), (1,-1), (1,0), (1,1)]

        for idx in range(n):
            gx, gy = indices_grid_x[idx], indices_grid_y[idx]
            candidats = []
            for dx, dy in offsets:
                candidats.extend(grid.get((gx+dx, gy+dy), []))

            pts_cands = coords[candidats]
            pt_curr = coords[idx]
            d = np.sqrt(np.sum((pts_cands - pt_curr)**2, axis=1))
            # on prend les k plus proches (en ignorant soi-même)
            k = min(len(d)-1, k_voisins)
            best_idx = np.argpartition(d, k)[:k+1]  # +1 pour exclure soi-même
            final_indices = np.array(candidats)[best_idx]
            final_dists = d[best_idx]

            mask = final_indices != idx
            indices_voisins[idx,:len(final_indices[mask])] = final_indices[mask]
            dists_voisins[idx,:len(final_indices[mask])] = final_dists[mask]

        self.matrice_voisins = (indices_voisins, dists_voisins)
        print("Matrice sparse avec grille calculée.")
        return self.matrice_voisins


    def plus_proche_voisin_sparse_grille(self, index: int, remaining: Optional[set] = None) -> int:
        """Retourne le plus proche voisin dans la matrice sparse calculée via grille."""
        if self.matrice_voisins is None:
            self.calcul_matrice_sparse_grille(self.k_voisins)

        indices_voisins, _ = self.matrice_voisins
        candidats = indices_voisins[index]
        if remaining is not None:
            candidats = [i for i in candidats if i in remaining]
        if not candidats:   # secours si aucun voisin valide
            candidats = list(remaining)
        return candidats[0]


    def calcul_distance_route(self, ordre: List[int]) -> float:
        """VERSION OPTIMISÉE avec numpy vectorisé."""
        if not ordre:
            return 0.0
        
        if self.matrice_od is None:
            self.calcul_matrice_cout_od()
        
        # OPTIMISATION: Vectorisation au lieu de boucle
        ordre_arr = np.array(ordre)
        indices_i = ordre_arr[:-1]
        indices_j = ordre_arr[1:]
        dist = self.matrice_od[indices_i, indices_j].sum()
        
        return float(dist)
    

    def route_heuristique(self, methode: Optional[str] = None) -> "Route":

        print(f"\nMéthode utilisée : {methode.upper()} (nb_lieux = {self.nb_lieux})")

        if methode == "ppv":
            n = self.nb_lieux
            remaining = set(range(1, n))
            ordre = [0]
            current = 0

            for _ in range(n - 1):
                nxt = self.plus_proche_voisin(current, remaining)
                ordre.append(nxt)
                remaining.remove(nxt)
                current = nxt

            ordre.append(0)
            return Route(self, ordre)
        
        if methode == "ppv_adaptatif":
            n = self.nb_lieux
            remaining = set(range(1, n))
            ordre = [0]
            current = 0
            distance_totale = 0.0  # 👈 on initialise la distance

            for _ in range(n - 1):
                nxt = self.plus_proche_voisin_adaptatif(current, remaining)
                # calcul direct de la distance entre current et nxt
                li = self.liste_lieux[current]
                lj = self.liste_lieux[nxt]
                dx = li.x - lj.x
                dy = li.y - lj.y
                distance_totale += (dx*dx + dy*dy)**0.5

                ordre.append(nxt)
                remaining.remove(nxt)
                current = nxt

            # ajouter le retour au point de départ
            li = self.liste_lieux[current]
            lj = self.liste_lieux[0]
            dx = li.x - lj.x
            dy = li.y - lj.y
            distance_totale += (dx*dx + dy*dy)**0.5

            ordre.append(0)
            route = Route(self, ordre)
            # on stocke la distance calculée directement
            route.distance_directe = distance_totale
            return route


        elif methode == "2opt":
            route_init = Route(self)
            route_init.ameliorer_2opt()
            return route_init

        elif methode == "aco":
            aco = ACO_Optimized(graph=self)  # Utilise la version optimisée
            meilleur_ordre, _ = aco.optimiser(verbose=False)
            return Route(self, meilleur_ordre)

        else:
            raise ValueError("Méthode inconnue. Utilisez 'ppv', '2opt' ou 'aco'.")


# =========================================
# CLASSE ACO OPTIMISÉE
# =========================================

class ACO_Optimized:
    """
    Ant Colony Optimization OPTIMISÉ pour le problème du voyageur de commerce.
    Combinaison des deux versions : vectorisation numpy, callback, gestion temps max.
    """
    def __init__(
        self,
        graph,
        nb_fourmis: int = 30,
        nb_iterations: int = 100,
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.5,
        Q: float = 100.0,
        route_initiale=None,
        temps_max: float = None
    ):
        self.graph = graph
        self.nb_fourmis = nb_fourmis
        self.nb_iterations = nb_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.route_initiale = route_initiale
        self.temps_max = temps_max

        self.n = self.graph.nb_lieux

        # Initialisation phéromones
        self.pheromones = self._initialiser_pheromones(route_initiale)

        # Heuristique 1/dist
        with np.errstate(divide='ignore', invalid='ignore'):
            self.heuristique = np.where(self.graph.matrice_od > 0, 1.0 / self.graph.matrice_od, 0)
        np.fill_diagonal(self.heuristique, 0)

        # Pré-calculs vectorisés
        self.eta_beta = self.heuristique ** self.beta
        self.tau_alpha = None
        self._update_tau_alpha()

        # Historique et meilleure solution
        self.meilleure_route = None
        self.meilleure_distance = float('inf')
        self.historique_distances = []

    def _initialiser_pheromones(self, route_initiale) -> np.ndarray:
        tau_0 = 0.1
        pheromones = np.full((self.n, self.n), tau_0)
        if route_initiale is not None:
            distance_init = route_initiale.calcul_distance()
            bonus = self.Q / distance_init
            bonus_factor = self.n
            for i in range(len(route_initiale.ordre) - 1):
                a, b = route_initiale.ordre[i], route_initiale.ordre[i+1]
                pheromones[a, b] += bonus * bonus_factor
                pheromones[b, a] += bonus * bonus_factor
        return pheromones

    def _update_tau_alpha(self):
        self.tau_alpha = self.pheromones ** self.alpha

    def _construire_solution(self) -> list[int]:
        tour = [0]
        visite = np.zeros(self.n, dtype=bool)
        visite[0] = True
        ville_actuelle = 0

        for _ in range(self.n - 1):
            prochaine = self._choisir_prochaine_ville_fast(ville_actuelle, visite)
            tour.append(prochaine)
            visite[prochaine] = True
            ville_actuelle = prochaine

        tour.append(0)
        return tour

    def _choisir_prochaine_ville_fast(self, ville_actuelle: int, visite: np.ndarray) -> int:
        mask = ~visite
        if not np.any(mask):
            return 0
        probas = self.tau_alpha[ville_actuelle] * self.eta_beta[ville_actuelle] * mask
        somme = probas.sum()
        if somme == 0:
            return np.random.choice(np.where(mask)[0])
        return np.random.choice(self.n, p=probas/somme)

    def _evaporation(self):
        self.pheromones *= (1 - self.rho)
        self._update_tau_alpha()

    def _depot_pheromones(self, tours):
        for ordre, distance in tours:
            depot = self.Q / distance
            aretes_i = np.array(ordre[:-1])
            aretes_j = np.array(ordre[1:])
            self.pheromones[aretes_i, aretes_j] += depot
            self.pheromones[aretes_j, aretes_i] += depot
        self._update_tau_alpha()

    def optimiser(self, verbose: bool = True, callback=None) -> tuple[list[int], float]:
        start_time = time.time()
        for iteration in range(self.nb_iterations):
            # Vérification du temps max avant l'itération
            if self.temps_max is not None and time.time() - start_time >= self.temps_max:
                if verbose:
                    print(f"\n⏹️ Temps max dépassé ({self.temps_max}s). Retour du meilleur résultat.")
                return self.meilleure_route, self.meilleure_distance

            tours = []
            for _ in range(self.nb_fourmis):
                # Vérification du temps max au milieu d'une iteration
                if self.temps_max is not None and time.time() - start_time >= self.temps_max:
                    if verbose:
                        print(f"\n⏹️ Temps max dépassé pendant la construction. Retour du meilleur résultat.")
                    return self.meilleure_route, self.meilleure_distance

                ordre = self._construire_solution()
                distance = self.graph.calcul_distance_route(ordre)
                tours.append((ordre, distance))
                if distance < self.meilleure_distance:
                    self.meilleure_distance = distance
                    self.meilleure_route = ordre.copy()

            # Mise à jour phéromones
            self._evaporation()
            self._depot_pheromones(tours)
            self.historique_distances.append(self.meilleure_distance)

            # Callback pour affichage live
            if callback is not None and self.meilleure_route is not None:
                callback=lambda r, p=None: aff.update_affichage(r, p)


            if verbose:
                print(f"Itération {iteration+1}/{self.nb_iterations} - Meilleure distance : {self.meilleure_distance:.2f}")

        return self.meilleure_route, self.meilleure_distance



# =========================================
# ============= CLASSE ROUTE ==============
# =========================================

class Route:
    """Représente une route (ordre de visites)."""

    def __init__(self, graph: Graph, ordre: Optional[List[int]] = None):
        self.graph = graph
        if ordre is None:
            perm = list(range(1, graph.nb_lieux))
            random.shuffle(perm)
            ordre_gen = [0] + perm + [0]
            self.ordre = ordre_gen
        else:
            if ordre[0] != 0:
                ordre = [0] + ordre
            if ordre[-1] != 0:
                ordre = ordre + [0]
            self.ordre = ordre

    def calcul_distance(self) -> float:
        return self.graph.calcul_distance_route(self.ordre)

    def ameliorer_2opt(self, callback=None, temps_max=None):
        import time
        start_time = time.time()

        improved = True
        best_distance = self.calcul_distance()
        best_ordre = self.ordre.copy()
        passe = 0

        while improved:
            passe += 1
            improved = False
            print(f"=== Passe {passe} ===")
            for i in range(1, len(best_ordre) - 2):
                for j in range(i + 1, len(best_ordre) - 1):
                    if j - i == 1:
                        continue

                    # Vérifie si temps_max est dépassé
                    if temps_max is not None and (time.time() - start_time) > temps_max:
                        print(f"\n⚠ Temps max {temps_max}s atteint, arrêt prématuré")
                        self.ordre = best_ordre.copy()
                        return best_ordre, best_distance

                    new_ordre = best_ordre[:i] + best_ordre[i:j][::-1] + best_ordre[j:]
                    new_route = Route(self.graph, new_ordre)
                    new_distance = new_route.calcul_distance()

                    # Affichage statique des indices
                    print(f"i = {i}, j = {j}", end="\r", flush=True)

                    if new_distance < best_distance:
                        best_ordre = new_ordre
                        best_distance = new_distance
                        improved = True
                        print(f"  -> Amélioration trouvée : {best_distance:.2f}")

                        # Callback pour affichage intermédiaire
                        if callback is not None:
                            callback(best_ordre, best_distance)

            self.ordre = best_ordre.copy()
            print(f"Fin de passe {passe}, distance actuelle : {best_distance:.2f}\n")

        return best_ordre, best_distance





    def __repr__(self):
        return f"Route(dist={self.calcul_distance():.2f}, ordre={self.ordre})"
    
    # ==========================================
    # --- NOUVELLE MÉTHODE POUR SPARSE + GRILLE ---
    # ==========================================
    @classmethod
    def from_sparse_grille(cls, graph: Graph) -> "Route":
        """
        Construit une route en utilisant la méthode sparse + grille.
        Retourne une instance de Route.
        """
        remaining = set(range(1, graph.nb_lieux))
        ordre = [0]
        current = 0

        while remaining:
            nxt = graph.plus_proche_voisin_sparse_grille(current, remaining)
            ordre.append(nxt)
            remaining.remove(nxt)
            current = nxt

        # retour au point de départ
        ordre.append(0)

        return cls(graph, ordre)


# =========================================
# CLASSE AFFICHAGE
# =========================================
class Affichage(tk.Tk):
    def __init__(self, graph):
        super().__init__()
        self.graph = graph
        self.title("JC.Ilan C.Paul H.Ewen T.Lucas        Groupe 5")
        self.canvas = tk.Canvas(self, width=LARGEUR, height=HAUTEUR, bg='white')
        self.canvas.pack()
        self.text_zone = tk.Text(self, height=5, width=80, bg='lightyellow')
        self.text_zone.pack()

        self.affiche_pheromones = False
        self.pheromones = None
        self.route = None

        self.bind('<Escape>', lambda e: self.destroy())
        self.bind('f', self.toggle_pheromones)

        # Nouveau flag pour limiter l'affichage
        self.simple_affichage = self.graph.nb_lieux > 200

    # ------------------------------
    # AFFICHAGE DES LIEUX ET ROUTE
    # ------------------------------
    def afficher_lieux(self, route):
        if self.simple_affichage:
            # On ne dessine pas les ronds ni les textes
            return

        self.canvas.delete("lieux")
        self.canvas.delete("ordres")
        for ordre_idx, lieu_idx in enumerate(route.ordre[:-1]):
            lieu = self.graph.liste_lieux[lieu_idx]
            couleur = "red" if lieu_idx == 0 else "#D3D3D3"
            self.canvas.create_oval(lieu.x-RAYON_LIEU, lieu.y-RAYON_LIEU, 
                                    lieu.x+RAYON_LIEU, lieu.y+RAYON_LIEU,
                                    fill=couleur, outline="black", tags="lieux")
            self.canvas.create_text(lieu.x, lieu.y, text=str(lieu_idx), tags="lieux")
            self.canvas.create_text(lieu.x, lieu.y-14, text=str(ordre_idx), fill="gray", tags="ordres")

    def afficher_route(self, route):
        self.canvas.delete("route")
        for i in range(len(route.ordre)-1):
            a = self.graph.liste_lieux[route.ordre[i]]
            b = self.graph.liste_lieux[route.ordre[i+1]]
            # Si simple affichage, on peut changer couleur/épaisseur pour mieux voir
            width = 2 if not self.simple_affichage else 1
            self.canvas.create_line(a.x, a.y, b.x, b.y, fill='blue', dash=(6,6), width=width, tags="route")

    def afficher_pheromones_graph(self, pheromones):
        """Affiche les phéromones sur le canvas."""
        self.canvas.delete("pheromones")
        if not self.affiche_pheromones or pheromones is None:
            return
        MAX_WIDTH = 8
        SEUIL_RELATIF = 0.1
        max_ph = np.max(pheromones) if np.max(pheromones) > 0 else 1
        for i in range(self.graph.nb_lieux):
            for j in range(i+1, self.graph.nb_lieux):
                p = pheromones[i,j]
                if p < SEUIL_RELATIF * max_ph:
                    continue
                a = self.graph.liste_lieux[i]
                b = self.graph.liste_lieux[j]
                width = max(1, (p / max_ph) * MAX_WIDTH)
                self.canvas.create_line(a.x, a.y, b.x, b.y, fill="lightpink", width=width, tags="pheromones")

    def update_affichage(self, route, pheromones=None):
        """Met à jour tout l'affichage."""
        self.route = route
        self.pheromones = pheromones
        self.canvas.delete("all")
        self.afficher_lieux(route)
        if pheromones is not None and self.affiche_pheromones:
            self.afficher_pheromones_graph(pheromones)
        self.afficher_route(route)
        self.update()  # Permet à l'ACO de rafraîchir le canvas directement

    # ------------------------------
    # LOGGING
    # ------------------------------
    def log(self, msg):
        self.text_zone.insert(tk.END, msg + "\n")
        self.text_zone.see(tk.END)

    # ------------------------------
    # INTERACTIONS
    # ------------------------------
    def toggle_pheromones(self, e=None):
        self.affiche_pheromones = not self.affiche_pheromones
        if self.route is not None:
            self.update_affichage(self.route, self.pheromones)




# =========================================
# PROGRAMME PRINCIPAL
# =========================================

if __name__ == '__main__':
    # ===============================
    #   CONFIGURATION
    # ===============================
    csv_file = None  # <-- mettre None pour générer des randoms
    nb_lieux = 50   # utilisé uniquement si csv_file=None
    tps_max = 40

    # Création du graphe
    if csv_file is not None:
        g = Graph(csv_file=csv_file)
    else:
        g = Graph(nb_lieux=nb_lieux)

    # ===============================
    #   PHASE 1 : Heuristique
    # ===============================
    if g.nb_lieux <= 10000:
        methode_heuristique = "ppv"
    else:
        methode_heuristique = "ppv_sparse_grille"
        # calcul de la sparse matrix avec grille si nécessaire
        g.calcul_matrice_sparse_grille(k_voisins=100)

    print(f"\n========== PHASE 1 : MÉTHODE HEURISTIQUE ({methode_heuristique.upper()}) ==========")
    t0 = time.time()

    # construction de la route selon la méthode
    if methode_heuristique == "ppv_sparse_grille":
        ordre_init = g.plus_proche_voisin_sparse_grille(0, remaining=set(range(1, g.nb_lieux)))
        route_heur = Route.from_sparse_grille(g)
    else:
        route_heur = g.route_heuristique("ppv")

    dist_heur = route_heur.calcul_distance()
    t1 = time.time()
    temps_heur = t1 - t0

    print(f"Distance obtenue avec {methode_heuristique.upper()} : {dist_heur:.2f}")
    print(f"Temps d'exécution ({methode_heuristique.upper()} seul) : {temps_heur:.3f} s")

    # ===============================
    #   INITIALISATION AFFICHAGE
    # ===============================
    # ===============================
    #   INITIALISATION AFFICHAGE
    # ===============================
    aff = Affichage(g)

    # Affichage immédiat de la route initiale PPV
    aff.update_affichage(route_heur)
    aff.log(f"Heuristique {methode_heuristique.upper()} : {dist_heur:.2f}")

    # ===============================
    #   PHASE 2 : Optimisation
    # ===============================
    if g.nb_lieux <= 10000:
        # ACO Optimisé
        print("\n========== PHASE 2 : ACO OPTIMISÉ ==========")
        t2 = time.time()
        aco = ACO_Optimized(
            graph=g,
            nb_fourmis=200,
            nb_iterations=10000,
            alpha=1.0,
            beta=4.0,
            rho=0.3,
            Q=100.0,
            route_initiale=route_heur,
            temps_max=tps_max
        )
        # Callback pour mise à jour live
        meilleur_ordre, meilleure_distance = aco.optimiser(
            verbose=True,
            callback=lambda r, p=None: aff.update_affichage(Route(g, r), p)
        )
        t3 = time.time()
        temps_aco = t3 - t2
    else:
        # 2OPT sur PPV
        print("\n========== PHASE 2 : 2OPT SUR PPV ==========")
        t2 = time.time()

        # On modifie ameliorer_2opt pour qu'il accepte un callback
        def callback_2opt(route_ordre, distance):
            route = Route(g, route_ordre)
            aff.update_affichage(route)
            aff.log(f"2OPT intermédiaire : {distance:.2f}")

        meilleur_ordre, meilleure_distance = route_heur.ameliorer_2opt(
            temps_max=60,  # par exemple 60 secondes max
            callback=callback_2opt
        )

        
        tps_max

        t3 = time.time()
        temps_aco = t3 - t2
        route_2opt = Route(g, meilleur_ordre)
        aff.update_affichage(route_2opt)
        aff.log(f"2OPT final : {meilleure_distance:.2f}")


    # ===============================
    #   COMPARAISON
    # ===============================
    improvement = (dist_heur - meilleure_distance) / dist_heur * 100
    print("\n========== COMPARAISON ==========")
    print(f"📏 Distance {methode_heuristique.upper()} : {dist_heur:.2f}")
    print(f"📏 Distance finale : {meilleure_distance:.2f}")
    print(f"⏱️  Temps {methode_heuristique.upper()} : {temps_heur:.3f} s")
    print(f"⏱️  Temps optimisation : {temps_aco:.3f} s")
    if improvement > 0:
        print(f"⚡ Amélioration : {improvement:.2f}%")
    elif improvement == 0:
        print("Solution finale égale à l’heuristique.")
    else:
        print(f"⚠ Solution finale moins bonne de {-improvement:.2f}%")

    # ===============================
    #   AFFICHAGE FINAL
    # ===============================
    def afficher_final():
        route_finale = Route(g, meilleur_ordre)
        aff.update_affichage(route_finale)
        aff.log(f"Distance finale : {meilleure_distance:.2f}  ({improvement:.2f}% d'amélioration)")
        aff.log(f"Temps heuristique : {temps_heur:.2f}s")
        aff.log(f"Temps total : {temps_heur + temps_aco:.2f}s")

    aff.after(2000, afficher_final)
    aff.mainloop()
