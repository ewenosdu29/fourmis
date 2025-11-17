import csv
import random
import time
from math import sqrt
from typing import List, Optional, Tuple

import numpy as np
import tkinter as tk
from tkinter import scrolledtext

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

    def __init__(self, nb_lieux: int = NB_LIEUX, largeur: int = LARGEUR, hauteur: int = HAUTEUR):
        self.largeur = largeur
        self.hauteur = hauteur
        self.nb_lieux = int(nb_lieux)
        self.liste_lieux: List[Lieu] = []
        self.matrice_od: Optional[np.ndarray] = None
        self.generer_lieux_aleatoires(self.nb_lieux)

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

    # def plus_proche_voisin(self, index: int, visited: Optional[List[bool]] = None) -> int:
    #     if self.matrice_od is None:
    #         self.calcul_matrice_cout_od()
    #     row = self.matrice_od[index]
    #     n = self.nb_lieux
    #     best_idx = -1
    #     best_d = float('inf')
    #     for j in range(n):
    #         if j == index:
    #             continue
    #         if visited is not None and visited[j]:
    #             continue
    #         d = row[j]
    #         if d < best_d:
    #             best_d = d
    #             best_idx = j
    #     return best_idx
    
    def plus_proche_voisin(self, index: int, remaining: Optional[set] = None) -> int:
        """Retourne l'indice du plus proche voisin du lieu `index` dans remaining.
        Si remaining est None, on considère tous les lieux sauf index.
        """
        if self.matrice_od is None:
            self.calcul_matrice_cout_od()

        row = self.matrice_od[index]

        # conversion en array numpy pour vectorisation
        rem_list = np.array(list(remaining))
        distances = row[rem_list]
        best_idx = rem_list[np.argmin(distances)]
        return int(best_idx)
    

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
        if self.matrice_od is None:
            self.calcul_matrice_cout_od()

        if methode is None:
            methode = "2opt" if self.nb_lieux < 100 else "ppv"

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
    Améliorations: précalcul des puissances, vectorisation numpy.
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
        temps_max: float = None     # ⬅️ NOUVEAU
    ):
        self.graph = graph
        self.nb_fourmis = nb_fourmis
        self.nb_iterations = nb_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.route_initiale = route_initiale
        self.temps_max = temps_max     # ⬅️ sauvegarde du temps maximum autorisé
       
        if self.graph.matrice_od is None:
            self.graph.calcul_matrice_cout_od()
       
        self.n = self.graph.nb_lieux

        # Initialisation phéromones
        self.pheromones = self._initialiser_pheromones(route_initiale)
       
        # Heuristique 1/dist
        with np.errstate(divide='ignore', invalid='ignore'):
            self.heuristique = np.where(
                self.graph.matrice_od > 0,
                1.0 / self.graph.matrice_od,
                0
            )
        np.fill_diagonal(self.heuristique, 0)
       
        # Pré-calculs
        self.eta_beta = self.heuristique ** self.beta
        self.tau_alpha = None
        self._update_tau_alpha()
       
        # Historique
        self.meilleure_route = None
        self.meilleure_distance = float('inf')
        self.historique_distances = []
   
    def _initialiser_pheromones(self, route_initiale) -> np.ndarray:
        """Initialise la matrice de phéromones."""
        tau_0 = 0.1
        pheromones = np.full((self.n, self.n), tau_0)
       
        if route_initiale is not None:
            distance_init = route_initiale.calcul_distance()
            bonus = self.Q / distance_init
            bonus_factor = NB_LIEUX

            for i in range(len(route_initiale.ordre) - 1):
                a = route_initiale.ordre[i]
                b = route_initiale.ordre[i + 1]
                pheromones[a, b] += bonus * bonus_factor
                pheromones[b, a] += bonus * bonus_factor
       
        return pheromones
    
    def _update_tau_alpha(self):
        """OPTIMISATION: Précalculer tau^alpha après chaque mise à jour."""
        self.tau_alpha = self.pheromones ** self.alpha
   
    def _construire_solution(self) -> List[int]:
        """Une fourmi construit une solution complète."""
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
        """Choix de la prochaine ville vectorisé."""
        mask = ~visite
        if not np.any(mask):
            return 0
        
        probas = self.tau_alpha[ville_actuelle] * self.eta_beta[ville_actuelle]
        probas = probas * mask
        
        somme = probas.sum()
        if somme == 0:
            return np.random.choice(np.where(mask)[0])
        
        probas = probas / somme
        return np.random.choice(self.n, p=probas)
   
    def _evaporation(self):
        """Évaporation des phéromones."""
        self.pheromones *= (1 - self.rho)
        self._update_tau_alpha()
   
    def _depot_pheromones(self, tours):
        """Dépôt de phéromones vectorisé."""
        for ordre, distance in tours:
            depot = self.Q / distance
            
            aretes_i = np.array(ordre[:-1])
            aretes_j = np.array(ordre[1:])
            
            self.pheromones[aretes_i, aretes_j] += depot
            self.pheromones[aretes_j, aretes_i] += depot
        
        self._update_tau_alpha()
   
    def optimiser(self, verbose: bool = True):
        """Optimisation ACO avec interruption possible par temps_max."""
        
        start_time = time.time()

        for iteration in range(int(self.nb_iterations)):

            # ⏳ Vérification du temps au début de l'itération
            if self.temps_max is not None:
                if time.time() - start_time >= self.temps_max:
                    if verbose:
                        print(f"\n⏹️ Temps max dépassé ({self.temps_max}s). Retour du meilleur résultat.")
                    return self.meilleure_route, self.meilleure_distance

            tours = []
            for _ in range(self.nb_fourmis):

                # ⏳ Vérification même au milieu d'une iteration
                if self.temps_max is not None:
                    if time.time() - start_time >= self.temps_max:
                        if verbose:
                            print(f"\n⏹️ Temps max dépassé pendant la construction. Retour du meilleur résultat.")
                        return self.meilleure_route, self.meilleure_distance

                ordre = self._construire_solution()
                distance = self.graph.calcul_distance_route(ordre)
                tours.append((ordre, distance))
                
                if distance < self.meilleure_distance:
                    self.meilleure_distance = distance
                    self.meilleure_route = ordre.copy()

            # Mise à jour des phéromones
            self._evaporation()
            self._depot_pheromones(tours)
           
            self.historique_distances.append(self.meilleure_distance)
           
            if verbose:
                print(f"Itération {iteration+1} - Meilleure distance: {self.meilleure_distance:.2f}")
       
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

    def ameliorer_2opt(self):
        improved = True
        best_distance = self.calcul_distance()
        best_ordre = self.ordre.copy()

        while improved:
            improved = False
            for i in range(1, len(best_ordre) - 2):
                for j in range(i + 1, len(best_ordre) - 1):
                    if j - i == 1:
                        continue
                    new_ordre = best_ordre[:i] + best_ordre[i:j][::-1] + best_ordre[j:]
                    new_route = Route(self.graph, new_ordre)
                    new_distance = new_route.calcul_distance()

                    if new_distance < best_distance:
                        best_ordre = new_ordre
                        best_distance = new_distance
                        improved = True
            self.ordre = best_ordre.copy()
        return best_ordre, best_distance

    def __repr__(self):
        return f"Route(dist={self.calcul_distance():.2f}, ordre={self.ordre})"


# =========================================
# CLASSE AFFICHAGE
# =========================================

class Affichage:
    """Affichage Tkinter du graphe."""

    def __init__(self, graph: Graph, routes_population: Optional[List[Route]] = None, group_name: str = "Groupe TSP"):
        self.graph = graph
        if self.graph.matrice_od is None:
            self.graph.calcul_matrice_cout_od()
        self.routes_population = routes_population or []
        self.best_route: Optional[Route] = None
        if self.routes_population:
            self.best_route = min(self.routes_population, key=lambda r: r.calcul_distance())
        self.show_population = False
        self.show_matrix = False
        self.N_best = N_BEST

        self.root = tk.Tk()
        self.root.title(f"TSP - {group_name}")
        self.canvas = tk.Canvas(self.root, width=self.graph.largeur, height=self.graph.hauteur, bg="white")
        self.canvas.pack()
        self.text = scrolledtext.ScrolledText(self.root, height=8)
        self.text.pack(fill=tk.BOTH, expand=False)

        self.root.bind('<Escape>', lambda e: self.root.quit())
        self.root.bind('p', lambda e: self.toggle_population())
        self.root.bind('m', lambda e: self.toggle_matrix())

        self._draw_all()

    def _coord_canvas(self, lieu: Lieu) -> Tuple[float, float]:
        return lieu.x, lieu.y

    def _draw_lieux(self):
        self.canvas.delete('lieu')
        for idx, lieu in enumerate(self.graph.liste_lieux):
            x, y = self._coord_canvas(lieu)
            x0, y0 = x - RAYON_LIEU, y - RAYON_LIEU
            x1, y1 = x + RAYON_LIEU, y + RAYON_LIEU
            self.canvas.create_oval(x0, y0, x1, y1, fill='white', outline='black', tags='lieu')
            self.canvas.create_text(x, y, text=str(idx), tags='lieu')

    def _draw_route(self, route: Route, color: str = 'blue', dashed: bool = False, width: int = 2, tag: str = 'best'):
        if not route or not route.ordre:
            return
        coords = []
        for idx in route.ordre:
            lieu = self.graph.liste_lieux[idx]
            coords.extend(self._coord_canvas(lieu))
        dash = (4, 6) if dashed else None
        self.canvas.delete(tag)
        self.canvas.create_line(*coords, fill=color, width=width, dash=dash, tags=tag)

    def _draw_all(self):
        self.canvas.delete('all')
        self._draw_lieux()
        if self.best_route:
            self._draw_route(self.best_route)
        self._log_status()

    def _log_status(self):
        self.text.insert(tk.END, f"Heure: {time.strftime('%H:%M:%S')} - Lieux: {self.graph.nb_lieux}\n")
        if self.best_route:
            self.text.insert(tk.END, f"Meilleure distance: {self.best_route.calcul_distance():.2f}\n")
        self.text.see(tk.END)

    def toggle_population(self):
        self.show_population = not self.show_population
        self._draw_all()

    def toggle_matrix(self):
        self.show_matrix = not self.show_matrix
        self._draw_all()

    def mainloop(self):
        self.root.mainloop()


# =========================================
# PROGRAMME PRINCIPAL
# =========================================

if __name__ == '__main__':
   
    methode_heuristique = "ppv"
    nb_lieux = 5 # Augmente pour tester les performances

    # Création du graphe
    g = Graph(nb_lieux=nb_lieux)

    # ====== Phase 1 : Heuristique seule ======
    print("\n========== PHASE 1 : MÉTHODE HEURISTIQUE ==========")
    t0 = time.time()
    route_heur = g.route_heuristique(methode_heuristique)
    t1 = time.time()
    dist_heur = route_heur.calcul_distance()
    temps_heur = t1 - t0
    print(f"Distance obtenue avec {methode_heuristique.upper()} : {dist_heur:.2f}")
    print(f"Temps d'exécution ({methode_heuristique.upper()} seul) : {temps_heur:.3f} s")

    # ====== Phase 2 : ACO OPTIMISÉ avec heuristique ======
    print("\n========== PHASE 2 : ACO OPTIMISÉ (avec heuristique) ==========")
    t2 = time.time()
    aco = ACO_Optimized(
        graph = g,
        nb_fourmis = min(NB_LIEUX, 200),
        nb_iterations = 200,
        alpha = 1.0,
        beta = 4.0,
        rho = 0.3,
        Q = 100.0,
        route_initiale = route_heur,
    )

    meilleur_ordre, meilleure_distance = aco.optimiser(verbose=True)
    t3 = time.time()
    temps_aco = t3 - t2
    print(f"Distance finale avec ACO : {meilleure_distance:.2f}")
    print(f"Temps d'exécution (ACO + {methode_heuristique.upper()}) : {temps_aco:.3f} s")

    # Comparaison
    improvement = (dist_heur - meilleure_distance) / dist_heur * 100
    print("\n========== COMPARAISON ==========")
    print(f"📏 Distance {methode_heuristique.upper()} : {dist_heur:.2f}")
    print(f"📏 Distance ACO : {meilleure_distance:.2f}")
    print(f"⏱️  Temps {methode_heuristique.upper()} : {temps_heur:.3f} s")
    print(f"⏱️  Temps ACO : {temps_aco:.3f} s")
    print(f"⚡ Vitesse relative : {temps_heur/temps_aco:.2f}x")
    if improvement > 0:
        print(f"L'ACO a amélioré la solution de {improvement:.2f}%")
    elif improvement == 0:
        print("L'ACO a trouvé la même distance que l'heuristique")
    else:
        print(f"L'ACO a trouvé une solution moins bonne ({-improvement:.2f}% moins bonne)")

    # Affichage avec Tkinter
    route_aco = Route(g, meilleur_ordre)
    aff = Affichage(
        g,
        routes_population=[route_heur, route_aco],
        group_name=f"Comparaison {methode_heuristique.upper()} vs ACO ({g.nb_lieux} points)"
    )
    aff.mainloop()