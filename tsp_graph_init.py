import csv
import random
import time
from math import sqrt
import numpy as np
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
import threading


# Constantes
TPS_MAX = 10
CSV_FILE = "graph_5.csv"
LARGEUR = 800
HAUTEUR = 600
NB_LIEUX = 1000
RAYON_LIEU = 8
SEUIL_BIG_DATA = 10000


from math import sqrt

class Lieu:
    def __init__(self, x, y, name=None):
        self.x = float(x)
        self.y = float(y)
        self.name = str(name) if name is not None else ""

    def distance(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        return sqrt(dx * dx + dy * dy)

    def __repr__(self):
        return f"Lieu(name={self.name!r}, x={self.x:.2f}, y={self.y:.2f})"


class Graph:
    def __init__(self, nb_lieux=NB_LIEUX, largeur=LARGEUR, hauteur=HAUTEUR, csv_file=None):
        self.largeur = largeur
        self.hauteur = hauteur
        self.nb_lieux = int(nb_lieux)
        self.liste_lieux = []
        self.matrice_od = None
        self.coords_np = None
        self.is_sparse = False
        self.k_voisins = 100

        if csv_file is not None:
            print(f"Chargement des lieux depuis : {csv_file}")
            self.charger_graph(csv_file)
        else:
            print(f"Génération de {nb_lieux} lieux aléatoires")
            self.generer_lieux_aleatoires(self.nb_lieux)

    def charger_graph(self, chemin_fichier):
        self.liste_lieux = []
        try:
            with open(chemin_fichier, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                reader.fieldnames = [name.strip() for name in reader.fieldnames]
                for i, row in enumerate(reader):
                    x = float(row.get('x', row.get('lon', 0)))
                    y = float(row.get('y', row.get('lat', 0)))
                    name = row.get('nom', row.get('name', str(i)))
                    self.liste_lieux.append(Lieu(x, y, name=name))
            self.nb_lieux = len(self.liste_lieux)
        except FileNotFoundError:
            self.generer_lieux_aleatoires(NB_LIEUX)

    def generer_lieux_aleatoires(self, nb):
        self.liste_lieux = []
        coords = np.random.uniform(0, [self.largeur, self.hauteur], size=(nb, 2))
        for i in range(nb):
            self.liste_lieux.append(Lieu(coords[i][0], coords[i][1], name=str(i)))
        self.nb_lieux = len(self.liste_lieux)

    def calcul_matrice_cout_od(self):
        n = self.nb_lieux
        self.coords_np = np.array([[l.x, l.y] for l in self.liste_lieux], dtype=np.float32)

        if n <= SEUIL_BIG_DATA:
            print("Calcul Matrice Complète...")
            diff = self.coords_np[:, np.newaxis, :] - self.coords_np[np.newaxis, :, :]
            self.matrice_od = np.sqrt(np.sum(diff**2, axis=-1))
            self.is_sparse = False
        else:
            print(f"Mode BIG DATA ({n} lieux). Indexation Spatiale...")
            self.is_sparse = True
            self.matrice_od = self._calculer_voisins_grille(self.coords_np, n)

    def _calculer_voisins_grille(self, coords, n):
        t0 = time.time()
        grid_size = 50 
        grid = {}
        
        ix = (coords[:, 0] // grid_size).astype(int)
        iy = (coords[:, 1] // grid_size).astype(int)
        for i in range(n):
            k = (ix[i], iy[i])
            if k not in grid: grid[k] = []
            grid[k].append(i)
            
        indices_voisins = np.zeros((n, self.k_voisins), dtype=int)
        
        def process_chunk(start, end):
            offsets = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,0), (0,1), (1,-1), (1,0), (1,1)]
            for i in range(start, end):
                k_curr = (ix[i], iy[i])
                cands = []
                for dx, dy in offsets:
                    nk = (k_curr[0]+dx, k_curr[1]+dy)
                    if nk in grid: cands.extend(grid[nk])
                
                cands_idx = np.array(cands, dtype=int)
                if len(cands_idx) <= self.k_voisins:
                    cands_idx = np.random.randint(0, n, self.k_voisins+1)
                
                pts_c = coords[cands_idx]
                pt_i = coords[i]
                d = np.sqrt(np.sum((pts_c - pt_i)**2, axis=1))
                
                k_take = min(len(d)-1, self.k_voisins + 1)
                part_idx = np.argpartition(d, k_take)[:k_take+1]
                sorted_idx = part_idx[np.argsort(d[part_idx])]
                
                raw_idx = cands_idx[sorted_idx]
                final = raw_idx[raw_idx != i][:self.k_voisins]
                indices_voisins[i, :len(final)] = final

        n_cpu = max(1, 7)
        chunk = n // n_cpu
        with ThreadPoolExecutor(max_workers=n_cpu) as exe:
            for i in range(n_cpu):
                s = i * chunk
                e = n if i == n_cpu - 1 else (i+1) * chunk
                exe.submit(process_chunk, s, e)
                
        print(f"Indexation terminée ({time.time()-t0:.2f}s)")
        return (indices_voisins, None)

    def plus_proche_voisin(self, index, remaining=None):
        if self.matrice_od is None: self.calcul_matrice_cout_od()
        
        if self.is_sparse:
            voisins_possibles, _ = self.matrice_od
            candidats = voisins_possibles[index]
            for c in candidats:
                if c in remaining: return int(c)
            return next(iter(remaining))
        else:
            row = self.matrice_od[index]
            rem_list = np.array(list(remaining))
            dists = row[rem_list]
            return int(rem_list[np.argmin(dists)])

    def calcul_distance_route(self, ordre):
        if not ordre: return 0.0
        if self.coords_np is None: self.calcul_matrice_cout_od()
        
        o = np.array(ordre)
        pts_a = self.coords_np[o[:-1]]
        pts_b = self.coords_np[o[1:]]
        return float(np.sum(np.sqrt(np.sum((pts_a - pts_b)**2, axis=1))))

    def route_heuristique(self, methode=None):
        print(f"\nMéthode utilisée : {methode.upper()} (nb_lieux = {self.nb_lieux})")

        n = self.nb_lieux
        if methode == "ppv":
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
        else:
            return self.route_heuristique("ppv") 


class TSP_ACO:
    def __init__(
        self,
        graph,
        nb_fourmis=30,
        nb_iterations=100,
        alpha=1.0,
        beta=2.0,
        rho=0.5,
        Q=100.0,
        route_initiale=None,
        temps_max=None
    ):
        self.graph = graph
        self.nb_fourmis = nb_fourmis
        self.nb_iterations = nb_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.temps_max = temps_max
        self.n = self.graph.nb_lieux
        self.pheromones = self._initialiser_pheromones(route_initiale)

        with np.errstate(divide='ignore', invalid='ignore'):
            self.heuristique = np.where(self.graph.matrice_od > 0, 1.0 / self.graph.matrice_od, 0)
        np.fill_diagonal(self.heuristique, 0)

        self.eta_beta = self.heuristique ** self.beta
        self.tau_alpha = None
        self._update_tau_alpha()

        self.meilleure_route = None
        self.meilleure_distance = float('inf')
        self.historique_distances = []

    def _initialiser_pheromones(self, route_initiale):
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

    def _construire_solution(self):
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

    def _choisir_prochaine_ville_fast(self, ville_actuelle, visite):
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

    def optimiser(self, verbose=False, callback=None):
        start_time = time.time()
        for iteration in range(self.nb_iterations):
            if self.temps_max is not None and time.time() - start_time >= self.temps_max:
                if verbose: print(f"\nTemps max dépassé. Retour du meilleur résultat.")
                return self.meilleure_route, self.meilleure_distance

            tours = []
            for _ in range(self.nb_fourmis):
                if self.temps_max is not None and time.time() - start_time >= self.temps_max:
                    if verbose: print(f"\nTemps max dépassé pendant la construction.")
                    return self.meilleure_route, self.meilleure_distance

                ordre = self._construire_solution()
                distance = self.graph.calcul_distance_route(ordre)
                tours.append((ordre, distance))
                if distance < self.meilleure_distance:
                    self.meilleure_distance = distance
                    self.meilleure_route = ordre.copy()

            self._evaporation()
            self._depot_pheromones(tours)
            self.historique_distances.append(self.meilleure_distance)

            iteration_str = f"Itération {iteration + 1}/{self.nb_iterations}"

            if callback is not None and self.meilleure_route is not None:
                callback(self.meilleure_route, self.pheromones, iteration_str)

            if verbose:
                print(f"Itération {iteration+1}/{self.nb_iterations} - Meilleure distance : {self.meilleure_distance:.2f}")

        return self.meilleure_route, self.meilleure_distance


class Route:
    def __init__(self, graph, ordre=None):
        self.graph = graph
        if ordre is None:
            perm = list(range(1, graph.nb_lieux))
            random.shuffle(perm)
            ordre_gen = [0] + perm + [0]
            self.ordre = ordre_gen
        else:
            if ordre[0] != 0: ordre = [0] + ordre
            if ordre[-1] != 0: ordre = ordre + [0]
            self.ordre = ordre
        
        self.distance = self.calcul_distance()

    def calcul_distance(self):
        return self.graph.calcul_distance_route(self.ordre)

    def ameliorer_2opt(self, callback=None, temps_max=None):
        start_time = time.time()

        improved = True
        best_distance = self.distance
        best_ordre = self.ordre.copy()
        # Initialisation du compteur de passes 2-OPT (une passe correspond à une itération complète sur toutes les paires)
        passe_2opt = 0

        while improved:
            passe_2opt += 1
            improved = False
            for i in range(1, len(best_ordre) - 2):
                for j in range(i + 1, len(best_ordre) - 1):
                    if j - i == 1: continue

                    if temps_max is not None and (time.time() - start_time) > temps_max:
                        print(f"\nTemps max {temps_max}s atteint, arrêt prématuré")
                        self.ordre = best_ordre.copy()
                        self.distance = best_distance
                        return best_ordre, best_distance

                    new_ordre = best_ordre[:i] + best_ordre[i:j][::-1] + best_ordre[j:]
                    new_route = Route(self.graph, new_ordre)
                    new_distance = new_route.distance

                    if new_distance < best_distance:
                        best_ordre = new_ordre
                        best_distance = new_distance
                        improved = True
                        print(f"  -> Amélioration trouvée : {best_distance:.2f}")

                        if callback is not None:
                            callback(best_ordre, best_distance, passe_2opt)

            self.ordre = best_ordre.copy()
            self.distance = best_distance

        # Retourne le résultat final après convergence
        return best_ordre, best_distance

    def __repr__(self):
        return f"Route(dist={self.calcul_distance():.2f}, ordre={self.ordre})"
    
    @classmethod
    def from_sparse_grille(cls, graph):
        remaining = set(range(1, graph.nb_lieux))
        ordre = [0]
        current = 0

        while remaining:
            nxt = graph.plus_proche_voisin(current, remaining) 
            ordre.append(nxt)
            remaining.remove(nxt)
            current = nxt

        ordre.append(0)
        return cls(graph, ordre)


class Affichage(tk.Tk):
    def __init__(self, graph):
        super().__init__()
        self.graph = graph
        self.title("JC.Ilan C.Paul H.Ewen T.Lucas        Groupe 5")
        
        self.canvas = tk.Canvas(self, width=LARGEUR, height=HAUTEUR, bg='white')
        self.canvas.pack()
        
        self.text_zone = tk.Text(self, height=7, width=80, bg='lightgray', fg='black')
        self.text_zone.pack(fill='x')

        # Préparation zone STATIQUE
        self.text_zone.insert(tk.END, "Meilleur Score : N/A ")
        self.text_zone.tag_add("static_score", "1.0", "1.end")
        self.text_zone.tag_configure("static_score", font=('Helvetica', 10, 'bold'), background='lightblue')
        self.text_zone.insert(tk.END, "\n")
        
        self.log_start_index = "2.0" 

        self.simple_affichage = self.graph.nb_lieux > 200
        self.affiche_pheromones = False
        self.pheromones = None
        self.route = None

        self.bind('<Escape>', lambda e: self.destroy())
        self.bind('f', self.toggle_pheromones)

    
    def afficher_lieux(self, route):
        if self.simple_affichage: return
        
        self.canvas.delete("lieux", "ordres")
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
            self.canvas.create_line(a.x, a.y, b.x, b.y, fill='blue', dash=(6,6), width=2, tags="route")

    def afficher_pheromones_graph(self, pheromones):
        self.canvas.delete("pheromones")
        if not self.affiche_pheromones or pheromones is None: return
        MAX_WIDTH = 8
        SEUIL_RELATIF = 0.1
        max_ph = np.max(pheromones) if np.max(pheromones) > 0 else 1
        for i in range(self.graph.nb_lieux):
            for j in range(i+1, self.graph.nb_lieux):
                p = pheromones[i,j]
                if p < SEUIL_RELATIF * max_ph: continue
                a = self.graph.liste_lieux[i]
                b = self.graph.liste_lieux[j]
                width = max(1, (p / max_ph) * MAX_WIDTH)
                self.canvas.create_line(a.x, a.y, b.x, b.y, fill="lightpink", width=width, tags="pheromones")

    def update_affichage(self, route, pheromones=None):
        self.route = route
        self.pheromones = pheromones
        self.canvas.delete("all")
        self.afficher_lieux(route)
        if pheromones is not None and self.affiche_pheromones:
            self.afficher_pheromones_graph(pheromones)
        self.afficher_route(route)
        self.update_idletasks()
        self.update()

    def log(self, msg):
        self.text_zone.insert(tk.END, msg + "\n")
        self.text_zone.see(tk.END)

    def update_best_score(self, distance, iteration_info=""):
        new_text = f"Meilleur Score : {distance:.2f}    |    {iteration_info}"
        self.text_zone.delete("1.0", "1.end - 1c")
        self.text_zone.insert("1.0", new_text, "static_score")

    def toggle_pheromones(self, e=None):
        self.affiche_pheromones = not self.affiche_pheromones
        if self.route is not None:
            self.update_affichage(self.route, self.pheromones)


def run_optimization(g, aff, route_heur, dist_heur, temps_heur, tps_max):
    
    global meilleure_distance, meilleur_ordre
    meilleure_distance = float('inf')
    meilleur_ordre = route_heur.ordre.copy()
    pheromones_finales = None

    n = g.nb_lieux
    t2 = time.time()
    
    # PHASE 2 : ACO (N < SEUIL_BIG_DATA)
    if n < SEUIL_BIG_DATA:
        print("\n========== PHASE 2 : ACO ==========")
        aff.after(0, aff.log, "Démarrage ACO...")
        
        def aco_callback(route_ordre, pheromones, iteration_info):
            r = Route(g, route_ordre)
            aff.after(0, aff.update_affichage, r, pheromones)
            aff.after(0, aff.update_best_score, r.distance, iteration_info)

        aco = TSP_ACO(
            graph=g,
            nb_fourmis=100,
            nb_iterations=10000,
            alpha=1.0, beta=4.0, rho=0.3, Q=100.0,
            route_initiale=route_heur,
            temps_max=tps_max
        )
        meilleur_ordre, meilleure_distance = aco.optimiser(
            callback=aco_callback
        )
        route_finale = Route(g, meilleur_ordre)
        
        pheromones_finales = aco.pheromones

    # PHASE 2 : 2-OPT CLASSIQUE (N = SEUIL_BIG_DATA)
    elif n == SEUIL_BIG_DATA:
        print("\n========== PHASE 2 : 2-OPT CLASSIQUE ==========")
        aff.after(0, aff.log, "Démarrage 2-OPT Classique...")

        # Le callback reçoit (ordre, distance, passe_2opt)
        def callback_2opt(route_ordre, distance, passe_2opt):
            r = Route(g, route_ordre)
            aff.after(0, aff.update_affichage, r)
            iteration_info = f"Passe 2-OPT: {passe_2opt}"
            aff.after(0, aff.update_best_score, distance, iteration_info) 

        meilleur_ordre, meilleure_distance = route_heur.ameliorer_2opt(
            temps_max=tps_max,
            # Le callback doit être mis à jour pour accepter le 3ème argument
            callback=lambda r_ordre, dist, passe: callback_2opt(r_ordre, dist, passe)
        )
        route_finale = Route(g, meilleur_ordre)
        pheromones_finales = None

    # PHASE 2 : 2-OPT BIG DATA (N > SEUIL_BIG_DATA)
    else:
        print("\n========== PHASE 2 : 2-OPT BIG DATA ==========")
        aff.after(0, aff.log, "Démarrage 2-OPT Big Data...")

        # Le callback reçoit (ordre, distance, passe_2opt)
        def callback_2opt(route_ordre, distance, passe_2opt):
            r = Route(g, route_ordre)
            aff.after(0, aff.update_affichage, r)
            iteration_info = f"Passe 2-OPT: {passe_2opt}"
            aff.after(0, aff.update_best_score, distance, iteration_info)

        meilleur_ordre, meilleure_distance = route_heur.ameliorer_2opt(
            temps_max=tps_max,
            callback=lambda r_ordre, dist, passe: callback_2opt(r_ordre, dist, passe)
        )
        route_finale = Route(g, meilleur_ordre)
        pheromones_finales = None

    t3 = time.time()
    temps_opt = t3 - t2

    # AFFICHAGE FINAL
    improvement = (dist_heur - meilleure_distance) / dist_heur * 100
    print("\n========== COMPARAISON ==========")
    
    def afficher_final_secure(ph_final):
        aff.pheromones = ph_final
        aff.update_affichage(route_finale, pheromones=ph_final)
        aff.log(f"--- RÉSULTAT FINAL ---")
        aff.log(f"Distance finale : {meilleure_distance:.2f}  ({improvement:.2f}% d'amélioration)")
        aff.log(f"Temps heuristique : {temps_heur:.2f}s")
        aff.log(f"Temps total : {temps_heur + temps_opt:.2f}s")
        aff.update_best_score(meilleure_distance, "Terminé")

    aff.after(2000, afficher_final_secure, pheromones_finales)


if __name__ == '__main__':
    # Configuration initiale
    csv_file = CSV_FILE
    nb_lieux = 10
    tps_max = TPS_MAX

    g = Graph(csv_file=csv_file, nb_lieux=nb_lieux)
    g.calcul_matrice_cout_od() 

    n = g.nb_lieux
    t0 = time.time()
    
    # PHASE 1 : Heuristique PPV
    if n <= SEUIL_BIG_DATA:
        methode_heuristique = "ppv"
        route_heur = g.route_heuristique("ppv")
    else:
        methode_heuristique = "ppv_sparse_grille"
        
        # Logique PPV Big Data
        remaining = set(range(1, n))
        ordre = [0]
        curr = 0
        visite = np.zeros(n, dtype=bool)
        visite[0] = True
        for _ in range(n - 1):
            nxt = g.plus_proche_voisin(curr, remaining) 
            ordre.append(nxt)
            visite[nxt] = True
            remaining.remove(nxt)
            curr = nxt
        ordre.append(0)
        route_heur = Route(g, ordre)

    dist_heur = route_heur.calcul_distance()
    temps_heur = time.time() - t0

    aff = Affichage(g)
    
    aff.update_affichage(route_heur)
    aff.log(f"Heuristique {methode_heuristique.upper()} : {dist_heur:.2f} (base)")
    aff.update_best_score(dist_heur)
    
    optimization_thread = threading.Thread(
        target=run_optimization, 
        args=(g, aff, route_heur, dist_heur, temps_heur, tps_max)
    )

    optimization_thread.start()
    
    aff.mainloop()