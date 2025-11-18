import csv
import math
import random
import time
import sys
import os
import numpy as np
import tkinter as tk
from tkinter import scrolledtext, Text
from concurrent.futures import ThreadPoolExecutor

# ============================================================
# 1. CONFIGURATION
# ============================================================
FICHIER_CSV = "graph_200k.csv"
LARGEUR = 1000
HAUTEUR = 800
NB_LIEUX = 100  # BIG DATA
SEUIL_BIG_DATA = 2000  # Seuil pour passer en mode optimisation RAM

# ============================================================
# 2. CLASSE LIEU
# ============================================================
class Lieu:
    """Représentation d'un lieu (x,y) avec un nom."""
    def __init__(self, x: float, y: float, name: str = None):
        self.x = float(x)
        self.y = float(y)
        self.name = str(name) if name is not None else ""

    def distance(self, other: "Lieu") -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)

    def __repr__(self):
        return f"Lieu(name={self.name!r}, x={self.x:.2f}, y={self.y:.2f})"

# ============================================================
# 3. CLASSE GRAPH (Moteur Big Data caché sous le capot)
# ============================================================
class Graph:
    """Graphe de lieux optimisé pour le Big Data."""
    def __init__(self, nb_lieux: int = NB_LIEUX, largeur: int = LARGEUR, hauteur: int = HAUTEUR, csv_file: str = None):
        self.largeur = largeur
        self.hauteur = hauteur
        self.nb_lieux = int(nb_lieux)
        self.liste_lieux = []
        self.matrice_od = None
        
        # Optimisations internes
        self.coords_np = None
        self.is_sparse = False
        self.k_voisins = 20

        if csv_file is not None:
            print(f"Chargement des lieux depuis le fichier CSV : {csv_file}")
            self.charger_lieux_depuis_csv(csv_file)
        else:
            print(f"Génération de {nb_lieux} lieux aléatoires")
            self.generer_lieux_aleatoires(self.nb_lieux)

    def charger_lieux_depuis_csv(self, chemin_fichier: str):
        self.liste_lieux = []
        try:
            with open(chemin_fichier, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                # Nettoyage en-têtes
                reader.fieldnames = [name.strip() for name in reader.fieldnames]
                for i, row in enumerate(reader):
                    if i >= NB_LIEUX: break
                    try:
                        # Supporte format x,y ou lat,lon
                        x = float(row.get('x', row.get('lon', 0)))
                        y = float(row.get('y', row.get('lat', 0)))
                        name = row.get('nom', row.get('name', str(i)))
                        self.liste_lieux.append(Lieu(x, y, name=name))
                    except: continue
            self.nb_lieux = len(self.liste_lieux)
        except FileNotFoundError:
            self.generer_lieux_aleatoires(NB_LIEUX)

    def generer_lieux_aleatoires(self, nb: int):
        self.liste_lieux = []
        # Génération vectorisée rapide
        coords = np.random.uniform(10, min(self.largeur, self.hauteur)-10, size=(nb, 2))
        for i in range(nb):
            self.liste_lieux.append(Lieu(coords[i][0], coords[i][1], name=str(i)))
        self.nb_lieux = len(self.liste_lieux)

    def calcul_matrice_cout_od(self):
        """
        Calcule la matrice. Si N est grand, génère une grille spatiale
        pour éviter le crash RAM (méthode sparse).
        """
        n = self.nb_lieux
        # Cache Numpy indispensable pour la vitesse
        self.coords_np = np.array([[l.x, l.y] for l in self.liste_lieux], dtype=np.float32)

        if n <= SEUIL_BIG_DATA:
            print("Calcul Matrice Complète (Petit Graphe)...")
            diff = self.coords_np[:, np.newaxis, :] - self.coords_np[np.newaxis, :, :]
            self.matrice_od = np.sqrt(np.sum(diff**2, axis=-1))
            self.is_sparse = False
        else:
            print(f"⚠️ Mode BIG DATA ({n} lieux). Indexation Spatiale (Grille)...")
            self.is_sparse = True
            # On remplace la matrice par un tuple (indices, distances) des voisins
            self.matrice_od = self._calculer_voisins_grille(self.coords_np, n)

    def _calculer_voisins_grille(self, coords, n):
        """Grille spatiale Multi-Threadée pour trouver les voisins sans Scipy."""
        t0 = time.time()
        grid_size = 50 
        grid = {}
        
        # 1. Bucketing
        ix = (coords[:, 0] // grid_size).astype(int)
        iy = (coords[:, 1] // grid_size).astype(int)
        for i in range(n):
            k = (ix[i], iy[i])
            if k not in grid: grid[k] = []
            grid[k].append(i)
            
        # 2. Recherche Voisins (Multi-Thread)
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
                    # Fallback rare
                    cands_idx = np.random.randint(0, n, self.k_voisins+1)
                
                # Distances
                pts_c = coords[cands_idx]
                pt_i = coords[i]
                d = np.sqrt(np.sum((pts_c - pt_i)**2, axis=1))
                
                # Top K
                k_take = min(len(d)-1, self.k_voisins + 1)
                # argpartition est O(N)
                part_idx = np.argpartition(d, k_take)[:k_take+1]
                sorted_idx = part_idx[np.argsort(d[part_idx])]
                
                raw_idx = cands_idx[sorted_idx]
                # Exclure soi-même
                final = raw_idx[raw_idx != i][:self.k_voisins]
                indices_voisins[i, :len(final)] = final

        n_cpu = max(1, os.cpu_count() - 1)
        chunk = n // n_cpu
        with ThreadPoolExecutor(max_workers=n_cpu) as exe:
            for i in range(n_cpu):
                s = i * chunk
                e = n if i == n_cpu - 1 else (i+1) * chunk
                exe.submit(process_chunk, s, e)
                
        print(f"✅ Indexation terminée ({time.time()-t0:.2f}s)")
        return (indices_voisins, None) # On ne garde que les indices pour économiser RAM

    def plus_proche_voisin(self, index, remaining=None):
        """Adaptatif : utilise la grille si dispo, sinon matrice."""
        if self.matrice_od is None: self.calcul_matrice_cout_od()
        
        if self.is_sparse:
            # Mode Grille
            voisins_possibles, _ = self.matrice_od
            candidats = voisins_possibles[index]
            # On cherche le premier non visité
            # remaining est un set
            for c in candidats:
                if c in remaining: return int(c)
            # Secours : le premier de remaining (lent mais sûr)
            return next(iter(remaining))
        else:
            # Mode Matrice
            row = self.matrice_od[index]
            rem_list = np.array(list(remaining))
            dists = row[rem_list]
            return int(rem_list[np.argmin(dists)])

    def calcul_distance_route(self, ordre: list) -> float:
        if not ordre: return 0.0
        # Utilisation de Numpy pour vitesse max
        if self.coords_np is None: self.calcul_matrice_cout_od()
        
        o = np.array(ordre)
        pts_a = self.coords_np[o[:-1]]
        pts_b = self.coords_np[o[1:]]
        return float(np.sum(np.sqrt(np.sum((pts_a - pts_b)**2, axis=1))))

# ============================================================
# 4. CLASSE ROUTE
# ============================================================
class Route:
    """Représente une route."""
    def __init__(self, graph, ordre=None):
        self.graph = graph
        self.ordre = ordre if ordre else list(range(graph.nb_lieux)) + [0]
        
        if self.ordre[0] != 0: self.ordre = [0] + self.ordre
        if self.ordre[-1] != 0: self.ordre.append(0)

    def calcul_distance(self) -> float:
        return self.graph.calcul_distance_route(self.ordre)

    def ameliorer_2opt(self, temps_max=60, callback=None):
        """
        Version Multi-Threadée du 2-Opt.
        Optimise la route en parallèle pour tenir le temps imparti.
        """
        start_time = time.time()
        best_ordre = np.array(self.ordre)
        best_dist = self.calcul_distance()
        
        coords = self.graph.coords_np
        n_cpu = max(1, os.cpu_count() - 1)
        n_pts = len(best_ordre) - 1
        
        # Fonction interne exécutée par les threads
        def worker_2opt(current_tour_arr, seed):
            rng = np.random.default_rng(seed)
            tour = current_tour_arr.copy()
            
            # Stochastic 2-opt (Tests aléatoires massifs)
            # C'est la seule méthode viable pour > 10k villes
            iterations = 200000 # nb de swaps testés par thread
            
            idx_i = rng.integers(1, n_pts - 1, iterations)
            offset = rng.integers(1, 1000, iterations) # Local + Global search
            idx_j = (idx_i + offset) % n_pts
            
            # Filtre indices valides
            mask = (idx_j > idx_i) & (idx_j - idx_i > 0)
            idx_i = idx_i[mask]
            idx_j = idx_j[mask]
            
            for k in range(len(idx_i)):
                i, j = idx_i[k], idx_j[k]
                
                # Indices points
                a, b = tour[i-1], tour[i]
                c, d = tour[j], tour[j+1]
                
                # Coords
                pa, pb = coords[a], coords[b]
                pc, pd = coords[c], coords[d]
                
                # Gain distance (carré suffisant pour comparer)
                d_curr = (pa[0]-pb[0])**2 + (pa[1]-pb[1])**2 + (pc[0]-pd[0])**2 + (pc[1]-pd[1])**2
                d_new  = (pa[0]-pc[0])**2 + (pa[1]-pc[1])**2 + (pb[0]-pd[0])**2 + (pb[1]-pd[1])**2
                
                if d_new < d_curr:
                    tour[i:j+1] = tour[i:j+1][::-1] # Swap
            
            # Recalcul distance finale du thread
            pts_a = coords[tour[:-1]]
            pts_b = coords[tour[1:]]
            f_dist = float(np.sum(np.sqrt(np.sum((pts_a - pts_b)**2, axis=1))))
            
            return list(tour), f_dist

        # Boucle principale d'optimisation
        iteration = 0
        while (time.time() - start_time) < temps_max:
            iteration += 1
            
            # Lancement des threads
            with ThreadPoolExecutor(max_workers=n_cpu) as executor:
                futures = [executor.submit(worker_2opt, best_ordre, int(time.time()*1000)+i) 
                           for i in range(n_cpu)]
                
                improved = False
                for f in futures:
                    try:
                        o, d = f.result()
                        if d < best_dist:
                            best_dist = d
                            best_ordre = np.array(o)
                            improved = True
                    except: pass
            
            if improved and callback:
                callback(list(best_ordre), best_dist)
                
            if not improved:
                # Si on stagne, on peut sortir ou continuer pour chercher par chance
                pass

        self.ordre = list(best_ordre)
        return self.ordre, best_dist

# ============================================================
# 5. CLASSE AFFICHAGE (EXACTEMENT CELLE DEMANDÉE)
# ============================================================
class Affichage(tk.Tk):
    def __init__(self, graph):
        super().__init__()
        self.graph = graph
        self.title("JC.Ilan C.Paul H.Ewen T.Lucas        Groupe 5")
        self.canvas = tk.Canvas(self, width=LARGEUR, height=HAUTEUR, bg='white')
        self.canvas.pack()
        self.text_zone = tk.Text(self, height=5, width=80, bg='lightyellow')
        self.text_zone.pack()

        # Adaptation Big Data pour l'affichage
        self.simple_affichage = self.graph.nb_lieux > 2000
        self.affiche_pheromones = False
        self.pheromones = None
        self.route = None

        self.bind('<Escape>', lambda e: self.destroy())
        self.bind('f', self.toggle_pheromones)

    def afficher_lieux(self, route):
        if self.simple_affichage: return
        
        self.canvas.delete("lieux")
        self.canvas.delete("ordres")
        for ordre_idx, lieu_idx in enumerate(route.ordre[:-1]):
            lieu = self.graph.liste_lieux[lieu_idx]
            couleur = "red" if lieu_idx == 0 else "#D3D3D3"
            r = 8
            self.canvas.create_oval(lieu.x-r, lieu.y-r, lieu.x+r, lieu.y+r,
                                    fill=couleur, outline="black", tags="lieux")
            self.canvas.create_text(lieu.x, lieu.y, text=str(lieu_idx), tags="lieux")
            self.canvas.create_text(lieu.x, lieu.y-14, text=str(ordre_idx), fill="gray", tags="ordres")

    def afficher_route(self, route):
        self.canvas.delete("route")
        
        # Sampling pour éviter freeze si > 2000 points
        n = len(route.ordre)
        step = 1
        if n > 5000: step = 10
        if n > 50000: step = 100
        
        indices = route.ordre[::step]
        if indices[-1] != 0: indices.append(0)
        
        # Utilisation coords_np pour récupérer points rapidement
        if self.graph.coords_np is not None:
            pts = self.graph.coords_np[indices].flatten().tolist()
        else:
            pts = []
            for idx in indices:
                l = self.graph.liste_lieux[idx]
                pts.extend([l.x, l.y])
            
        self.canvas.create_line(pts, fill='blue', width=2, tags="route")

    def update_affichage(self, route, pheromones=None):
        self.route = route
        self.pheromones = pheromones
        self.canvas.delete("route") # Clean juste la route pour fluidité
        if not self.simple_affichage and not self.canvas.find_withtag("lieux"):
             self.afficher_lieux(route)
        self.afficher_route(route)
        self.update()

    def log(self, msg):
        self.text_zone.insert(tk.END, msg + "\n")
        self.text_zone.see(tk.END)

    def toggle_pheromones(self, e=None):
        pass # Pas de phéromones en mode 2-Opt

# =========================================
# PROGRAMME PRINCIPAL
# =========================================
if __name__ == '__main__':
    csv_file = FICHIER_CSV if os.path.exists(FICHIER_CSV) else None
    nb_lieux = NB_LIEUX
    tps_max = 60 # Temps max optimisation

    # Création du graphe
    if csv_file is not None:
        g = Graph(csv_file=csv_file)
    else:
        g = Graph(nb_lieux=nb_lieux)

    # Pré-calculs indispensables
    g.calcul_matrice_cout_od()

    # ===============================
    #   PHASE 1 : Heuristique (PPV)
    # ===============================
    methode_heuristique = "ppv"
    print(f"\n========== PHASE 1 : MÉTHODE HEURISTIQUE ({methode_heuristique.upper()}) ==========")
    t0 = time.time()

    # On construit le PPV manuellement pour utiliser la grille optimisée
    n = g.nb_lieux
    remaining = set(range(1, n))
    ordre = [0]
    curr = 0
    # Optimisation: conversion set en bool array pour vitesse si N grand
    if n > 5000:
        # Version Array (Ultra-rapide pour Big Data)
        visite = np.zeros(n, dtype=bool)
        visite[0] = True
        tour = np.zeros(n + 1, dtype=int)
        # On utilise l'accès direct à la grille (plus_proche_voisin est adaptatif)
        for i in range(1, n):
            nxt = g.plus_proche_voisin(curr, remaining) # remaining ignoré en mode sparse interne
            # Mise à jour manuelle du masque interne si nécessaire, 
            # mais ici on triche un peu: get_greedy du Graph précédent était mieux
            # On réutilise get_greedy logic simplifié ici:
            indices, _ = g.matrice_od
            local = indices[curr]
            found = False
            for v in local:
                if not visite[v]:
                    nxt = v; found = True; break
            if not found:
                while True:
                    c = random.randint(0, n-1)
                    if not visite[c]: nxt = c; break
            
            tour[i] = nxt
            visite[nxt] = True
            curr = nxt
        ordre = list(tour)
    else:
        # Version Set (Standard pour petits graphes)
        for _ in range(n - 1):
            nxt = g.plus_proche_voisin(curr, remaining)
            ordre.append(nxt)
            remaining.remove(nxt)
            curr = nxt
        ordre.append(0)

    route_heur = Route(g, ordre)
    dist_heur = route_heur.calcul_distance()
    temps_heur = time.time() - t0

    print(f"Distance obtenue : {dist_heur:.2f}")
    print(f"Temps d'exécution : {temps_heur:.3f} s")

    # ===============================
    #   INITIALISATION AFFICHAGE
    # ===============================
    aff = Affichage(g)
    aff.update_affichage(route_heur)
    aff.log(f"Heuristique {methode_heuristique.upper()} : {dist_heur:.2f}")

    # ===============================
    #   PHASE 2 : Optimisation (2-Opt)
    # ===============================
    print("\n========== PHASE 2 : 2OPT MULTI-THREAD ==========")
    t2 = time.time()

    def callback_2opt(route_ordre, distance):
        r = Route(g, route_ordre)
        aff.update_affichage(r)
        aff.log(f"2OPT Record : {distance:.2f}")

    meilleur_ordre, meilleure_distance = route_heur.ameliorer_2opt(
        temps_max=tps_max,
        callback=callback_2opt
    )

    temps_aco = time.time() - t2
    route_finale = Route(g, meilleur_ordre)
    aff.update_affichage(route_finale)
    aff.log(f"2OPT final : {meilleure_distance:.2f}")

    # ===============================
    #   COMPARAISON
    # ===============================
    improvement = (dist_heur - meilleure_distance) / dist_heur * 100
    print("\n========== COMPARAISON ==========")
    print(f"📏 Distance Initiale : {dist_heur:.2f}")
    print(f"📏 Distance Finale : {meilleure_distance:.2f}")
    print(f"⚡ Amélioration : {improvement:.2f}%")

    aff.mainloop()