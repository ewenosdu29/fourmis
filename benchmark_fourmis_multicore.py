import csv
import math
import random
import time
import sys
import os
import threading
import numpy as np
import tkinter as tk
from tkinter import scrolledtext
from concurrent.futures import ThreadPoolExecutor

# ============================================================
# 1. CONFIGURATION
# ============================================================
FICHIER_CSV = "graph_200k.csv"
LARGEUR = 1000
HAUTEUR = 800
NB_LIEUX = 200000  # MODE BIG DATA
SEUIL_BIG_DATA = 3000  # Au-delà, on active la grille et le sampling

# ============================================================
# 2. CLASSE LIEU
# ============================================================
class Lieu:
    def __init__(self, x, y, nom):
        self.x = float(x)
        self.y = float(y)
        self.nom = str(nom)

    def distance(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx*dx + dy*dy)
    
    def __repr__(self):
        return f"Lieu({self.nom})"

# ============================================================
# 3. CLASSE GRAPH (Optimisée Multi-Thread + Grille)
# ============================================================
class Graph:
    def __init__(self):
        self.liste_lieux = []
        self.matrice_od = None 
        self.is_sparse = False
        self.k_voisins = 20 # Nombre de voisins par ville

    def charger_graph(self, fichier_csv):
        self.liste_lieux = []
        print(f"Lecture de {fichier_csv}...")
        try:
            with open(fichier_csv, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                reader.fieldnames = [name.strip() for name in reader.fieldnames]
                for i, row in enumerate(reader):
                    if i >= NB_LIEUX: break
                    try:
                        x = float(row.get('x', row.get('lon', 0)))
                        y = float(row.get('y', row.get('lat', 0)))
                        self.liste_lieux.append(Lieu(x, y, row.get('nom', str(i))))
                    except: continue
            print(f"Graph chargé: {len(self.liste_lieux)} lieux.")
        except FileNotFoundError:
            self.generer_aleatoire()

    def generer_aleatoire(self):
        print(f"Génération interne de {NB_LIEUX} lieux...")
        # Génération vectorisée numpy
        coords = np.random.uniform(10, min(LARGEUR, HAUTEUR)-10, size=(NB_LIEUX, 2))
        self.liste_lieux = [Lieu(c[0], c[1], str(i)) for i, c in enumerate(coords)]

    def calcul_matrice_cout_od(self):
        """Calcule les distances. Si N est grand, utilise la Grille Spatiale."""
        n = len(self.liste_lieux)
        coords = np.array([[l.x, l.y] for l in self.liste_lieux], dtype=np.float32)
        
        if n <= SEUIL_BIG_DATA:
            print("Mode Petit Graphe: Matrice complète...")
            diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
            self.matrice_od = np.sqrt(np.sum(diff**2, axis=-1))
            self.is_sparse = False
        else:
            print(f"⚠️ Mode BIG DATA ({n} lieux). Indexation Spatiale Multi-Threadée...")
            self.is_sparse = True
            self.matrice_od = self._calculer_voisins_grille(coords, n)

    def _calculer_voisins_grille(self, coords, n):
        """
        Grille Spatiale Multi-Threadée.
        Divise la recherche des voisins sur tous les cœurs CPU.
        """
        t0 = time.time()
        grid_size = 50 
        grid = {}
        
        # 1. Remplissage Grille (Rapide en mono)
        print("   -> Remplissage de la grille (Bucketing)...")
        indices_grid_x = (coords[:, 0] // grid_size).astype(int)
        indices_grid_y = (coords[:, 1] // grid_size).astype(int)
        
        for idx in range(n):
            gx, gy = indices_grid_x[idx], indices_grid_y[idx]
            if (gx, gy) not in grid: grid[(gx, gy)] = []
            grid[(gx, gy)].append(idx)
            
        # 2. Recherche Voisins (Lourd -> Multi-thread)
        print(f"   -> Recherche des voisins (Multi-core)...")
        
        # Tableaux partagés (Thread-safe car écriture partitionnée)
        indices_voisins = np.zeros((n, self.k_voisins), dtype=int)
        dists_voisins = np.zeros((n, self.k_voisins), dtype=np.float32)
        
        def process_chunk(start_idx, end_idx):
            offsets = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,0), (0,1), (1,-1), (1,0), (1,1)]
            
            for idx in range(start_idx, end_idx):
                gx, gy = indices_grid_x[idx], indices_grid_y[idx]
                candidats = []
                
                for dx, dy in offsets:
                    key = (gx+dx, gy+dy)
                    if key in grid: candidats.extend(grid[key])
                
                cands_idx = np.array(candidats, dtype=int)
                
                # Fallback si case vide
                if len(cands_idx) <= self.k_voisins:
                     manquants = np.random.randint(0, n, self.k_voisins)
                     cands_idx = np.concatenate((cands_idx, manquants))

                # Calcul distance locale
                pts_cands = coords[cands_idx]
                pt_curr = coords[idx]
                d = np.sqrt(np.sum((pts_cands - pt_curr)**2, axis=1))
                
                # Top K
                k = min(len(d)-1, self.k_voisins + 1) 
                partition_idx = np.argpartition(d, k)[:k]
                best_local_indices = partition_idx[np.argsort(d[partition_idx])]
                
                raw_indices = cands_idx[best_local_indices]
                raw_dists = d[best_local_indices]
                
                # Masque soi-même
                mask = raw_indices != idx
                final_indices = raw_indices[mask][:self.k_voisins]
                final_dists = raw_dists[mask][:self.k_voisins]
                
                # Ecriture
                count = len(final_indices)
                indices_voisins[idx, :count] = final_indices
                dists_voisins[idx, :count] = final_dists

        # Lancement des threads
        num_threads = max(1, os.cpu_count() - 1)
        chunk_size = n // num_threads
        futures = []
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for i in range(num_threads):
                start = i * chunk_size
                end = n if i == num_threads - 1 else (i + 1) * chunk_size
                futures.append(executor.submit(process_chunk, start, end))
        
        print(f"✅ Indexation terminée en {time.time()-t0:.2f}s")
        return (indices_voisins, dists_voisins)

    def calcul_distance_route(self, route):
        coords = np.array([[l.x, l.y] for l in self.liste_lieux], dtype=np.float32)
        ordre = np.array(route.ordre)
        pts_a = coords[ordre[:-1]]
        pts_b = coords[ordre[1:]]
        return float(np.sum(np.sqrt(np.sum((pts_a - pts_b)**2, axis=1))))

    def plus_proche_voisin(self, index):
        if self.is_sparse: return self.matrice_od[0][index][0]
        dists = self.matrice_od[index].copy()
        dists[index] = np.inf
        return np.argmin(dists)

# ============================================================
# 4. CLASSE ROUTE
# ============================================================
class Route:
    def __init__(self, graph, ordre=None):
        self.graph = graph
        if ordre: self.ordre = ordre
        else: self.ordre = list(range(len(graph.liste_lieux))) + [0]
        
        if self.ordre[0] != 0: self.ordre = [0] + self.ordre
        if self.ordre[-1] != 0: self.ordre.append(0)

# ============================================================
# 5. CLASSE AFFICHAGE (Avec Sampling)
# ============================================================
class Affichage:
    def __init__(self, graph, group_name="TSP Multi-Thread"):
        self.graph = graph
        self.root = tk.Tk()
        self.root.title(f"{group_name} - {len(graph.liste_lieux)} Villes")
        self.canvas = tk.Canvas(self.root, width=LARGEUR, height=HAUTEUR, bg="black")
        self.canvas.pack()
        self.text_area = scrolledtext.ScrolledText(self.root, height=8)
        self.text_area.pack(fill=tk.BOTH)
        self.root.bind('<Escape>', lambda e: self.root.destroy())
        
        if len(graph.liste_lieux) < 2000: self.dessiner_lieux()

    def dessiner_lieux(self):
        self.canvas.delete("lieu")
        for l in self.graph.liste_lieux:
            self.canvas.create_oval(l.x-2, l.y-2, l.x+2, l.y+2, fill="gray", tags="lieu")

    def afficher_route(self, route, couleur="cyan"):
        self.canvas.delete("route")
        n = len(route.ordre)
        step = 1
        if n > 5000: step = 10
        if n > 50000: step = 100
        
        indices = route.ordre[::step]
        if indices[-1] != 0: indices.append(0)
        
        coords = np.array([[l.x, l.y] for l in self.graph.liste_lieux], dtype=np.float32)
        pts = coords[indices].flatten().tolist()
        
        self.canvas.create_line(pts, fill=couleur, width=1, tags="route")
        self.root.update()

    def log(self, msg):
        self.text_area.insert(tk.END, msg + "\n")
        self.text_area.see(tk.END)

# ============================================================
# 6. CLASSE TSP_ACO (Multi-Thread)
# ============================================================
class TSP_ACO:
    def __init__(self, graph, affichage, nb_fourmis=20, alpha=1.0, beta=3.0, rho=0.1):
        self.graph = graph
        self.app = affichage
        self.n = len(graph.liste_lieux)
        self.nb_fourmis = nb_fourmis
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = 1000.0
        
        # Init Phéromones
        if self.graph.is_sparse:
            self.k = self.graph.k_voisins
            self.pheromones = np.ones((self.n, self.k), dtype=np.float32) * 0.1
            indices, distances = self.graph.matrice_od
            with np.errstate(divide='ignore'):
                self.heuristique = 1.0 / (distances + 1e-9)
        else:
            self.pheromones = np.ones((self.n, self.n)) * 0.1
            with np.errstate(divide='ignore'):
                self.heuristique = 1.0 / (self.graph.matrice_od + 1e-9)

    def get_greedy_route(self):
        """Baseline PPV."""
        visite = np.zeros(self.n, dtype=bool)
        tour = np.zeros(self.n + 1, dtype=int)
        curr = 0
        visite[0] = True
        indices_voisins, _ = self.graph.matrice_od
        
        for i in range(1, self.n):
            local = indices_voisins[curr]
            found = False
            for v in local:
                if not visite[v]:
                    curr = v; found = True; break
            if not found:
                while True:
                    cand = random.randint(0, self.n-1)
                    if not visite[cand]: curr = cand; break
            tour[i] = curr
            visite[curr] = True
        return list(tour)

    def _tache_fourmi(self, seed):
        """Tâche unitaire pour un thread."""
        local_rng = random.Random(time.time() + seed)
        tour = [0]
        visite = set([0])
        curr = 0
        indices_voisins, _ = self.graph.matrice_od
        
        for _ in range(self.n - 1):
            candidats_idx = indices_voisins[curr]
            # Filtre rapide
            valid_tuple = [(k, idx) for k, idx in enumerate(candidats_idx) if idx not in visite]
            
            if valid_tuple:
                valid_local = [t[0] for t in valid_tuple]
                valid_real = [t[1] for t in valid_tuple]
                
                ph = self.pheromones[curr][valid_local]
                he = self.heuristique[curr][valid_local]
                probas = (ph ** self.alpha) * (he ** self.beta)
                
                # Choix stochastique rapide
                if local_rng.random() < 0.9:
                    nxt = valid_real[np.argmax(probas)]
                else:
                    nxt = valid_real[local_rng.randint(0, len(valid_real)-1)]
            else:
                while True:
                    cand = local_rng.randint(0, self.n-1)
                    if cand not in visite: nxt = cand; break
            
            tour.append(nxt)
            visite.add(nxt)
            curr = nxt
            
        tour.append(0)
        
        # Calcul distance (Thread safe)
        r = Route(self.graph, tour)
        d = self.graph.calcul_distance_route(r)
        return (tour, d)

    def resoudre(self, nb_iter=50):
        print("--- CALCUL PPV ---")
        t0 = time.time()
        ppv_ordre = self.get_greedy_route()
        t_ppv = time.time() - t0
        
        best_route = Route(self.graph, ppv_ordre)
        best_dist = self.graph.calcul_distance_route(best_route)
        
        print(f"⏱️ Temps PPV : {t_ppv:.4f} s")
        print(f"📏 Distance PPV : {best_dist:.2f}")
        self.app.log(f"PPV: {best_dist:.2f} ({t_ppv:.2f}s)")
        self.app.afficher_route(best_route, "red")
        
        # Injection Phéromones
        indices_voisins, _ = self.graph.matrice_od
        arr_t = np.array(ppv_ordre)
        for i in range(self.n):
            u, v = arr_t[i], arr_t[i+1]
            idx = np.where(indices_voisins[u] == v)[0]
            if idx.size > 0: self.pheromones[u, idx[0]] += 5.0

        # Lancement ACO Multi-Thread
        n_workers = max(1, os.cpu_count() - 1)
        print(f"Lancement sur {n_workers} threads...")
        
        for it in range(nb_iter):
            routes = []
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(self._tache_fourmi, i) for i in range(self.nb_fourmis)]
                for f in futures:
                    try:
                        o, d = f.result()
                        routes.append((o, d))
                        if d < best_dist:
                            best_dist = d
                            best_route = Route(self.graph, o)
                            self.app.log(f"RECORD: {d:.2f}")
                            # Maj GUI différée
                    except Exception as e: print(e)
            
            self.app.afficher_route(best_route, "cyan")
            
            # Evaporation
            self.pheromones *= (1 - self.rho)
            # Renforcement
            for o, d in routes:
                delta = self.Q / d
                if d <= best_dist * 1.05:
                    for i in range(self.n):
                        u, v = o[i], o[i+1]
                        idx = np.where(indices_voisins[u] == v)[0]
                        if idx.size > 0: self.pheromones[u, idx[0]] += delta
            
            self.app.log(f"Iter {it+1} | Best: {best_dist:.0f}")
            self.app.root.update()

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    g = Graph()
    g.charger_graph(FICHIER_CSV)
    g.calcul_matrice_cout_od()
    
    app = Affichage(g)
    # Plus de fourmis possibles grâce aux threads !
    solver = TSP_ACO(g, app, nb_fourmis=20) 
    
    app.root.after(100, lambda: solver.resoudre(nb_iter=1000))
    app.root.mainloop()