import csv
import math
import random
import time
import sys
import numpy as np
import tkinter as tk
from tkinter import scrolledtext

# ============================================================
# CONFIGURATION
# ============================================================
FICHIER_CSV = "graph_200k.csv"
LARGEUR = 1000
HAUTEUR = 800
NB_LIEUX = 200000  # MODE BIG DATA ACTIVÉ

# ============================================================
# CLASSE LIEU
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
# CLASSE GRAPH (Optimisée Grille Spatiale pour RAM)
# ============================================================
class Graph:
    def __init__(self):
        self.liste_lieux = []
        # matrice_od stockera un tuple (indices_voisins, distances_voisins)
        # pour éviter de saturer la RAM avec une matrice pleine
        self.matrice_od = None 
        self.is_sparse = False
        self.k_voisins = 40 # Nombre de voisins regardés par les fourmis

    def charger_graph(self, fichier_csv):
        self.liste_lieux = []
        print(f"Chargement de {fichier_csv}...")
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
            print("Fichier absent. Génération aléatoire...")
            self.generer_aleatoire()

    def generer_aleatoire(self):
        print(f"Génération aléatoire de {NB_LIEUX} lieux...")
        self.liste_lieux = []
        # Génération vectorisée rapide
        coords = np.random.uniform(10, min(LARGEUR, HAUTEUR)-10, size=(NB_LIEUX, 2))
        for i in range(NB_LIEUX):
            self.liste_lieux.append(Lieu(coords[i][0], coords[i][1], str(i)))

    def calcul_matrice_cout_od(self):
        """
        Prépare les distances. Si N est grand, utilise une Grille Spatiale
        pour trouver les voisins sans calculer la matrice géante (Crash RAM).
        """
        n = len(self.liste_lieux)
        coords = np.array([[l.x, l.y] for l in self.liste_lieux], dtype=np.float32)
        
        if n <= 2000:
            print("Calcul Matrice Complète (Mode Petit Graphe)...")
            diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
            self.matrice_od = np.sqrt(np.sum(diff**2, axis=-1))
            self.is_sparse = False
        else:
            print(f"⚠️ MODE BIG DATA ({n} lieux). Indexation spatiale en cours...")
            self.is_sparse = True
            
            # 1. Grille de hachage (Spatial Hashing)
            # Permet de trouver les voisins sans tout parcourir
            grid_size = 50 
            grid = {}
            for idx, (x, y) in enumerate(coords):
                gx, gy = int(x // grid_size), int(y // grid_size)
                if (gx, gy) not in grid: grid[(gx, gy)] = []
                grid[(gx, gy)].append(idx)
                
            # 2. Construction de la liste des voisins (Candidate List)
            indices_voisins = np.zeros((n, self.k_voisins), dtype=int)
            dists_voisins = np.zeros((n, self.k_voisins), dtype=np.float32)
            
            t0 = time.time()
            for idx, (x, y) in enumerate(coords):
                if idx % 10000 == 0: print(f"   Indexation {idx}/{n}...", end="\r")
                
                gx, gy = int(x // grid_size), int(y // grid_size)
                candidats = []
                
                # Recherche dans les cases adjacentes
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        key = (gx+dx, gy+dy)
                        if key in grid: candidats.extend(grid[key])
                
                candidats = np.array(candidats)
                # Fallback si case vide (rare)
                if len(candidats) <= self.k_voisins:
                    candidats = np.arange(n) 
                    
                pts_cand = coords[candidats]
                # Distance euclidienne vectorisée
                dists = np.sqrt(np.sum((pts_cand - coords[idx])**2, axis=1))
                
                # On prend les K+1 meilleurs
                nb_take = min(len(dists)-1, self.k_voisins + 1)
                # argpartition est O(N), beaucoup plus rapide que sort
                k_best_idx = np.argpartition(dists, nb_take)[:nb_take+1]
                
                # Tri propre des K meilleurs
                sorted_local = k_best_idx[np.argsort(dists[k_best_idx])]
                
                # Récupération des indices globaux et distances
                raw_idx = candidats[sorted_local]
                raw_dst = dists[sorted_local]
                
                # Masque pour s'exclure soi-même (distance 0)
                mask = (raw_idx != idx)
                final_idx = raw_idx[mask][:self.k_voisins]
                final_dst = raw_dst[mask][:self.k_voisins]
                
                # Remplissage
                nb = len(final_idx)
                indices_voisins[idx, :nb] = final_idx
                dists_voisins[idx, :nb] = final_dst
                
            self.matrice_od = (indices_voisins, dists_voisins)
            print(f"\n✅ Indexation terminée en {time.time()-t0:.1f}s.")

    def calcul_distance_route(self, route):
        # Calcul optimisé avec numpy
        coords = np.array([[l.x, l.y] for l in self.liste_lieux], dtype=np.float32)
        ordre = np.array(route.ordre)
        pts_a = coords[ordre[:-1]]
        pts_b = coords[ordre[1:]]
        return float(np.sum(np.sqrt(np.sum((pts_a - pts_b)**2, axis=1))))

# ============================================================
# CLASSE ROUTE
# ============================================================
class Route:
    def __init__(self, graph, ordre=None):
        self.graph = graph
        if ordre:
            self.ordre = ordre
        else:
            self.ordre = list(range(len(graph.liste_lieux))) + [0]
        
        if self.ordre[0] != 0: self.ordre = [0] + self.ordre
        if self.ordre[-1] != 0: self.ordre.append(0)

# ============================================================
# CLASSE AFFICHAGE
# ============================================================
class Affichage:
    def __init__(self, graph, group_name="TSP ACO"):
        self.graph = graph
        self.root = tk.Tk()
        self.root.title(f"{group_name} ({len(graph.liste_lieux)} villes)")
        self.canvas = tk.Canvas(self.root, width=LARGEUR, height=HAUTEUR, bg="black")
        self.canvas.pack()
        self.text_area = scrolledtext.ScrolledText(self.root, height=8)
        self.text_area.pack(fill=tk.BOTH)
        self.root.bind('<Escape>', lambda e: self.root.destroy())
        
        # Pas de ronds si trop de villes (sinon lag)
        if len(graph.liste_lieux) < 2000:
            self.dessiner_lieux()

    def dessiner_lieux(self):
        for l in self.graph.liste_lieux:
            self.canvas.create_oval(l.x-2, l.y-2, l.x+2, l.y+2, fill="gray", outline="")

    def afficher_route(self, route, color="cyan"):
        self.canvas.delete("route")
        # Sampling pour affichage fluide si > 5000 points
        step = 1
        if len(route.ordre) > 10000: step = 20
        if len(route.ordre) > 100000: step = 100
        
        pts = []
        coords = [[l.x, l.y] for l in self.graph.liste_lieux]
        indices = route.ordre[::step]
        if indices[-1] != 0: indices.append(0)
        
        for idx in indices:
            pts.extend(coords[idx])
            
        self.canvas.create_line(pts, fill=color, width=1, tags="route")
        self.root.update()

    def log(self, msg):
        self.text_area.insert(tk.END, msg + "\n")
        self.text_area.see(tk.END)

# ============================================================
# CLASSE TSP_ACO (Algorithme)
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
        
        # Initialisation Phéromones (Matrice Creuse)
        # On stocke une valeur pour chaque voisin connu
        self.k = self.graph.k_voisins
        self.pheromones = np.ones((self.n, self.k), dtype=np.float32) * 0.5
        
        # Pré-calcul Heuristique (1/d) pour les voisins
        indices, distances = self.graph.matrice_od
        with np.errstate(divide='ignore'):
            self.heuristique = 1.0 / (distances + 1e-9)

    def get_ppv_route(self):
        """Construit la route Plus Proche Voisin."""
        visite = np.zeros(self.n, dtype=bool)
        tour = np.zeros(self.n + 1, dtype=int)
        curr = 0
        visite[0] = True
        
        # On utilise la matrice creuse des voisins
        voisins_idx, _ = self.graph.matrice_od
        
        for i in range(1, self.n):
            local_neighbors = voisins_idx[curr]
            found = False
            for v in local_neighbors:
                if not visite[v]:
                    curr = v
                    found = True
                    break
            if not found: # Secours aléatoire
                while True:
                    cand = random.randint(0, self.n-1)
                    if not visite[cand]:
                        curr = cand
                        break
            tour[i] = curr
            visite[curr] = True
            
        return list(tour)

    def construire_solution(self):
        """Une fourmi construit un chemin (Logique ACO)."""
        tour = [0]
        visite = set([0])
        curr = 0
        
        indices_voisins, _ = self.graph.matrice_od
        
        for _ in range(self.n - 1):
            candidats_idx = indices_voisins[curr]
            
            # Filtre non visités
            valid_local_idx = []
            valid_real_idx = []
            for k, real_idx in enumerate(candidats_idx):
                if real_idx not in visite:
                    valid_local_idx.append(k)
                    valid_real_idx.append(real_idx)
            
            if valid_real_idx:
                # Formule ACO
                ph = self.pheromones[curr][valid_local_idx]
                he = self.heuristique[curr][valid_local_idx]
                probas = (ph ** self.alpha) * (he ** self.beta)
                
                # Choix pondéré (Argmax avec bruit pour vitesse)
                choice = np.argmax(probas * np.random.uniform(0.8, 1.2, size=len(probas)))
                nxt = valid_real_idx[choice]
            else:
                # Secours
                while True:
                    cand = random.randint(0, self.n-1)
                    if cand not in visite:
                        nxt = cand
                        break
            
            tour.append(nxt)
            visite.add(nxt)
            curr = nxt
            
        tour.append(0)
        return tour

    def resoudre(self, nb_iter=100):
        # 1. CALCUL ET AFFICHAGE PPV (DEMANDÉ)
        print("--- CALCUL PPV ---")
        t0 = time.time()
        ppv_ordre = self.get_ppv_route()
        t1 = time.time()
        
        ppv_route = Route(self.graph, ppv_ordre)
        ppv_dist = self.graph.calcul_distance_route(ppv_route)
        
        print(f"⏱️  Temps PPV : {t1-t0:.4f} s")
        print(f"📏 Distance PPV : {ppv_dist:.2f}")
        self.app.log(f"Base PPV: {ppv_dist:.2f} ({t1-t0:.2f}s)")
        self.app.afficher_route(ppv_route, "red")
        
        # Injection Phéromones sur le PPV
        arr_t = np.array(ppv_ordre)
        voisins_idx, _ = self.graph.matrice_od
        for i in range(self.n):
            u, v = arr_t[i], arr_t[i+1]
            # On trouve l'index local de v chez u
            # np.where sur petit tableau (taille 40) est rapide
            idx = np.where(voisins_idx[u] == v)[0]
            if idx.size > 0: self.pheromones[u, idx[0]] += 5.0

        # 2. BOUCLE ACO
        best_dist = ppv_dist
        best_route = ppv_route
        start_t = time.time()
        
        for it in range(nb_iter):
            routes = []
            # Lancement fourmis
            for k in range(self.nb_fourmis):
                o = self.construire_solution()
                r = Route(self.graph, o)
                d = self.graph.calcul_distance_route(r)
                routes.append((o, d))
                
                if d < best_dist:
                    best_dist = d
                    best_route = r
                    self.app.log(f"RECORD: {d:.2f}")
                    self.app.afficher_route(best_route, "cyan")
            
            # Evaporation
            self.pheromones *= (1 - self.rho)
            
            # Renforcement
            for o, d in routes:
                delta = self.Q / d
                for i in range(self.n):
                    u, v = o[i], o[i+1]
                    idx = np.where(voisins_idx[u] == v)[0]
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
    
    # Peu de fourmis pour aller vite sur 200k
    solver = TSP_ACO(g, app, nb_fourmis=10)
    
    app.root.after(100, lambda: solver.resoudre(nb_iter=500))
    app.root.mainloop()