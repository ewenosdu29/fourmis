"""
Fichier : tsp_graph_init.py
But: implémentation des classes demandées pour le projet TSP.
Dépendances autorisées : numpy, random, time, pandas, tkinter, csv

Classes implémentées :
 - Lieu
 - Graph
 - Route
 - Affichage (Tkinter)

Usage rapide :
>>> from tsp_graph_init import Graph, Route, Affichage
>>> g = Graph(nb_lieux=20)                # génère aléatoirement 20 lieux
>>> g.calcul_matrice_cout_od()
>>> r = Route(g)                           # route aléatoire
>>> print(g.calcul_distance_route(r.ordre))
>>> Affichage(g, routes_population=[r]).mainloop()

Remarques :
 - Le format CSV accepté par charger_graph : colonnes x,y,[name] (avec ou sans en-tête).
 - TOUCHES :
     - <Escape> : quitte l'application
     - 'p' : afficher/masquer les N meilleures routes (paramétrable via N_BEST)
     - 'm' : afficher/masquer la matrice de coûts dans la zone de texte

"""

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
NB_LIEUX = 20
RAYON_LIEU = 8
N_BEST = 5  # par defaut pour l'affichage des N meilleures routes


class Lieu:
    """Représentation d'un lieu (x,y) avec un nom.
    Fournit une méthode distance_to pour la distance euclidienne.
    """

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
    """Graphe de lieux.

    Attributs principaux :
      - liste_lieux : List[Lieu]
      - matrice_od : numpy.ndarray (NB x NB) des distances
    """

    def __init__(self, nb_lieux: int = NB_LIEUX, largeur: int = LARGEUR, hauteur: int = HAUTEUR):
        self.largeur = largeur
        self.hauteur = hauteur
        self.nb_lieux = int(nb_lieux)
        self.liste_lieux: List[Lieu] = []
        self.matrice_od: Optional[np.ndarray] = None
        # si pas de chargement CSV, on génère aléatoirement
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

    def charger_graph(self, filename: str):
        """Charger les lieux depuis un CSV. Format attendu : x,y[,name]
        Accepte ou non une ligne d'en-tête.
        """
        lieux = []
        with open(filename, newline="", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            rows = [r for r in reader if r]

        # détecter en-tête si le premier élément n'est pas convertible en float
        def is_float(s):
            try:
                float(s)
                return True
            except Exception:
                return False

        start = 0
        if rows:
            first = rows[0]
            if not rows[0] or not is_float(first[0]):
                start = 1

        for r in rows[start:]:
            if len(r) >= 2:
                x = float(r[0])
                y = float(r[1])
                name = r[2] if len(r) >= 3 else None
                lieux.append(Lieu(x, y, name))

        if not lieux:
            raise ValueError("Aucun lieu trouv\u00e9 dans le fichier CSV")

        # si les coordonnées dépassent la zone, on les met à l'échelle
        xs = [l.x for l in lieux]
        ys = [l.y for l in lieux]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)

        # échelle linéaire si besoin
        def rescale(val, mn, mx, out_min, out_max):
            if mx == mn:
                return (out_min + out_max) / 2
            return out_min + (val - mn) * (out_max - out_min) / (mx - mn)

        scaled = []
        margin = 20
        for l in lieux:
            sx = rescale(l.x, minx, maxx, margin, self.largeur - margin)
            sy = rescale(l.y, miny, maxy, margin, self.hauteur - margin)
            scaled.append(Lieu(sx, sy, l.name))

        self.liste_lieux = scaled
        self.nb_lieux = len(self.liste_lieux)
        self.matrice_od = None

    def calcul_matrice_cout_od(self):
        """Calcule la matrice symétrique des distances euclidiennes entre tous les lieux.
        Stocke le résultat dans self.matrice_od (numpy.ndarray shape (n,n)).
        Affiche également la matrice dans la console.
        """
        n = self.nb_lieux
        mat = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                d = self.liste_lieux[i].distance(self.liste_lieux[j])
                mat[i, j] = d
                mat[j, i] = d
        self.matrice_od = mat

        # affichage formaté
        #print("Matrice des distances euclidiennes :")
        #for row in mat:
        #    print(" ".join(f"{val:6.2f}" for val in row))

        return mat


    def plus_proche_voisin(self, index: int, visited: Optional[List[bool]] = None) -> int:
        """Retourne l'indice du plus proche voisin du lieu `index` non visité.
        Si visited est None, on renvoie le plus proche parmi tous sauf index.
        """
        if self.matrice_od is None:
            self.calcul_matrice_cout_od()
        row = self.matrice_od[index]
        n = self.nb_lieux
        best_idx = -1
        best_d = float('inf')
        for j in range(n):
            if j == index:
                continue
            if visited is not None and visited[j]:
                continue
            d = row[j]
            if d < best_d:
                best_d = d
                best_idx = j
        return best_idx

    def nearest_insertion_route(self) -> "Route":
        if self.matrice_od is None:
            self.calcul_matrice_cout_od()

        n = self.nb_lieux
        if n < 2:
            return Route(self, list(range(n)))

        # Commencer par 0
        start = 0

        # trouver la ville la plus proche de 0
        nearest = min(((i, self.matrice_od[start, i]) for i in range(1, n)),key=lambda x: x[1])[0]

        tour = [start, nearest]
        remaining = set(range(n)) - set(tour)

        while remaining:
            # trouver la ville la plus proche du tour
            nearest_city, best_dist = None, float("inf")
            for city in remaining:
                d = min(self.matrice_od[city, t] for t in tour)
                if d < best_dist:
                    best_dist = d
                    nearest_city = city

            # meilleure position d'insertion
            best_pos, best_increase = None, float("inf")
            for i in range(len(tour)):
                j = (i + 1) % len(tour)
                increase = (self.matrice_od[tour[i], nearest_city] + self.matrice_od[nearest_city, tour[j]] -self.matrice_od[tour[i], tour[j]])
               
                if increase < best_increase:
                    best_increase = increase
                    best_pos = j

            tour.insert(best_pos, nearest_city)
            remaining.remove(nearest_city)

        # fermer le tour
        if tour[-1] != tour[0]:
            tour.append(tour[0])

        return Route(self, tour)
    

    def calcul_distance_route(self, ordre: List[int]) -> float:
        """Calcule la distance totale d'une route (liste d'indices)."""
        if not ordre:
            return 0.0
        if self.matrice_od is None:
            self.calcul_matrice_cout_od()
        dist = 0.0
        for a, b in zip(ordre[:-1], ordre[1:]):
            dist += self.matrice_od[a, b]
        return float(dist)
    
    def route_heuristique(self, methode: str = "ppv") -> "Route":
        if self.matrice_od is None:
            self.calcul_matrice_cout_od()

        if methode == "ppv":
            n = self.nb_lieux
            visited = [False] * n
            ordre = [0]
            visited[0] = True
            current = 0
            for _ in range(n - 1):
                nxt = self.plus_proche_voisin(current, visited)
                ordre.append(nxt)
                visited[nxt] = True
                current = nxt
            ordre.append(0)
            return Route(self, ordre)

        elif methode == "2opt":
            route_init = Route(self)
            route_init.ameliorer_2opt()
            return route_init

        elif methode == "lk":
            route_init = Route(self)
            route_init.ameliorer_lin_kernighan()  # <- plus besoin de passer route_init
            return route_init
        
        elif methode == "ni":
            return self.nearest_insertion_route()

        else:
            raise ValueError("Méthode inconnue. Utilisez 'ppv', '2opt' ou 'lk'.")





class Route:
    """Représente une route (ordre de visites). Par contrainte, doit commencer et finir par 0.

    Attributs :
      - ordre : List[int] (ex: [0,3,8,1,...,0])
    """

    def __init__(self, graph: Graph, ordre: Optional[List[int]] = None):
        self.graph = graph
        if ordre is None:
            # génère une permutation aléatoire avec 0 en premier et dernier
            perm = list(range(1, graph.nb_lieux))
            random.shuffle(perm)
            ordre_gen = [0] + perm + [0]
            self.ordre = ordre_gen
        else:
            # on normalise pour s'assurer que commence et finit par 0
            if ordre[0] != 0:
                ordre = [0] + ordre
            if ordre[-1] != 0:
                ordre = ordre + [0]
            self.ordre = ordre

    def calcul_distance(self) -> float:
        return self.graph.calcul_distance_route(self.ordre)
    
    def ameliorer_2opt(self):
        """Améliore la route actuelle par l’algorithme 2-opt"""
        improved = True
        best_distance = self.calcul_distance()
        best_ordre = self.ordre.copy()

        while improved:
            improved = False
            for i in range(1, len(best_ordre) - 2):
                for j in range(i + 1, len(best_ordre) - 1):
                    if j - i == 1:
                        continue  # évite les inversions inutiles
                    new_ordre = best_ordre[:i] + best_ordre[i:j][::-1] + best_ordre[j:]
                    new_route = Route(self.graph, new_ordre)
                    new_distance = new_route.calcul_distance()

                    if new_distance < best_distance:
                        best_ordre = new_ordre
                        best_distance = new_distance
                        improved = True
            self.ordre = best_ordre.copy()
        return best_ordre, best_distance
    
    def ameliorer_lin_kernighan(self, max_iter: int = 100):
        """
        Améliore la route actuelle avec une heuristique simplifiée Lin-Kernighan.
        """
        best_distance = self.calcul_distance()
        best_order = self.ordre.copy()
        n = len(best_order)

        for iteration in range(max_iter):
            improved = False
            for i in range(1, n - 2):
                for j in range(i + 1, n - 1):
                    # inverser le segment i:j pour créer une nouvelle route candidate
                    new_order = best_order[:i] + best_order[i:j][::-1] + best_order[j:]
                    new_route = Route(self.graph, new_order)
                    new_distance = new_route.calcul_distance()

                    if new_distance < best_distance:
                        best_order = new_order
                        best_distance = new_distance
                        improved = True
                        break  # sortir de la boucle j
                if improved:
                    break  # sortir de la boucle i
            if not improved:
                break  # plus d'amélioration possible

        self.ordre = best_order
        return self.ordre, best_distance

    

    def is_valid(self) -> bool:
        return len(self.ordre) >= 2 and self.ordre[0] == 0 and self.ordre[-1] == 0 and len(set(self.ordre[1:-1])) == (len(self.ordre) - 2)

    def __repr__(self):
        return f"Route(dist={self.calcul_distance():.2f}, ordre={self.ordre})"

    ### potentiellement rajout des opérateurs de comparaison


class Affichage:
    """Affichage Tkinter du graphe.

    - Affiche les lieux (cercles numérotés)
    - Affiche la meilleure route (ligne bleue pointillée)
    - Zone de texte en dessous pour messages et affichage de matrice
    - Touches : ESC pour quitter, 'p' pour afficher/masquer N meilleures routes (gris clair), 'm' pour afficher/masquer matrice des coûts
    """

    def __init__(
        self,
        graph: Graph,
        routes_population: Optional[List[Route]] = None,
        group_name: str = "Groupe TSP",
        methode: str = "inconnue"
    ):
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
        self.methode = methode  # <-- ajout du paramètre méthode

        # Tkinter setup
        self.root = tk.Tk()
        self.root.title(f"TSP - {group_name}")
        self.canvas = tk.Canvas(self.root, width=self.graph.largeur, height=self.graph.hauteur, bg="white")
        self.canvas.pack()

        # zone de texte scrollable
        self.text = scrolledtext.ScrolledText(self.root, height=8)
        self.text.pack(fill=tk.BOTH, expand=False)

        # bind keys
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
        for order_idx, node in enumerate(route.ordre[:-1]):
            x, y = self._coord_canvas(self.graph.liste_lieux[node])
            self.canvas.create_text(x, y - 12, text=str(order_idx), font=("Arial", 8), tags=tag)

    def _draw_population(self):
        self.canvas.delete('population')
        if not self.routes_population:
            return
        sorted_routes = sorted(self.routes_population, key=lambda r: r.calcul_distance())
        for r in sorted_routes[: self.N_best]:
            self._draw_route(r, color='lightgray', dashed=False, width=1, tag='population')

    def _draw_cost_matrix_in_text(self):
        self.text.delete('1.0', tk.END)
        if not self.show_matrix:
            return
        mat = self.graph.matrice_od
        n = self.graph.nb_lieux
        self.text.insert(tk.END, "Matrice des co\u00fbts (distances euclidiennes)\n")
        for i in range(n):
            row = ' '.join(f"{mat[i,j]:6.1f}" for j in range(n))
            self.text.insert(tk.END, row + "\n")

    def _draw_all(self):
        self.canvas.delete('all')
        self._draw_lieux()
        if self.show_population:
            self._draw_population()
        if self.best_route:
            self._draw_route(self.best_route, color='blue', dashed=True, width=2, tag='best_route')
        self._draw_cost_matrix_in_text()
        self._log_status()

    def _log_status(self):
        self.text.insert(tk.END, f"Heure: {time.strftime('%H:%M:%S')} - Lieux: {self.graph.nb_lieux}\n")
        self.text.insert(tk.END, f"Méthode utilisée: {self.methode}\n")  # <-- affichage dans les logs
        if self.best_route:
            self.text.insert(tk.END, f"Meilleure distance: {self.best_route.calcul_distance():.2f}\n")
        if self.show_population and self.routes_population:
            bests = sorted(self.routes_population, key=lambda r: r.calcul_distance())[: self.N_best]
            self.text.insert(tk.END, "N meilleures routes:\n")
            for i, r in enumerate(bests):
                self.text.insert(tk.END, f" {i+1}. dist={r.calcul_distance():.2f} ordre={r.ordre}\n")
        self.text.see(tk.END)

    def toggle_population(self):
        self.show_population = not self.show_population
        self._draw_all()

    def toggle_matrix(self):
        self.show_matrix = not self.show_matrix
        self._draw_all()

    def mainloop(self):
        self.root.mainloop()




if __name__ == '__main__':
    g = Graph(nb_lieux=20)
    g.calcul_matrice_cout_od()

    # Toutes les heuristiques disponibles
    methodes = ["ppv", "2opt", "lk", "ni"]
    routes = []

    # Dictionnaire pour associer chaque route à sa méthode
    method_to_route = {}

    for methode in methodes:
        route = g.route_heuristique(methode)
        print(f"Route trouvée avec {methode}: {route.ordre}")
        print(f"Distance totale: {route.calcul_distance():.2f}")
        routes.append(route)
        method_to_route[methode] = route

    # Trouver la meilleure méthode (celle avec la plus petite distance)
    best_methode, best_route = min(method_to_route.items(), key=lambda item: item[1].calcul_distance())

    print(f"\n👉 Meilleure méthode : {best_methode} ({best_route.calcul_distance():.2f})")

    # Affichage avec toutes les routes et indication de la meilleure méthode
    aff = Affichage(
        g,
        routes_population=routes,
        group_name="Toutes les heuristiques",
        methode=best_methode  # ✅ ajouté ici
    )
    aff.mainloop()


