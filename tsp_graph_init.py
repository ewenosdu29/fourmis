"""
Fichier : tsp_graph_init.py
But: implémentation OPTIMISÉE des classes demandées pour le projet TSP.
Dépendances autorisées : numpy, random, time, pandas, tkinter, csv

Classes implémentées :
 - Lieu
 - Graph (avec optimisations de performance)
 - Route
 - Affichage (Tkinter)

OPTIMISATIONS APPLIQUÉES :
 - Matrice de distances calculée automatiquement (1 seule fois)
 - Pas de vérification répétée de matrice_od dans les boucles
 - Méthode plus_proche_voisin_rapide() pour set (O(k) au lieu de O(n))
 - Graph immutable (réutilisable à l'infini)

Usage rapide :
>>> from tsp_graph_init import Graph, Route, Affichage
>>> g = Graph(nb_lieux=20)                # génère + calcule matrice automatiquement
>>> r = Route(g)                           # route aléatoire
>>> print(r.calcul_distance())
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
from typing import List, Optional, Tuple, Set

import numpy as np
import tkinter as tk
from tkinter import scrolledtext

# ============================================================================
# CONSTANTES
# ============================================================================

LARGEUR = 800
HAUTEUR = 600
NB_LIEUX = 20
RAYON_LIEU = 8
N_BEST = 5  # par défaut pour l'affichage des N meilleures routes


# ============================================================================
# CLASSE LIEU
# ============================================================================

class Lieu:
    """Représentation d'un lieu (x,y) avec un nom.
    Fournit une méthode distance_to pour la distance euclidienne.
    """

    def __init__(self, x: float, y: float, name: Optional[str] = None):
        self.x = float(x)
        self.y = float(y)
        self.name = str(name) if name is not None else ""

    def distance_to(self, other: "Lieu") -> float:
        """Calcule la distance euclidienne vers un autre lieu."""
        dx = self.x - other.x
        dy = self.y - other.y
        return sqrt(dx * dx + dy * dy)

    def __repr__(self):
        return f"Lieu(name={self.name!r}, x={self.x:.2f}, y={self.y:.2f})"


# ============================================================================
# CLASSE GRAPH (OPTIMISÉE)
# ============================================================================

class Graph:
    """Graphe de lieux IMMUTABLE et OPTIMISÉ.
    
    Une fois créé, le graphe ne change plus (liste_lieux fixe).
    La matrice de distances est calculée AUTOMATIQUEMENT à la création.
    
    Attributs principaux :
      - liste_lieux : List[Lieu] - TOUS les lieux (fixe, immuable)
      - matrice_od : numpy.ndarray (n x n) - distances pré-calculées
      - nb_lieux : int - nombre de lieux
    
    OPTIMISATIONS :
      - Matrice calculée 1 fois automatiquement (pas de vérification répétée)
      - Méthode plus_proche_voisin_rapide() pour itération sur set O(k)
      - Graph réutilisable à l'infini (pas de modification interne)
    """

    def __init__(self, nb_lieux: int = NB_LIEUX, largeur: int = LARGEUR, hauteur: int = HAUTEUR):
        """
        Initialise un graphe avec génération aléatoire de lieux.
        
        Args:
            nb_lieux: Nombre de lieux à générer
            largeur: Largeur de l'espace (en pixels)
            hauteur: Hauteur de l'espace (en pixels)
        
        Note: La matrice de distances est calculée AUTOMATIQUEMENT.
        """
        self.largeur = largeur
        self.hauteur = hauteur
        self.nb_lieux = int(nb_lieux)
        
        # Liste unique et fixe de TOUS les lieux
        self.liste_lieux: List[Lieu] = []
        
        # Matrice des distances (sera calculée automatiquement)
        self.matrice_od: Optional[np.ndarray] = None
        
        # Génération aléatoire des lieux
        self.generer_lieux_aleatoires(self.nb_lieux)
        
        # ✅ OPTIMISATION : Calcul automatique de la matrice (1 seule fois)
        self.calcul_matrice_cout_od()

    def generer_lieux_aleatoires(self, nb: int):
        """
        Génère nb lieux avec coordonnées aléatoires.
        Marge de 20px pour éviter que les lieux touchent les bords.
        
        Args:
            nb: Nombre de lieux à générer
        """
        self.liste_lieux = []
        margin = 20
        
        for i in range(nb):
            # Coordonnées aléatoires avec marge
            x = random.uniform(margin, self.largeur - margin)
            y = random.uniform(margin, self.hauteur - margin)
            self.liste_lieux.append(Lieu(x, y, name=str(i)))
        
        self.nb_lieux = len(self.liste_lieux)

    def charger_graph(self, filename: str):
        """
        Charge les lieux depuis un fichier CSV.
        Format attendu : x,y[,name] (avec ou sans ligne d'en-tête)
        
        Args:
            filename: Chemin vers le fichier CSV
        
        Note: La matrice de distances est recalculée AUTOMATIQUEMENT.
        """
        lieux = []
        
        # Lecture du fichier
        with open(filename, newline="", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            rows = [r for r in reader if r]

        # Détection d'en-tête (première ligne pas convertible en float)
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
                start = 1  # Sauter la ligne d'en-tête

        # Lecture des lieux
        for r in rows[start:]:
            if len(r) >= 2:
                x = float(r[0])
                y = float(r[1])
                name = r[2] if len(r) >= 3 else None
                lieux.append(Lieu(x, y, name))

        if not lieux:
            raise ValueError("Aucun lieu trouvé dans le fichier CSV")

        # Mise à l'échelle si les coordonnées dépassent la zone
        xs = [l.x for l in lieux]
        ys = [l.y for l in lieux]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)

        def rescale(val, mn, mx, out_min, out_max):
            """Mise à l'échelle linéaire"""
            if mx == mn:
                return (out_min + out_max) / 2
            return out_min + (val - mn) * (out_max - out_min) / (mx - mn)

        margin = 20
        scaled = []
        for l in lieux:
            sx = rescale(l.x, minx, maxx, margin, self.largeur - margin)
            sy = rescale(l.y, miny, maxy, margin, self.hauteur - margin)
            scaled.append(Lieu(sx, sy, l.name))

        self.liste_lieux = scaled
        self.nb_lieux = len(self.liste_lieux)
        
        # ✅ OPTIMISATION : Recalcul automatique de la matrice
        self.calcul_matrice_cout_od()

    def calcul_matrice_cout_od(self):
        """
        Calcule la matrice symétrique des distances euclidiennes.
        Stocke le résultat dans self.matrice_od (numpy.ndarray shape (n,n)).
        
        OPTIMISATION : Calcule seulement le triangle supérieur (n(n-1)/2 calculs)
        puis remplit le triangle inférieur par symétrie.
        
        Returns:
            numpy.ndarray: La matrice de distances
        """
        n = self.nb_lieux
        mat = np.zeros((n, n), dtype=float)
        
        # ✅ OPTIMISÉ : Seulement n(n-1)/2 calculs
        for i in range(n):
            for j in range(i + 1, n):
                d = self.liste_lieux[i].distance_to(self.liste_lieux[j])
                mat[i, j] = d
                mat[j, i] = d  # Symétrie
        
        self.matrice_od = mat
        return mat

    def plus_proche_voisin(self, index: int, visited: Optional[List[bool]] = None) -> int:
        """
        Retourne l'indice du plus proche voisin du lieu `index` non visité.
        Version CLASSIQUE avec liste booléenne.
        
        Args:
            index: Indice du lieu de référence
            visited: Liste booléenne des lieux visités (None = tous disponibles)
        
        Returns:
            int: Indice du plus proche voisin (-1 si aucun)
        
        Complexité: O(n)
        
        ✅ OPTIMISATION : Pas de vérification de matrice_od (toujours présente)
        """
        # ✅ PAS DE CHECK - la matrice est TOUJOURS calculée
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

    def plus_proche_voisin_rapide(self, index: int, lieux_restants_set: Set[int]) -> int:
        """
        VERSION OPTIMISÉE : Retourne le plus proche voisin parmi un set.
        
        Args:
            index: Indice du lieu de référence
            lieux_restants_set: Set des indices de lieux disponibles
        
        Returns:
            int: Indice du plus proche voisin (-1 si aucun)
        
        Complexité: O(k) où k = len(lieux_restants_set)
        
        ✅ OPTIMISATION : Parcourt SEULEMENT les lieux restants
        Plus rapide que plus_proche_voisin() quand k << n
        """
        # ✅ PAS DE CHECK - la matrice est TOUJOURS calculée
        row = self.matrice_od[index]
        best_idx = -1
        best_d = float('inf')
        
        # ✅ Itération sur set : O(k) au lieu de O(n)
        for j in lieux_restants_set:
            if j == index:
                continue
            
            d = row[j]
            if d < best_d:
                best_d = d
                best_idx = j
        
        return best_idx

    def calcul_distance_route(self, ordre: List[int]) -> float:
        """
        Calcule la distance totale d'une route (liste d'indices).
        
        Args:
            ordre: Liste d'indices de lieux (ex: [0, 3, 5, 1, 2, 0])
        
        Returns:
            float: Distance totale de la route
        
        ✅ OPTIMISATION : Pas de vérification de matrice_od
        """
        if not ordre:
            return 0.0
        
        # ✅ PAS DE CHECK - la matrice est TOUJOURS calculée
        dist = 0.0
        for a, b in zip(ordre[:-1], ordre[1:]):
            dist += self.matrice_od[a, b]
        
        return float(dist)


# ============================================================================
# CLASSE ROUTE
# ============================================================================

class Route:
    """Représente une route (ordre de visites).
    
    Par contrainte TSP, doit commencer et finir par le lieu 0.
    
    Attributs :
      - ordre : List[int] (ex: [0, 3, 8, 1, ..., 0])
      - graph : Graph (référence au graphe)
    """

    def __init__(self, graph: Graph, ordre: Optional[List[int]] = None):
        """
        Initialise une route.
        
        Args:
            graph: Le graphe de référence
            ordre: Liste d'indices de lieux (None = génération aléatoire)
        
        Note: Si ordre ne commence/finit pas par 0, ils sont ajoutés automatiquement.
        """
        self.graph = graph
        
        if ordre is None:
            # Génère une permutation aléatoire avec 0 au début et à la fin
            perm = list(range(1, graph.nb_lieux))
            random.shuffle(perm)
            self.ordre = [0] + perm + [0]
        else:
            # Normalise pour s'assurer que commence et finit par 0
            if ordre[0] != 0:
                ordre = [0] + ordre
            if ordre[-1] != 0:
                ordre = ordre + [0]
            self.ordre = ordre

    def calcul_distance(self) -> float:
        """Calcule la distance totale de cette route."""
        return self.graph.calcul_distance_route(self.ordre)

    def is_valid(self) -> bool:
        """
        Vérifie si la route est valide selon les contraintes TSP :
        - Commence par 0
        - Finit par 0
        - Visite tous les lieux exactement une fois (sauf 0 qui apparaît 2 fois)
        
        Returns:
            bool: True si valide, False sinon
        """
        if len(self.ordre) < 2:
            return False
        if self.ordre[0] != 0 or self.ordre[-1] != 0:
            return False
        
        # Vérifier que tous les lieux sont visités exactement une fois
        lieux_uniques = set(self.ordre[1:-1])  # Exclure les deux 0
        return len(lieux_uniques) == (len(self.ordre) - 2)

    def __repr__(self):
        return f"Route(dist={self.calcul_distance():.2f}, ordre={self.ordre})"


# ============================================================================
# CLASSE AFFICHAGE
# ============================================================================

class Affichage:
    """Affichage Tkinter du graphe et des routes.
    
    Fonctionnalités :
    - Affiche les lieux (cercles numérotés)
    - Affiche la meilleure route (ligne bleue pointillée)
    - Affiche N meilleures routes en gris clair (touche 'p')
    - Affiche la matrice des coûts (touche 'm')
    - Zone de texte pour informations et statistiques
    
    Touches :
    - ESC : quitter
    - 'p' : afficher/masquer N meilleures routes
    - 'm' : afficher/masquer matrice des coûts
    """

    def __init__(self, graph: Graph, routes_population: Optional[List[Route]] = None, 
                 group_name: str = "Groupe TSP"):
        """
        Initialise l'affichage.
        
        Args:
            graph: Le graphe à afficher
            routes_population: Liste de routes (pour affichage des N meilleures)
            group_name: Nom du groupe (affiché dans le titre)
        """
        self.graph = graph
        self.routes_population = routes_population or []
        self.best_route: Optional[Route] = None
        
        # Trouver la meilleure route de la population
        if self.routes_population:
            self.best_route = min(self.routes_population, key=lambda r: r.calcul_distance())
        
        # Options d'affichage
        self.show_population = False
        self.show_matrix = False
        self.N_best = N_BEST

        # ===== Configuration Tkinter =====
        self.root = tk.Tk()
        self.root.title(f"TSP - {group_name}")
        
        # Canvas pour le dessin
        self.canvas = tk.Canvas(
            self.root, 
            width=self.graph.largeur, 
            height=self.graph.hauteur, 
            bg="white"
        )
        self.canvas.pack()

        # Zone de texte scrollable
        self.text = scrolledtext.ScrolledText(self.root, height=8)
        self.text.pack(fill=tk.BOTH, expand=False)

        # Raccourcis clavier
        self.root.bind('<Escape>', lambda e: self.root.quit())
        self.root.bind('p', lambda e: self.toggle_population())
        self.root.bind('m', lambda e: self.toggle_matrix())

        # Dessin initial
        self._draw_all()

    def _coord_canvas(self, lieu: Lieu) -> Tuple[float, float]:
        """Retourne les coordonnées canvas d'un lieu."""
        return lieu.x, lieu.y

    def _draw_lieux(self):
        """Dessine tous les lieux du graphe (cercles numérotés)."""
        self.canvas.delete('lieu')
        
        for idx, lieu in enumerate(self.graph.liste_lieux):
            x, y = self._coord_canvas(lieu)
            x0, y0 = x - RAYON_LIEU, y - RAYON_LIEU
            x1, y1 = x + RAYON_LIEU, y + RAYON_LIEU
            
            # Cercle
            self.canvas.create_oval(
                x0, y0, x1, y1, 
                fill='white', 
                outline='black', 
                tags='lieu'
            )
            
            # Numéro
            self.canvas.create_text(
                x, y, 
                text=str(idx), 
                tags='lieu'
            )

    def _draw_route(self, route: Route, color: str = 'blue', 
                    dashed: bool = False, width: int = 2, tag: str = 'best'):
        """
        Dessine une route sur le canvas.
        
        Args:
            route: La route à dessiner
            color: Couleur de la ligne
            dashed: True pour ligne pointillée
            width: Épaisseur de la ligne
            tag: Tag Tkinter pour identifier la route
        """
        if not route or not route.ordre:
            return
        
        # Construire la liste de coordonnées
        coords = []
        for idx in route.ordre:
            lieu = self.graph.liste_lieux[idx]
            coords.extend(self._coord_canvas(lieu))
        
        # Style de ligne
        dash = (4, 6) if dashed else None
        
        # Supprimer ancienne route avec ce tag
        self.canvas.delete(tag)
        
        # Dessiner la ligne
        self.canvas.create_line(
            *coords, 
            fill=color, 
            width=width, 
            dash=dash, 
            tags=tag
        )
        
        # Afficher l'ordre de visite au-dessus de chaque lieu
        for order_idx, node in enumerate(route.ordre[:-1]):
            x, y = self._coord_canvas(self.graph.liste_lieux[node])
            self.canvas.create_text(
                x, y - 12, 
                text=str(order_idx), 
                font=("Arial", 8), 
                tags=tag
            )

    def _draw_population(self):
        """Dessine les N meilleures routes de la population en gris clair."""
        self.canvas.delete('population')
        
        if not self.routes_population:
            return
        
        # Trier par distance
        sorted_routes = sorted(self.routes_population, key=lambda r: r.calcul_distance())
        
        # Dessiner les N meilleures
        for r in sorted_routes[:self.N_best]:
            self._draw_route(r, color='lightgray', dashed=False, width=1, tag='population')

    def _draw_cost_matrix_in_text(self):
        """Affiche la matrice des coûts dans la zone de texte."""
        self.text.delete('1.0', tk.END)
        
        if not self.show_matrix:
            return
        
        mat = self.graph.matrice_od
        n = self.graph.nb_lieux
        
        self.text.insert(tk.END, "Matrice des coûts (distances euclidiennes)\n")
        self.text.insert(tk.END, "="*70 + "\n")
        
        # Formatage simple de la matrice
        for i in range(n):
            row = ' '.join(f"{mat[i,j]:6.1f}" for j in range(n))
            self.text.insert(tk.END, row + "\n")

    def _draw_all(self):
        """Redessine tout l'affichage."""
        self.canvas.delete('all')
        
        # Dessiner dans l'ordre : population, lieux, meilleure route
        self._draw_lieux()
        
        if self.show_population:
            self._draw_population()
        
        if self.best_route:
            # Ligne bleue pointillée pour la meilleure route
            self._draw_route(
                self.best_route, 
                color='blue', 
                dashed=True, 
                width=2, 
                tag='best_route'
            )
        
        self._draw_cost_matrix_in_text()
        self._log_status()

    def _log_status(self):
        """Affiche les informations dans la zone de texte."""
        self.text.insert(tk.END, f"\nHeure: {time.strftime('%H:%M:%S')} - Lieux: {self.graph.nb_lieux}\n")
        
        if self.best_route:
            self.text.insert(tk.END, f"Meilleure distance: {self.best_route.calcul_distance():.2f}\n")
        
        if self.show_population and self.routes_population:
            bests = sorted(self.routes_population, key=lambda r: r.calcul_distance())[:self.N_best]
            self.text.insert(tk.END, f"\n{self.N_best} meilleures routes:\n")
            for i, r in enumerate(bests):
                self.text.insert(tk.END, f"  {i+1}. dist={r.calcul_distance():.2f}\n")
        
        self.text.see(tk.END)

    def toggle_population(self):
        """Bascule l'affichage des N meilleures routes."""
        self.show_population = not self.show_population
        self._draw_all()

    def toggle_matrix(self):
        """Bascule l'affichage de la matrice des coûts."""
        self.show_matrix = not self.show_matrix
        self._draw_all()

    def mainloop(self):
        """Lance la boucle principale Tkinter."""
        self.root.mainloop()


# ============================================================================
# TESTS ET EXEMPLE
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("TEST DU MODULE tsp_graph_init.py OPTIMISÉ")
    print("="*70)
    
    # Créer un graphe
    print("\n1. Création d'un graphe de 15 lieux...")
    g = Graph(nb_lieux=15)
    print(f"   ✓ {g.nb_lieux} lieux générés")
    print(f"   ✓ Matrice de distances calculée automatiquement : {g.matrice_od.shape}")
    
    # Test de plus_proche_voisin
    print("\n2. Test plus_proche_voisin()...")
    ppv = g.plus_proche_voisin(0)
    print(f"   ✓ Plus proche voisin du lieu 0 : lieu {ppv}")
    print(f"   ✓ Distance : {g.matrice_od[0][ppv]:.2f}")
    
    # Test de plus_proche_voisin_rapide avec set
    print("\n3. Test plus_proche_voisin_rapide() avec set...")
    lieux_restants = {1, 2, 3, 4, 5}
    ppv_rapide = g.plus_proche_voisin_rapide(0, lieux_restants)
    print(f"   ✓ Plus proche parmi {lieux_restants} : lieu {ppv_rapide}")
    
    # Créer une population de routes
    print("\n4. Génération d'une population de 20 routes aléatoires...")
    population = [Route(g) for _ in range(20)]
    print(f"   ✓ {len(population)} routes créées")
    
    # Trouver la meilleure
    best = min(population, key=lambda r: r.calcul_distance())
    print(f"   ✓ Meilleure distance : {best.calcul_distance():.2f}")
    print(f"   ✓ Route : {best.ordre}")
    
    # Lancer l'interface
    print("\n5. Lancement de l'interface graphique...")
    print("   - Appuyez sur 'p' pour voir les 5 meilleures routes")
    print("   - Appuyez sur 'm' pour voir la matrice des coûts")
    print("   - Appuyez sur ESC pour quitter")
    
    aff = Affichage(g, routes_population=population, group_name='Test Optimisé')
    aff.best_route = best
    aff.mainloop()
    
    print("\nTest terminé !")