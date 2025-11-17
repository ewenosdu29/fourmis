import random
import numpy as np
import random
from tkinter import *
from tkinter import ttk
import csv
import time


LARGEUR=800 
HAUTEUR=600
NB_LIEUX = 20

# Classe Lieu : Mémorise les coordonnées x et y du lieu à visiter et son nom
class Lieu:

    """
    Mémorise les coordonnées x et y du lieu à visiter et son nom
    """
    def __init__(self, x, y, nom):
        self.x = x
        self.y = y
        self.nom = nom

    def distance(self, autre_lieu):
        """Calcule la distance euclidienne entre deux lieux."""
        return np.sqrt((self.x - autre_lieu.x) ** 2 + (self.y - autre_lieu.y) ** 2)

    def __repr__(self):
        return f"Lieu({self.x}, {self.y}, {self.nom})"


class Graph:
    """
    Mémorise une liste de lieux (variable liste_lieux).
    """
    def __init__(self, name_file):
        self.liste_lieux = self.charger_graph(name_file)
        self.matrice_od = [] # Matrice de distances entre les lieux
        self.calcul_matrice_cout_od()
        self.nb_lieux = NB_LIEUX

    def calcul_matrice_cout_od (self, nb_lieux = NB_LIEUX) :
        """
        calculer ou importer une matrice de distances entre chaque lieu du graphe et 
        stocker ce résultat dans une variable de classe matrice_od.

        des coordonnées qui devront être adaptées pour tenir dans un espace défini grâce à deux constantes LARGEUR=800 et HAUTEUR=600. 

        """

        # Definition de la matrice
        self.matrice_od = [[0] * nb_lieux for _ in range(nb_lieux)]

        #Remplissage avec les distances entre les lieux

        for i in range(nb_lieux):
            lieu1 = self.liste_lieux[i]
            for j in range(nb_lieux):
                lieu2 = self.liste_lieux[j]
                if i != j:
                    distance_pt = lieu1.distance(lieu2)
                    self.matrice_od[i][j] = distance_pt

        return self.matrice_od

    def plus_proche_voisin(self, index_lieu):
        """"
        Renvoie le plus proche voisin d'un lieu en utilisant la matrice de distances
        """       
        # Récupère la liste et filtre les 0
        distances = self.matrice_od[index_lieu]
        distances_sans_zero = [(i, dist) for i, dist in enumerate(distances) if dist != 0]

        # Trouver l'indice de la distance minimale
        min_index, min_dist = min(distances_sans_zero, key=lambda x: x[1])

        #retourner le voisin correspondant
        voisin = min_index

        return voisin
    
    def construire_route_heuristique(self, depart=0):
        """
        Construit une route complète en utilisant l'heuristique du plus proche voisin
        """
        non_visites = set(range(self.nb_lieux))
        route = [depart]
        non_visites.remove(depart)
        
        current = depart
        while non_visites:
            # Trouver le plus proche voisin parmi les non visités
            distances = [(i, self.matrice_od[current][i]) for i in non_visites]
            plus_proche = min(distances, key=lambda x: x[1])[0]
            route.append(plus_proche)
            non_visites.remove(plus_proche)
            current = plus_proche
        
        route.append(depart)  # Retour au point de départ
        return route



    def charger_graph(self, name_file):
        """
        Lecture dans un fichier CSV de la liste des coordonnées des lieux

        """
        liste_lieux = [] 

        x_max = float('-inf')
        y_max = float('-inf')

        with open(name_file, newline='', encoding='utf-8') as csvfile:
            file = csv.reader(csvfile)
           
            next(file)  # Ignorer l'en-tête "x,y"

            for line in file:
                    x, y = map(float, line)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)

        line_index=0
        with open(name_file, newline='', encoding='utf-8') as csvfile:
            file = csv.reader(csvfile)
            next(file)  # Ignorer l'en-tête "x,y"

            for line in file:
                x, y = map(float, line)

                # Normalisation avec les maxima trouvés
                x_scaled = x / (x_max+20) * LARGEUR
                y_scaled = y / (y_max+20) * HAUTEUR

                liste_lieux.append(Lieu(x_scaled, y_scaled, line_index))

                line_index +=1

        return liste_lieux

    def calcul_distance_route(self, route):
        """
        Calcule la distance totale d'une route
        """
        distance = 0

        for i in range(len(route.ordre) - 1):
            # Convertit les indices de la route en indices de la matrice
            index1 = int(route.ordre[i])
            index2 = int(route.ordre[i + 1])
            distance += self.matrice_od[index1][index2]
        route.distance = distance




class Route:
    def __init__(self, ordre_init = None):
        """
        ordre_init : Liste des indices des lieux in clus dans la route.
        """
        if not ordre_init or len(ordre_init) <= 2:
            raise ValueError("Une route doit contenir au moins trois lieux.")
        if ordre_init[0] != ordre_init[-1]:
            raise ValueError("Le premier et le dernier élément doivent être identiques.")
        self.ordre = ordre_init  # Liste des indices des lieux
        self.distance = None


    def __repr__(self):
        """Affichage de la route sous forme des indices des lieux."""
        return f"Route({self.ordre} - {self.distance})"


class Display:
    def __init__(self,height,width):
        self.dp = Tk()
        self.dp.title("Groupe F")
        self.display_best_route = 1
        frm = ttk.Frame(self.dp)
        frm.grid()
        self.display_best_route = 0

        # affichage graphe
        self.canvas = Canvas(self.dp, width=width, height=height)
        self.canvas.grid(column=0, row=0)

        # footer
        self.footer = Label(self.dp, text="text", font=("Arial", 10), bg="gray", fg="white")
        self.footer.grid(column=0, row=2, sticky="ew")
        self.update_footer(0, "Inf", 0, 0)

        self.dp.bind("<KeyPress>",self.key_event)

    def show(self):
        #affichage
        self.dp.mainloop()
        
    def update_footer(self, iteration, best_distance, elapsed_time, initial_distance):
        # Gérer les valeurs initiales et les types
        if initial_distance == 0 or best_distance == "Inf" or best_distance == float('inf'):
            self.footer.config(text=f"Initialisation en cours...")
        else:
            improvement = ((initial_distance - best_distance) / initial_distance) * 100
            self.footer.config(text=f"Itération {iteration + 1}: Distance initiale: {initial_distance:.2f} | Meilleure distance ACO: {best_distance:.2f} | Amélioration: {improvement:.1f}% | Temps: {elapsed_time:.2f}s")

    def key_event(self, event):
        """Gestion des touches clavier"""
        if event.keysym == "Escape":
            self.dp.destroy()  # Ferme proprement l'interface Tkinter
        elif event.keysym == "m":
            self.display_best_route = (self.display_best_route + 1) % 2
    
    def draw_nodes(self,nodes):
        
        for node in nodes:
            self.canvas.create_oval(node.x-5,node.y+5,node.x+5,node.y-5,fill="pink")
            self.canvas.create_text(node.x,node.y-10,text=node.nom)

    def draw_route(self,nodes,route,pheromones,iteration,best_distance,elapsed_time,initial_distance):
        self.canvas.delete('routes')
        self.canvas.delete('best_route')
        if self.display_best_route == 1:
            color="blue"
            dash=(4,1)
            for i in range(len(route.ordre) - 1):
                point_a = route.ordre[i]
                point_b = route.ordre[i + 1]  
                self.canvas.create_line(
                    nodes[point_a].x, nodes[point_a].y,
                    nodes[point_b].x, nodes[point_b].y,
                    fill=color,dash=dash,tags='best_route'
                )
        else:
            color="black"
            dash=None
            for i in range(len(pheromones)):
                for j in range(len(pheromones)):
                    if pheromones[i][j] >= 0.8:
                        self.canvas.create_line(
                            nodes[i].x, nodes[i].y,
                            nodes[j].x, nodes[j].y,
                        width = pheromones[i][j],
                        fill=color,dash=dash,tags='routes'
                        )
        
        self.canvas.update_idletasks()
        self.update_footer(iteration, best_distance, elapsed_time, initial_distance)


class TSP_ACO:
    def __init__(self, graph, num_ants=8, num_iterations=500, alpha=1.0, beta=2.0, evaporation_rate=0.5, pheromone_intensity=1000.0,iteration_max=500):
        self.graph = graph
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha  # Influence des phéromones
        self.beta = beta  # Influence de l'heuristique (distance inverse)
        self.evaporation_rate = evaporation_rate
        self.pheromone_intensity = pheromone_intensity
        self.iteration_max=iteration_max
        # Initialisation des phéromones sur les arêtes
        self.pheromones = np.ones((graph.nb_lieux, graph.nb_lieux))
        self.best_route = None
        self.best_distance = float('inf')
        self.start_time = None  # Temps de départ
        
        # Calcul de la solution initiale avec l'heuristique du plus proche voisin
        route_initiale = graph.construire_route_heuristique()
        route_obj = Route(route_initiale)
        graph.calcul_distance_route(route_obj)
        self.initial_distance = route_obj.distance
        print(f"Distance initiale (heuristique plus proche voisin): {self.initial_distance:.2f}")
    
    def run(self,iteration):
        if self.start_time is None:
            self.start_time = time.time()
        
        if iteration !=self.iteration_max:
            routes = self.construct_solutions()
            self.update_pheromones(routes)
            self.display_progress(iteration)
            display.dp.after(1, lambda:self.run(iteration+1))

    
    def construct_solutions(self):
        routes = []
        for _ in range(self.num_ants):
            route = self.construct_route()
            self.graph.calcul_distance_route(route)
            if route.distance < self.best_distance:
                self.best_distance = route.distance
                self.best_route = route
            routes.append((route, route.distance))
        return routes
    
    def construct_route(self):
        unvisited = list(range(self.graph.nb_lieux))
        start = random.choice(unvisited)
        route = [start]
        unvisited.remove(start)
        
        while unvisited:
            current = route[-1]
            probabilities = self.compute_probabilities(current, unvisited)
            next_city = random.choices(unvisited, probabilities)[0]
            route.append(next_city)
            unvisited.remove(next_city)
        
        route.append(start)  # Retour au point de départ
        return Route(route)
    
    def compute_probabilities(self, current, unvisited):
        pheromones = np.array([self.pheromones[current][j] for j in unvisited])
        heuristic = np.array([1 / self.graph.matrice_od[current][j] if self.graph.matrice_od[current][j] > 0 else 0 for j in unvisited])
        scores = (pheromones ** self.alpha) * (heuristic ** self.beta)
        return scores / scores.sum()
    
    def update_pheromones(self, routes):
        self.pheromones *= (1 - self.evaporation_rate)
        for route, distance in routes:
            contribution = self.pheromone_intensity / distance
            for i in range(len(route.ordre) - 1):
                a, b = route.ordre[i], route.ordre[i + 1]
                self.pheromones[a][b] += contribution
                self.pheromones[b][a] += contribution
    
    def display_progress(self, iteration):
        elapsed_time = time.time() - self.start_time

        if iteration ==self.iteration_max-1:
            display.display_best_route=1
            # Afficher le résumé final dans la console
            improvement = ((self.initial_distance - self.best_distance) / self.initial_distance) * 100
            print(f"\n=== RÉSULTATS FINAUX ===")
            print(f"Distance initiale (heuristique): {self.initial_distance:.2f}")
            print(f"Meilleure distance (ACO): {self.best_distance:.2f}")
            print(f"Amélioration: {improvement:.1f}%")
            print(f"Temps d'exécution: {elapsed_time:.2f}s")
        
        display.draw_route(self.graph.liste_lieux, self.best_route, self.pheromones, iteration, self.best_distance, elapsed_time, self.initial_distance)


        
# Main
if __name__ == "__main__":

    graph = Graph('graph_20.csv')
    tsp = TSP_ACO(graph)
    display = Display(600, 800)
    display.draw_nodes(graph.liste_lieux)
    tsp.run(0)
    display.dp.mainloop()