import networkx as nx ##Uso y manejo de aristas y nodos
import matplotlib.pyplot as plt
import tracemalloc
import time 
import random

##Función para dibujar el grafo
def graficar_grafo(grafo, nodo_raiz, nodo_solucion):
    G = nx.Graph()
    ##Recorrer todos los nodos e ir añadiendo rayas
    ##entre cada nodo
    for nodo,hijos in grafo.items():
        #print(f"El padre {nodo}, tiene los hijos:")
        for hijo in hijos:
            G.add_edge(nodo,hijo)
            #print(hijo)
    
    pos = nx.spring_layout(G)
        

    # pos = {
    #     1: (0,0),
    #     2: (-1,-1),
    #     3: (1,-1),
    #     4: (-2,-2),
    #     5: (0,-2),
    #     6: (2,-2)
    # }

    #nx.draw(G, pos, with_labels = True, node_color ='lightblue', node_size = 400, font_size = 11, font_weight = 'bold')
    #plt.show()

def BFS (grafo, nodo_raiz, nodo_solucion):
    ##Implementamos BFS
    cola = [nodo_raiz]
    visitados = [False]*len(grafo)
    
    if nodo_raiz == nodo_solucion:
        print(f"Se encontro la solucion en el nodo {nodo_raiz}")
        return True
    
    visitados[nodo_raiz] = True

    while len(cola) > 0 :
        
        nodo_actual = cola.pop(0)
        print(f"El nodo actual es: {nodo_actual}")
        if nodo_actual == nodo_solucion:
            print(f"Se encontro la solucion en el nodo {nodo_actual}")
            return True
        for vecino in grafo[nodo_actual]:
            
            if visitados[vecino] == False:
                cola.append(vecino)
        
    


nodo_solucion = 1

grafo = {
    0: [2,3],
    1: [1,4,5],
    2: [1,6],
    3: [2],
    4: [2],
    5: [3]
}

BFS (grafo,5, nodo_solucion)
