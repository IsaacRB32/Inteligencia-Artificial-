import networkx as nx ##Uso y manejo de aristas y nodos
import matplotlib.pyplot as plt

def graficar_grafo(grafo, nodo_raiz):
    G = nx.Graph()

##Recorrer todos los nodos e ir añadiendo rayas
##entre cada nodo
    for nodo,hijos in grafo.items():
        for hijo in hijos:
            G.add_edge(nodo,hijo)
        
    pos = nx.spring_layout(G)

    pos = {
        1: (0,0),
        2: (-1,-1),
        3: (1,-1),
        4: (-2,-2),
        5: (0,-2),
        6: (2,-2)
    }

    nx.draw(G, pos, with_labels = True, node_color ='lightblue', node_size = 400, font_size = 11, font_weight = 'bold')
    plt.show()

grafo = {
    1: [2,3],
    2: [1,4,5],
    3: [1,6],
    4: [2],
    5: [2],
    6: [3]
}

graficar_grafo(grafo,1)

##Para activar el entorno virtual = & "C:/Users/isaac/Documents/6TO SEMESTRE/IA/.venv/Scripts/Activate.ps1"
'''
Practica 1

1) del dfs agregar un numero aleatorio que este dentro de la cantidad de nodos que hay en el grafo y que el algoritmo dfs ahora se
detenga cuando encuentre la solución 

2) Agregarle al 1) la parte gráfica donde se muestre la solución en un color diferente

3) Implementar 1) y 2) pero con BFS
'''