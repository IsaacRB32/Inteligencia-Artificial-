import networkx as nx ##Uso y manejo de aristas y nodos
import matplotlib.pyplot as plt
import tracemalloc
import time 
import random

##FUNCIÓN PARA HACER EL LLENADO DE LAS POSICIONES
def funcionLLenadoraDePosicionesBFS (grafo, nodo_raiz):
    cola = [nodo_raiz]

    nivel = {nodo_raiz: 0}

    while len(cola) > 0 :
        nodo_actual = cola.pop(0)
        for vecino in grafo[nodo_actual]:
            if vecino not in nivel:
                nivel[vecino] = nivel[nodo_actual] + 1
                cola.append(vecino)

    niveles = {}
    for nodo, profundida in nivel.items():
        niveles.setdefault(profundida,[]).append(nodo)
    pos = {}
    for prof, nodos in niveles.items():
        cantidad = len(nodos)
        for i,nodo in enumerate(nodos):
            offset = (cantidad - 1) / 2
            x = i - offset 
            y = -prof
            pos[nodo] = (x,y)
    return pos


##FUNCIÓN DIBUJAR EL GRÁFO
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
        
    pos = funcionLLenadoraDePosicionesBFS(grafo,nodo_raiz)

    colores = []
    for nodo in G.nodes():
        if nodo == nodo_solucion:
            colores.append('#ff6666')
        elif nodo == nodo_raiz:
            colores.append('lightgreen')
        else:
            colores.append('lightblue')

    print(G.nodes())
    nx.draw(G, pos, with_labels = True, node_color=colores, node_size = 400, font_size = 11, font_weight = 'bold')
    plt.show()

## Definir funcion para el DFS 
def dfs(grafo, nodo_raiz, nodo_solucion):
    ## Medir consumo de memoria 
    tracemalloc.start()
    ## Medir tiempo 
    tiempo_inicial = time.time()
    ## Definimos el nodo raiz como 1er elemento 
    pila = [nodo_raiz]
    ## Definir una lista para los visitados 
    visitados = [False]*len(grafo) ##************

    if nodo_raiz == nodo_solucion:
        print(f"Se encontro la solucion en el nodo {nodo_raiz}")
        return True
    
    visitados[nodo_raiz] = True
    
    ## Bucle para ir buscando la solucion y si la pila se
    ## queda vacia es que no encontro solucion 
    while len(pila) > 0 :
        ## Sacar el ultimo nodo que habia en la pila y lo guardamos en nodo_actual 
        nodo_actual = pila[-1]
        print(f"El nodo actual es: {nodo_actual}")
        if nodo_actual == nodo_solucion:
            print(f"Se encontro la solucion en el nodo {nodo_actual}")
            return True
        
        ## Eliminar el ultimo nodo de la pila 
        pila = pila[:-1]
        
        ## Agregar y recorrer a los respectivos hijos 
        for hijos in grafo[nodo_actual]:
            ## Preguntar si algunos de los hijos ya fue visitado 
            if visitados[hijos] == False:
                visitados[hijos] = True
                pila = pila + [hijos]
        ## Falta preguntar si la solucion esta o no en lagunos de los 
        ## nodos hijos o en el nodo actual 
        
    tiempo_final = time.time()
    actual, pico = tracemalloc.get_traced_memory()
    
    tracemalloc.stop()
    
    print(f"El tiempo de ejecucion fue de: {tiempo_final - tiempo_inicial} segundos")
    print(f"La memoria actual consumida es de: {actual/10**6} y la memoria pico es de {pico/10**6}")    

grafo = {
    0 : [1,2,3],
    1 : [0,4,5],
    2 : [0,6,7],
    3 : [0,8],
    4 : [1],
    5 : [1,10],
    6 : [2],
    7 : [2],
    8 : [3,9],
    9 : [8],
    10 :[5]
}  
nodo_raiz = random.choice(list(grafo.keys()))
nodo_solucion = random.choice(list(grafo.keys()))
print(f"El nodo raiz es: {nodo_raiz}")
print(f"El nodo solucion es: {nodo_solucion}")
print("***Inicio de recorrido en profundidad***")
dfs(grafo,nodo_raiz,nodo_solucion)
graficar_grafo(grafo,nodo_raiz,nodo_solucion)