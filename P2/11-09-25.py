import numpy as np
import matplotlib.pyplot as plt

# Laberinto 20x20 con 1 = pared y 0 = camino
maze = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,1,0,1,0,0,1,0,0,0,1,0,1,0,0,0,1],
    [1,1,1,0,1,0,1,0,1,1,0,1,0,1,0,1,1,1,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,1],
    [1,0,1,1,1,0,1,1,1,1,0,1,1,1,1,1,0,1,0,1],
    [1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1],
    [1,1,1,0,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1],
    [1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1],
    [1,0,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1],
    [1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,1],
    [1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1],
    [1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1],
    [1,0,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,0,1],
    [1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1],
    [1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1],
    [1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
    [1,0,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,0,1],
    [1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,1],
    [1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
])

print(maze)

# Dibujar el laberinto
plt.imshow(maze, cmap="binary")
plt.show()


##Posición Inicial
punto_inicial = (1,1)
## Meta
meta = ()

##Reglas de movimiento
movimientos =[(0,1),(1,0),(0,-1),(-1,0)]

def dfs (maze,punto_inicial,meta):
    ##Asignamos el punto inicial
    pila = [(punto_inicial),[]]
    filas = np.shape(maze)[0]
    columnas = np.shape(maze)[1]

    visitados = np.zeros((filas,columnas))
    ## A través de la variable considerados se irá guardando tanto los nodod ya visitados
    ##  y el camino que se irpa construyendo 
    considerados = []

    while len(pila) > 0:
        nodo_actual, camino = pila[-1]
        pila = pila[:-1]

        ## Guardar el nodo inical dentro de la variable considerados 
        considerados += nodo_actual

        ## Nodo actual es la solución?
        if nodo_actual == meta : 
            return camino + [nodo_actual], considerados
        
        visitados[nodo_actual[0], nodo_actual[1]]

        for vecinos in movimientos:
            vecino = (nodo_actual[0] + vecinos[0], nodo_actual[1] + vecinos[1])
            