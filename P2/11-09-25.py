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
    [1,1,1,0,1,1,1,1,0,1,1,1,0,1,0,1,1,1,1,1],
    [1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,1,1,1,1,0,1,0,1,1,1,0,1,1,1,1,1,0,1],
    [1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,1],
    [1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1],
    [1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1],
    [1,0,1,1,1,0,1,1,1,1,1,1,1,1,0,1,1,1,0,1],
    [1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1],
    [1,1,1,0,1,1,0,1,0,1,1,1,0,1,1,0,1,1,1,1],
    [1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
    [1,0,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,0,1],
    [1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,1],
    [1,1,0,1,0,1,1,1,0,1,1,1,1,1,0,1,1,1,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
])

print(maze)

# Dibujar el laberinto
plt.imshow(maze, cmap="binary")
#plt.show()


##Posición Inicial
punto_inicial = (1,1)
## Meta
meta = (18,18)

##Reglas de movimiento
movimientos =[(0,1),(1,0),(0,-1),(-1,0)]

def dfs (maze,punto_inicial,meta):
    ##Asignamos el punto inicial
    pila = [(punto_inicial,[])]
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
        considerados += [nodo_actual]

        ## Nodo actual es la solución?
        if nodo_actual == meta : 
            return camino + [nodo_actual], considerados
        
        visitados[nodo_actual[0], nodo_actual[1]] = 1

        for vecinos in movimientos:
            vecino = (nodo_actual[0] + vecinos[0], nodo_actual[1] + vecinos[1])
            ##Corroborar que vecino este dentro del mapa  y que el vecino sea un nodo transitable
            ##Y que no se haya visitado ese nodo previamente 
            if (( 0 <= vecino[0] < filas ) and ( 0 <= vecino[1] <columnas) and (maze[vecino[0],vecino[1]] == 0) and (visitados[vecino[0],vecino[1]] == 0)):
                ##Concatenar a la pila
                pila += [(vecino, camino + [nodo_actual])]
                

    return None, considerados

def desplegar_laberinto (maze, camino, considerados):
    ##Desplejar el mapa 
    plt.imshow(maze, cmap = 'binary')
    ##Desplegue de considerados
    if considerados:
        ##Este for regresa todas las posiciones almacenadas en considerados 
        for i in considerados:
            plt.plot(i[1],i[0],'o', color = 'blue')
    if camino:
        ##Este for regresa todas las posiciones almacenadas en considerados 
        for i in camino:
            plt.plot(i[1],i[0],'o', color = 'red')
    plt.show()


##Llamado de las funciones

camino, considerados = dfs(maze, punto_inicial,meta)
desplegar_laberinto(maze, camino, considerados)



