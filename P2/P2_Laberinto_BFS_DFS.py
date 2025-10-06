import numpy as np
import matplotlib.pyplot as plt
import random
import tracemalloc
import time 

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

##Reglas de movimiento
movimientos =[(0,1),(1,0),(0,-1),(-1,0)]


def dfs (maze,punto_inicial,meta):
    ## Medir consumo de memoria 
    tracemalloc.start()
    ## Medir tiempo 
    tiempo_inicial = time.time()

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

        ##Marcamos como visitado 
        visitados[nodo_actual[0], nodo_actual[1]] = 1
        ## Nodo actual es la solución?
        if nodo_actual == meta : 
            tiempo_final = time.time()
            actual, pico = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return camino + [nodo_actual], considerados, tiempo_final - tiempo_inicial, pico / 10**6

        for vecinos in movimientos:
            vecino = (nodo_actual[0] + vecinos[0], nodo_actual[1] + vecinos[1])
            ##Corroborar que vecino este dentro del mapa  y que el vecino sea un nodo transitable
            ##Y que no se haya visitado ese nodo previamente 
            if (( 0 <= vecino[0] < filas ) and ( 0 <= vecino[1] <columnas) and (maze[vecino[0],vecino[1]] == 0) and (visitados[vecino[0],vecino[1]] == 0)):
                ##Concatenar a la pila
                pila += [(vecino, camino + [nodo_actual])]

    tiempo_final = time.time()
    actual, pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()          
    return None, considerados, tiempo_final - tiempo_inicial, pico / 10**6

def BFS (maze,punto_inicial,meta):
    ## Medir consumo de memoria 
    tracemalloc.start()
    ## Medir tiempo 
    tiempo_inicial = time.time()

    ##Asignamos el punto inicial
    ##Cambiamos la pila LIFO por una cola FIFO
    cola = [(punto_inicial,[])]
    filas = np.shape(maze)[0]
    columnas = np.shape(maze)[1]
    visitados = np.zeros((filas,columnas))
    
    ## A través de la variable considerados se irá guardando tanto los nodod ya visitados
    ##  y el camino que se irpa construyendo 
    considerados = []
    visitados[punto_inicial[0], punto_inicial[1]] = 1
    while len(cola) > 0:
        ##Ahora sacamos el elemento más antiguo en la cola FIFO
        nodo_actual, camino = cola[0]
        cola = cola[1:]

        ## Guardar el nodo inical dentro de la variable considerados 
        considerados += [nodo_actual]
        
        ##Marcamos como visitado
        visitados[nodo_actual[0], nodo_actual[1]] = 1
        
        ## Nodo actual es la solución?
        if nodo_actual == meta : 
            tiempo_final = time.time()
            actual, pico = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return camino + [nodo_actual], considerados, tiempo_final - tiempo_inicial, pico / 10**6
        
        ##visitados[nodo_actual[0], nodo_actual[1]] = 1

        ##movimientos =[(0,1),(1,0),(0,-1),(-1,0)]
        for vecinos in movimientos:
            ##Se definene las 4 direcciones posibles 
            vecino_calculado = (nodo_actual[0] + vecinos[0], nodo_actual[1] + vecinos[1])
            ##Corroborar que vecino este dentro del mapa  y que el vecino sea un nodo transitable
            ##Y que no se haya visitado ese nodo previamente 
            if (( 0 <= vecino_calculado[0] < filas ) and ( 0 <= vecino_calculado[1] <columnas) and (maze[vecino_calculado[0],vecino_calculado[1]] == 0) and (visitados[vecino_calculado[0],vecino_calculado[1]] == 0)):
                ##Concatenar a la cola
                cola += [(vecino_calculado, camino + [nodo_actual])]
                visitados[vecino_calculado[0],vecino_calculado[1]] = 1
    tiempo_final = time.time()
    actual, pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return None, considerados, tiempo_final - tiempo_inicial, pico / 10**6

# def desplegar_laberinto (maze, camino, considerados):
#     ##Desplejar el mapa 
#     plt.imshow(maze, cmap = 'binary')
#     ##Desplegue de considerados
#     if considerados:
#         ##Este for regresa todas las posiciones almacenadas en considerados 
#         for i in considerados:
#             plt.plot(i[1],i[0],'o', color = 'blue')
#             plt.pause(0.1)
#     if camino:
#         ##Este for regresa todas las posiciones almacenadas en considerados 
#         for i in camino:
#             plt.plot(i[1],i[0],'o', color = 'red')
#             plt.pause(0.1)
#     plt.show()

def desplegar_laberinto (maze, camino, considerados, axes, titulo, tiempo, memoria):
    ##Desplejar el mapa 
    axes.imshow(maze, cmap = 'binary')

    inicio = camino[0]
    meta = camino[-1]

    axes.set_title(titulo)
    ## Le quitamos los nombres a los axis 
    axes.set_xticks([])
    axes.set_yticks([])
    axes.text(
        ##A la mitad de lo ancho
        0.5, 
        ##Un poco debajo del xaxis
        -0.07, 
        f"Tiempo: {tiempo:.10f} s\nMemoria pico: {memoria:.5f} MB",
        ##lo pasoa a sistema de ref del axes
        transform=axes.transAxes,
        ha='center',
        va='top',
        fontsize=9,
        color='black'
    )
    ## Muestra inicio y meta
    axes.plot(inicio[1], inicio[0], '*', color='lime', markersize=10, zorder=6)
    axes.plot(meta[1],   meta[0],   '*', color='gold', markersize=10, zorder=6)
     ## Mostrar nodos considerados
    if considerados:
        for i in considerados:
            axes.plot(i[1], i[0], 'o', color='blue')
            plt.pause(0.1)
    ## Mostrar camino encontrado
    if camino:
        for i in camino:
            axes.plot(i[1], i[0], 'o', color='red')
            plt.pause(0.1)


##------------------------------------ MAIN ------------------------------------


##Llamado de las funciones
cantidad_filas = np.shape(maze)[0]
cantidad_columnas = np.shape(maze)[1]

coordenadas = []

for i in range(cantidad_filas):
    for j in range(cantidad_columnas):
        coordenadas += [(i,j)]

while True :
    ##Posición Inicial
    punto_inicial = random.choice(coordenadas)
    ## Meta
    meta = random.choice(coordenadas)

    print(f"Punto inicial: {punto_inicial}")
    print(f"Meta: {meta}")
    
    ## Validacion para lugar transitable
    if maze[punto_inicial[0], punto_inicial[1]] == 1 or maze[meta[0], meta[1]] == 1:
        print("La meta o el punto inicial estan en un lugar no transitable")
        continue
    ## Validacion para que no sean dos puntos iguales
    if punto_inicial == meta:
        print("El punto inicial y la meta son iguales")
        continue
    
    ##
    print("Puntos validos para ejecutar DFS y BFS")

    lienzo, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    lienzo.suptitle("Comparacion entre DFS y BFS")

    camino_dfs, considerados_dfs, tiempo_dfs, memoria_dfs = dfs(maze, punto_inicial, meta)
    ##Le mandamos el primer axes a dfs
    desplegar_laberinto(maze, camino_dfs, considerados_dfs, axes[0], "DFS", tiempo_dfs, memoria_dfs)
    ##Le mandamos el segundo axes a bfs
    camino_bfs, considerados_bfs, tiempo_bfs, memoria_bfs = BFS(maze, punto_inicial, meta)
    desplegar_laberinto(maze, camino_bfs, considerados_bfs, axes[1], "BFS", tiempo_bfs, memoria_bfs)

    plt.show()
    break

