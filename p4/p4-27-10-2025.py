import numpy as np
import matplotlib.pyplot as plt
import random
import tracemalloc
import time 
from matplotlib.colors import ListedColormap


# Laberinto 40x40 con 3 = agua, 2 = pasto, 1 = pared y 0 = camino
maze = np.array([
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
[1,0,0,2,0,0,1,1,0,3,3,3,2,0,1,1,0,3,3,3,0,1,0,2,2,0,1,1,0,3,3,3,0,2,1,0,1,0,0,1],
[1,0,1,0,1,0,2,1,1,0,3,0,1,1,1,0,1,0,3,0,1,1,1,0,0,0,0,1,0,3,0,3,0,1,1,0,1,0,0,1],
[1,0,2,0,0,0,2,0,1,0,0,0,1,0,0,0,1,0,3,3,1,0,1,0,1,1,1,0,1,0,3,0,3,0,1,0,2,0,0,1],
[1,0,1,1,1,1,1,0,1,1,3,0,1,0,1,0,1,0,3,0,0,0,0,0,1,0,2,0,0,0,3,0,3,3,1,0,1,0,0,1],
[1,0,0,0,2,0,1,0,0,3,1,0,1,0,2,0,1,0,3,3,1,1,1,0,1,0,1,1,1,0,3,3,3,0,1,0,1,0,0,1],
[1,1,0,1,0,0,1,0,1,0,1,0,1,0,2,0,1,0,0,0,1,0,1,0,1,0,2,2,0,0,2,0,3,3,1,0,1,0,0,1],
[1,0,0,1,0,2,0,0,2,2,2,2,2,2,2,0,1,0,3,0,1,0,1,0,1,1,1,1,1,0,2,0,3,3,1,0,1,0,0,1],
[1,0,1,1,0,1,1,0,1,0,1,0,3,0,1,0,0,0,3,3,1,0,1,0,2,2,3,3,0,0,2,0,3,0,1,0,1,0,0,1],
[1,0,0,0,0,1,0,0,3,3,0,1,0,0,0,0,0,1,3,3,3,0,0,0,0,0,2,0,0,2,2,2,0,0,3,3,3,0,0,1],
[1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,3,1,0,0,0,3,0,0,0,1,0,0,0,0,0,3,0,3,0,0,1],
[1,0,0,0,2,2,2,2,2,2,2,0,0,0,0,1,0,0,3,3,0,1,0,2,2,2,0,0,1,0,3,3,1,1,1,3,0,0,0,1],
[1,0,1,1,1,0,1,1,1,0,1,1,1,1,0,1,1,1,2,0,0,0,0,0,3,1,0,3,3,3,0,0,3,0,0,0,0,2,0,1],
[1,0,0,0,0,0,0,1,0,2,0,0,0,1,0,3,0,0,3,0,1,1,1,0,0,1,1,1,0,0,3,3,3,0,0,2,2,0,0,1],
[1,1,1,3,1,1,0,1,0,1,1,1,0,1,1,3,1,1,3,0,1,0,1,0,2,1,0,2,2,0,3,0,0,0,1,0,0,0,0,1],
[1,0,0,3,3,3,0,0,0,0,0,1,0,0,0,3,0,0,3,0,0,2,2,2,0,1,3,2,1,0,1,3,3,3,0,0,0,3,0,1],
[1,0,1,1,0,3,3,1,1,1,0,1,0,1,1,3,3,3,1,0,0,2,3,1,3,0,0,0,0,1,2,2,0,0,3,3,3,0,0,1],
[1,0,0,1,0,3,3,0,0,0,0,0,0,0,0,1,0,3,0,0,1,1,1,1,1,1,0,2,1,1,1,0,0,3,0,0,0,0,0,1],
[1,1,0,1,0,1,1,1,0,1,1,1,1,1,0,1,1,1,0,0,3,3,3,1,0,0,3,2,1,2,1,0,1,3,3,0,0,1,0,1],
[1,0,0,0,0,2,0,0,0,0,2,0,0,0,0,0,0,2,0,0,0,0,3,0,0,0,2,2,2,2,0,0,1,2,0,0,3,0,0,1],
[1,0,1,1,1,1,1,1,0,1,1,0,1,1,1,0,1,0,1,1,1,0,1,1,3,3,3,0,0,0,0,0,3,0,1,0,1,0,1,1],
[1,0,0,0,0,0,1,0,0,0,1,0,1,0,2,0,1,0,2,0,0,0,1,0,0,3,0,1,3,3,3,0,3,3,1,0,1,0,0,1],
[1,0,1,1,1,0,1,1,1,0,1,0,1,0,2,1,1,0,1,1,1,0,1,0,1,0,3,1,3,0,1,0,3,3,3,0,1,0,0,1],
[1,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,1,0,1,0,0,0,3,1,3,0,1,0,3,0,0,3,3,3,0,1],
[1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,0,1,1,1,0,1,0,1,1,3,3,3,0,1,0,3,0,1,0,1,1,1,1],
[1,0,0,0,0,0,0,0,3,3,3,0,0,0,0,0,0,0,0,0,0,2,2,0,3,3,3,0,0,0,0,0,3,0,0,3,0,1,0,1],
[1,0,1,1,1,0,1,1,3,3,0,1,0,1,1,1,0,2,3,3,0,0,0,0,2,2,1,1,1,1,0,0,3,3,1,0,3,1,0,1],
[1,0,0,0,0,0,0,0,3,0,3,3,3,0,0,0,0,3,0,0,0,0,2,0,1,0,0,0,3,1,3,1,1,1,1,0,3,1,0,1],
[1,0,2,2,0,0,2,1,0,0,3,0,3,0,1,0,0,3,3,3,0,0,2,0,1,0,0,0,3,1,3,0,0,0,1,0,3,3,0,1],
[1,0,1,1,3,0,0,1,2,0,0,0,3,3,1,0,0,0,3,3,3,0,0,0,1,1,1,1,3,3,0,0,0,3,1,3,0,0,0,1],
[1,0,0,1,3,3,3,1,0,0,0,0,3,0,0,0,2,2,1,1,1,1,1,0,1,0,3,1,1,1,0,1,1,1,1,1,1,0,0,1],
[1,0,0,1,3,0,0,1,1,3,3,0,1,1,1,1,1,0,0,0,1,0,2,0,1,3,3,0,0,0,2,1,2,0,0,3,1,3,0,1],
[1,0,0,0,3,0,0,0,3,3,3,0,0,0,1,0,0,0,0,1,1,1,3,0,1,1,1,3,3,0,0,1,0,1,1,1,1,0,0,1],
[1,1,1,0,3,3,3,0,0,0,0,0,1,3,3,0,0,0,3,3,0,0,0,0,0,0,0,3,3,0,0,1,0,3,3,3,0,0,0,1],
[1,0,1,1,0,0,3,0,1,0,0,0,1,3,3,0,0,0,1,0,0,0,0,0,2,1,0,0,3,3,0,1,1,1,1,0,0,0,0,1],
[1,0,0,0,0,0,3,0,1,1,1,1,1,0,0,0,0,3,1,1,1,1,1,0,0,1,0,0,0,3,3,3,0,0,1,0,3,0,0,1],
[1,0,0,0,0,0,3,0,0,3,3,3,0,0,0,0,0,3,3,0,0,0,0,0,0,1,3,3,0,0,0,0,3,0,0,0,3,0,0,1],
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
])
colores = ['white', 'black', 'lightgreen', 'lightblue']

cmap_personalizado = ListedColormap(colores)
        
print(maze)
# Costos base según terreno
costo_terreno = {
    0: 10,   ## noraml
    2: 15,   ## pasto
    3: 20    ## agua
}

##Reglas de movimiento
movimientos_estrella = [
    (-1, 0),  # arriba
    (1, 0),   # abajo
    (0, -1),  # izquierda
    (0, 1),   # derecha
    (-1, -1), # arriba-izquierda
    (-1, 1),  # arriba-derecha
    (1, -1),  # abajo-izquierda
    (1, 1)    # abajo-derecha
]

##funcion de Heuristica
def heuristica(nodo_actual,meta):
    return (abs((meta[0]-nodo_actual[0]) + (meta[1]-nodo_actual[1])))

def A_estrella(mapa, punto_inicial, meta):
    ## Medir consumo de memoria 
    tracemalloc.start()
    ## Medir tiempo 
    tiempo_inicial = time.time()
    lista_abierta = [(punto_inicial, 0, heuristica(punto_inicial, meta), [])] 
    ## definir a los considerados 
    filas = np.shape(mapa)[0]
    columnas = np.shape(mapa)[1]
    lista_cerrada = np.zeros((filas, columnas))
    ## A traves de la varible considerados vamos a ir guardando los nodos que ya hemos visitado y el camino que se irá construyendo
    considerados = []
    while len(lista_abierta) > 0:
        menor_f = lista_abierta[0][2]
        nodo_actual,g_actual, f_actual, camino = lista_abierta[0]
        ##evaluar los nodos vecinos/hijos
        indice_menor_f = 0
        for i in range(1, len(lista_abierta)):
            ##Extraemos F del nodo a evaluar y lo comparamos con el menor F_actual
            if lista_abierta[i][2] < f_actual:
                menor_f = lista_abierta[i][2]
                nodo_actual, g_actual, f_actual, camino = lista_abierta[i]
                indice_menor_f = i
        ##Eliminar el nodo actual de la lista abierta
        lista_abierta = lista_abierta[:indice_menor_f] + lista_abierta[indice_menor_f+1:]
        ## guardar en considerados el nodo actual
        considerados += [nodo_actual]


        ##Evaluamos si es la meta
        if nodo_actual == meta:
            tiempo_final = time.time()
            actual, pico = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            #return camino + [nodo_actual], considerados,tiempo_final - tiempo_inicial, pico / 10**6
            return camino + [nodo_actual], considerados, tiempo_final - tiempo_inicial, pico / 10**6, len(considerados)
        ##ya visitamos el nodo actual
        lista_cerrada[nodo_actual[0], nodo_actual[1]]=1

        for vecinos in movimientos_estrella:
            vecino = (nodo_actual[0] + vecinos[0], nodo_actual[1] + vecinos[1])
            ##Corroborar que el vecino esté dentro del mapa y que el vecino sea un nodo transitable y que no haya sido visitado ese nodo previamente
            if ((0 <= vecino[0] < filas) and (0 <= vecino[1] < columnas) and (mapa[vecino[0], vecino[1]] != 1) and (lista_cerrada[vecino[0], vecino[1]] == 0)):
                tipo = mapa[vecino[0], vecino[1]]
            ##Calcular el valor de la g nuevas, evaluando si es vertical/horizontal o diagonal
                if (abs(vecinos[0]) + abs(vecinos[1])) == 2:
                    g_nuevo = g_actual + costo_terreno[tipo]*14
                else:
                    g_nuevo = g_actual + costo_terreno[tipo]*10
                f_nuevo = g_nuevo + heuristica(vecino, meta)

            ##verificar si el vecino ya está en la lista abierta
                banderita_lista_abierta = False
                for nodo, g, f, camino_tmp in lista_abierta:
                    if nodo == vecino and f <= f_nuevo:
                        banderita_lista_abierta = True
                        break

                if banderita_lista_abierta == False:
                    # agrega el camino extendiendo con el nodo actual (no el vecino)
                    lista_abierta += [(vecino, g_nuevo, f_nuevo, camino + [nodo_actual])]


##funcion de la nueva Heuristica
def heuristica_nueva(nodo_actual,meta):
    #return (abs((meta[0]-nodo_actual[0]) + (meta[1]-nodo_actual[1])))
    ##Vamos a cambiar la heurística de L1 a L2
    return (abs(meta[0]-nodo_actual[0])**2 + abs(meta[1]-nodo_actual[1])**2)**(1/2)

def A_estrella_nueva(mapa, punto_inicial, meta):
    ## Medir consumo de memoria 
    tracemalloc.start()
    ## Medir tiempo 
    tiempo_inicial = time.time()
    lista_abierta = [(punto_inicial, 0, heuristica_nueva(punto_inicial, meta), [])] 
    ## definir a los considerados 
    filas = np.shape(mapa)[0]
    columnas = np.shape(mapa)[1]
    lista_cerrada = np.zeros((filas, columnas))
    ## A traves de la varible considerados vamos a ir guardando los nodos que ya hemos visitado y el camino que se irá construyendo
    considerados = []
    while len(lista_abierta) > 0:
        menor_f = lista_abierta[0][2]
        nodo_actual,g_actual, f_actual, camino = lista_abierta[0]

        ##evaluar los nodos vecinos/hijos
        indice_menor_f = 0
        for i in range(1, len(lista_abierta)):
            ##Extraemos F del nodo a evaluar y lo comparamos con el menor F_actual
            if lista_abierta[i][2] < f_actual:
                menor_f = lista_abierta[i][2]
                nodo_actual, g_actual, f_actual, camino = lista_abierta[i]
                indice_menor_f = i
        ##Eliminar el nodo actual de la lista abierta
        lista_abierta = lista_abierta[:indice_menor_f] + lista_abierta[indice_menor_f+1:]
        ## guardar en considerados el nodo actual
        considerados += [nodo_actual]

        ##Evaluamos si es la meta
        if nodo_actual == meta:
            tiempo_final = time.time()
            actual, pico = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            #return camino + [nodo_actual], considerados,tiempo_final - tiempo_inicial, pico / 10**6
            return camino + [nodo_actual], considerados, tiempo_final - tiempo_inicial, pico / 10**6, len(considerados)
        ##ya visitamos el nodo actual
        lista_cerrada[nodo_actual[0], nodo_actual[1]]=1

        for vecinos in movimientos_estrella:
            vecino = (nodo_actual[0] + vecinos[0], nodo_actual[1] + vecinos[1])
            ##Corroborar que el vecino esté dentro del mapa y que el vecino sea un nodo transitable y que no haya sido visitado ese nodo previamente
            if ((0 <= vecino[0] < filas) and (0 <= vecino[1] < columnas) and (mapa[vecino[0], vecino[1]] != 1) and (lista_cerrada[vecino[0], vecino[1]] == 0)):
                # if (abs(vecinos[0]) + abs(vecinos[1])) == 2:
                #     g_nuevo = g_actual + 14 
                # else:
                #     g_nuevo = g_actual + 10
                ##Vamos a cambiar la métrica de medición de g, pasando de L2 a L3
                # Costos base según terreno
                ##Ya no es necesario hacer uso de in if para separar movimientos ortogonales y diagonales, porque la raíz ya lo hace
                ##Te lo dice la variable 'distancia'
                tipo = mapa[vecino[0], vecino[1]]
                distancia = (abs(vecinos[0])**3 + abs(vecinos[1])**3)**(1/3)
                g_nuevo = g_actual + costo_terreno[tipo]*distancia
                f_nuevo = g_nuevo + heuristica_nueva(vecino, meta)
            ##verificar si el vecino    ya está en la lista abierta
                banderita_lista_abierta = False 
                for nodo, g, f, camino_tmp in lista_abierta:
                    if nodo == vecino and f <= f_nuevo:
                        banderita_lista_abierta = True
                        break

                if banderita_lista_abierta == False:
                    # agrega el camino extendiendo con el nodo actual (no el vecino)
                    lista_abierta += [(vecino, g_nuevo, f_nuevo, camino + [nodo_actual])]



def Dijkstra (mapa, punto_inicial, meta):
    ## Medir consumo de memoria 
    tracemalloc.start()
    ## Medir tiempo 
    tiempo_inicial = time.time()
    lista_abierta = [(punto_inicial, 0, 0, [])]
    ## definir a los considerados 
    filas = np.shape(mapa)[0]
    columnas = np.shape(mapa)[1]
    lista_cerrada = np.zeros((filas, columnas))
    ## A traves de la varible considerados vamos a ir guardando los nodos que ya hemos visitado y el camino que se irá construyendo
    considerados = []
    while len(lista_abierta) > 0:
        menor_f = lista_abierta[0][2]
        nodo_actual,g_actual, f_actual, camino = lista_abierta[0]

        ##evaluar los nodos vecinos/hijos
        indice_menor_f = 0
        for i in range(1, len(lista_abierta)):
            ##Extraemos F del nodo a evaluar y lo comparamos con el menor F_actual
            if lista_abierta[i][2] < menor_f:
                menor_f = lista_abierta[i][2]
                nodo_actual, g_actual, f_actual, camino = lista_abierta[i]
                indice_menor_f = i
        ##Eliminar el nodo actual de la lista abierta
        lista_abierta = lista_abierta[:indice_menor_f] + lista_abierta[indice_menor_f+1:]
        ## guardar en considerados el nodo actual
        considerados += [nodo_actual]

        ##Evaluamos si es la meta
        if nodo_actual == meta:
            tiempo_final = time.time()
            actual, pico = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            #return camino + [nodo_actual], considerados,tiempo_final - tiempo_inicial, pico / 10**6
            return camino + [nodo_actual], considerados, tiempo_final - tiempo_inicial, pico / 10**6, len(considerados)
        ##ya visitamos el nodo actual
        lista_cerrada[nodo_actual[0], nodo_actual[1]]=1

        for vecinos in movimientos_estrella:
            vecino = (nodo_actual[0] + vecinos[0], nodo_actual[1] + vecinos[1])
            ##Corroborar que el vecino esté dentro del mapa y que el vecino sea un nodo transitable y que no haya sido visitado ese nodo previamente
            if ((0 <= vecino[0] < filas) and (0 <= vecino[1] < columnas) and (mapa[vecino[0], vecino[1]] != 1) and (lista_cerrada[vecino[0], vecino[1]] == 0)):
                tipo = mapa[vecino[0], vecino[1]]
                distancia = (abs(vecinos[0])**2 + abs(vecinos[1])**2)**(1/2)
                g_nuevo = g_actual + costo_terreno[tipo]*distancia
                f_nuevo = g_nuevo
            ##verificar si el vecino ya está en la lista abierta
                banderita_lista_abierta = False
                for nodo, g, f, camino_tmp in lista_abierta:
                    if nodo == vecino and f <= f_nuevo:
                        banderita_lista_abierta = True
                        break

                if banderita_lista_abierta == False:
                    # agrega el camino extendiendo con el nodo actual (no el vecino)
                    lista_abierta += [(vecino, g_nuevo, f_nuevo, camino + [nodo_actual])]

def desplegar_laberinto (maze, camino, considerados, axes, titulo, tiempo, memoria):
    ##Desplejar el mapa 
    #axes.imshow(maze, cmap = 'binary')
    axes.imshow(maze, cmap=cmap_personalizado)

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
        f"Tiempo: {tiempo:.10f} s\nMemoria pico: {memoria:.5f} MB\nNodos explorados: {len(considerados)}",
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
            #plt.pause(0.001)
    ## Mostrar camino encontrado
    if camino:
        for i in camino:
            axes.plot(i[1], i[0], 'o', color='red')
            #plt.pause(0.001)

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
    print("Puntos validos para ejecutar")

    lienzo, axes = plt.subplots(1, 3, figsize=(10, 5))
    
    lienzo.suptitle("Comparacion entre A*-Original, A*-Nuevo y Dijkstra")

    ##Le mandamos el segundo axes a bfs
    #camino_estrella, considerados_estrella, tiempo_estrella, memoria_estrella  = A_estrella(maze, punto_inicial, meta)
    camino_estrella, considerados_estrella, tiempo_estrella, memoria_estrella, nodos_estrella = A_estrella(maze, punto_inicial, meta)
    desplegar_laberinto(maze, camino_estrella, considerados_estrella, axes[0],"A*-Original", tiempo_estrella, memoria_estrella)

    camino_estrella, considerados_estrella, tiempo_estrella, memoria_estrella, nodos_estrella = A_estrella_nueva(maze, punto_inicial, meta)
    desplegar_laberinto(maze, camino_estrella, considerados_estrella, axes[1],"A*-Nueva", tiempo_estrella, memoria_estrella)

    camino_estrella, considerados_estrella, tiempo_estrella, memoria_estrella, nodos_estrella = Dijkstra(maze, punto_inicial, meta)
    desplegar_laberinto(maze, camino_estrella, considerados_estrella, axes[2],"Dijkstra", tiempo_estrella, memoria_estrella)

    plt.show()
    break

