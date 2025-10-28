import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
import tracemalloc
import time 

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
[1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,3,1,0,0,0,3,0,0,0,0,0,0,0,0,0,3,0,3,0,0,1],
[1,0,0,0,2,2,2,2,2,2,2,0,0,0,0,1,0,0,3,3,0,1,0,2,2,2,0,0,0,0,3,3,3,0,3,3,0,0,0,1],
[1,0,1,1,1,0,1,1,1,0,1,1,1,1,0,1,1,1,2,0,0,0,0,0,3,3,0,3,3,3,0,0,3,0,0,0,0,2,0,1],
[1,0,0,0,0,0,0,1,0,2,0,0,0,1,0,3,0,0,3,0,1,1,1,0,0,2,2,0,0,0,3,3,3,0,0,2,2,0,0,1],
[1,1,1,3,1,1,0,1,0,1,1,1,0,1,1,3,1,1,3,0,1,0,1,0,2,0,0,0,1,0,3,3,3,0,1,0,0,0,0,1],
[1,0,0,3,3,3,0,0,0,0,0,1,0,0,0,3,0,0,3,0,0,2,2,2,0,3,3,3,0,0,0,0,3,3,0,0,0,3,0,1],
[1,0,1,1,0,3,3,1,1,1,0,1,0,1,1,3,3,3,1,0,0,2,3,3,3,0,0,0,0,0,0,2,0,0,3,3,3,0,0,1],
[1,0,0,1,0,3,3,0,0,0,0,0,0,0,0,1,0,3,0,0,0,0,3,0,0,0,0,3,0,0,0,3,3,3,0,0,0,0,0,1],
[1,1,0,1,0,1,1,1,0,1,1,1,1,1,0,1,1,1,0,0,3,3,3,0,0,0,3,3,0,0,0,0,3,3,3,0,0,1,0,1],
[1,0,0,0,0,2,0,0,0,0,2,0,0,0,0,0,0,2,0,0,0,0,3,0,0,0,2,0,0,2,0,0,0,2,0,0,3,0,0,1],
[1,0,1,1,1,1,1,1,0,1,1,0,1,1,1,0,1,0,1,1,1,0,1,1,3,3,3,0,0,0,0,0,3,0,1,0,1,0,1,1],
[1,0,0,0,0,0,1,0,0,0,1,0,1,0,2,0,1,0,2,0,0,0,1,0,0,3,0,0,3,3,3,0,3,3,1,0,1,0,0,1],
[1,0,1,1,1,0,1,1,1,0,1,0,1,0,2,1,1,0,1,1,1,0,1,0,1,0,3,0,3,0,1,0,3,3,3,0,1,0,0,1],
[1,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,1,0,1,0,0,0,3,3,3,0,1,0,3,0,0,3,3,3,0,1],
[1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,0,1,1,1,0,1,0,1,1,3,3,3,0,1,0,3,0,1,0,1,1,1,1],
[1,0,0,0,0,0,0,0,3,3,3,0,0,0,0,0,0,0,0,0,0,2,2,0,3,3,3,0,0,0,0,0,3,0,0,3,0,0,0,1],
[1,0,1,1,1,0,1,1,3,3,0,1,0,1,1,1,0,2,3,3,0,0,0,0,2,2,0,0,3,3,0,0,3,3,0,0,3,0,0,1],
[1,0,0,0,0,0,0,0,3,0,3,3,3,0,0,0,0,3,0,0,0,0,2,0,0,0,0,0,3,3,3,0,0,0,0,0,3,3,0,1],
[1,0,2,2,0,0,2,1,0,0,3,0,3,0,1,0,0,3,3,3,0,0,2,0,0,0,0,0,3,3,3,0,0,0,0,0,3,3,0,1],
[1,0,1,1,3,0,0,1,2,0,0,0,3,3,1,0,0,0,3,3,3,0,0,0,0,0,0,3,3,3,0,0,0,3,3,3,0,0,0,1],
[1,0,0,1,3,3,3,1,0,0,0,0,3,0,0,0,2,2,2,0,0,0,0,0,0,0,3,3,0,0,0,0,3,0,0,0,3,0,0,1],
[1,0,0,1,3,0,0,1,1,3,3,0,1,1,1,1,1,0,0,0,0,0,2,0,3,3,3,0,0,0,2,2,2,0,0,3,3,3,0,1],
[1,0,0,0,3,0,0,0,3,3,3,0,0,0,1,0,0,0,0,0,0,3,3,0,0,0,3,3,3,0,0,0,0,3,0,0,3,0,0,1],
[1,1,1,0,3,3,3,0,0,0,0,0,1,3,3,0,0,0,3,3,0,0,0,0,0,0,0,3,3,0,0,0,0,3,3,3,0,0,0,1],
[1,0,1,1,0,0,3,0,1,0,0,0,1,3,3,0,0,0,3,0,0,0,0,0,2,2,0,0,3,3,0,0,0,0,3,0,0,0,0,1],
[1,0,0,0,0,0,3,0,1,1,1,1,1,0,0,0,0,3,3,3,0,0,0,0,0,3,0,0,0,3,3,3,0,0,0,0,3,0,0,1],
[1,0,0,0,0,0,3,0,0,3,3,3,0,0,0,0,0,3,3,0,0,0,0,0,0,3,3,3,0,0,0,0,3,0,0,0,3,0,0,1],
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
])


colores = ['white', 'black', 'lightgreen', 'lightblue']
# Índices: 0=camino, 1=pared, 2=verde (pasto - medio), 3=azul (agua - difícil)
cmap_personalizado = ListedColormap(colores)

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
def heuristica(nodo_actual,meta,n):
    x = abs(meta[0]-nodo_actual[0])
    y = abs(meta[1]-nodo_actual[1])
    return (x**n + y**n)**(1/n)

##función de Heuristica infinita 
def heuristica_inf(nodo_actual, meta):
    x = abs(meta[0] - nodo_actual[0])
    y = abs(meta[1] - nodo_actual[1])
    return max(x, y)

def heuristica_gen(nodo_actual, meta, n=2, modo=1):
    x = abs(meta[0] - nodo_actual[0])
    y = abs(meta[1] - nodo_actual[1])

    if modo == 1:
        # Lp: Manhattan si n=1, Euclidiana si n=2, etc.
        return (x**n + y**n)**(1/n)
    elif modo == 2:
        # L∞ (Chebyshev)
        return max(x, y)
    elif modo == 3:
        # L1 (Manhattan explícita)
        return x + y
    else:
        raise ValueError("Modo de heurística no válido.")

def A_estrella(mapa, punto_inicial, meta, n = 1, modo = 1):
    ## Medir consumo de memoria 
    tracemalloc.start()
    ## Medir tiempo 
    tiempo_inicial = time.time()
    lista_abierta = [(punto_inicial, 0, heuristica_gen(punto_inicial, meta, n, modo), [])] 
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
            return camino + [nodo_actual], considerados,tiempo_final - tiempo_inicial, pico / 10**6
        ##ya visitamos el nodo actual
        lista_cerrada[nodo_actual[0], nodo_actual[1]]=1

        for vecinos in movimientos_estrella:
            vecino = (nodo_actual[0] + vecinos[0], nodo_actual[1] + vecinos[1])
            ##Corroborar que el vecino esté dentro del mapa y que el vecino sea un nodo transitable y que no haya sido visitado ese nodo previamente
            ##Modificamos la condición para que acepte a 0, 2 y 3.
            if ((0 <= vecino[0] < filas) and (0 <= vecino[1] < columnas) and (mapa[vecino[0], vecino[1]] != 1) and (lista_cerrada[vecino[0], vecino[1]] == 0)):
                ## Vamos a calcular el peso con los nuevos valores que pusimos y 0 (0, 2 y 3).
                valor = mapa[vecino[0], vecino[1]]
                if valor == 0:
                    peso = 1
                elif valor == 2:
                    peso = 3
                elif valor == 3:
                    peso = 6
                ##Calcular el valor de la g nuevas, evaluando si es vertical/horizontal o diagonal
                ## Calcular el valor de g nuevo usando una métrica basada en la distancia real
                distancia = np.sqrt(vecinos[0]**2 + vecinos[1]**2)
                g_nuevo = g_actual + distancia * 10 * peso
                f_nuevo = g_nuevo + heuristica_gen(vecino, meta, n, modo)

            ##verificar si el vecino ya está en la lista abierta
                banderita_lista_abierta = False
                for nodo, g, f, camino_tmp in lista_abierta:
                    if nodo == vecino and f <= f_nuevo:
                        banderita_lista_abierta = True
                        break

                if banderita_lista_abierta == False:
                    # agrega el camino extendiendo con el nodo actual (no el vecino)
                    lista_abierta += [(vecino, g_nuevo, f_nuevo, camino + [nodo_actual])]
    tiempo_final = time.time()
    actual, pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print("No se encontro un camino hasta la meta")
    return [], considerados, tiempo_final - tiempo_inicial, pico / 10**6

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
            #plt.pause(0.01)
    ## Mostrar camino encontrado
    if camino:
        for i in camino:
            axes.plot(i[1], i[0], 'o', color='red')
            #plt.pause(0.01)


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

    n = int(input("Ingresa un valor para n (n >= 2): "))

    lienzo, axes = plt.subplots(1, 3, figsize=(15, 5))
    lienzo.suptitle("Comparación L₁, Lₚ y L∞")

    # L1 (Manhattan)
    camino1, cons1, t1, m1 = A_estrella(maze, punto_inicial, meta, 1, modo = 3)
    desplegar_laberinto(maze, camino1, cons1, axes[0], "A* L₁ (Manhattan)", t1, m1)

    # Lp (Minkowski)
    camino2, cons2, t2, m2 = A_estrella(maze, punto_inicial, meta, n, modo = 1)
    desplegar_laberinto(maze, camino2, cons2, axes[1], f"A* Lₚ (n={n})", t2, m2)

    # L∞ (Chebyshev)
    camino3, cons3, t3, m3 = A_estrella(maze, punto_inicial, meta, 1, modo = 2)
    desplegar_laberinto(maze, camino3, cons3, axes[2], "A* L∞ (Chebyshev)", t3, m3)

    plt.show()
    break


