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
movimientos =[(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]

##Función de heuristica
def heuristica(nodo_actual, meta):
    return (abs(meta[0]-nodo_actual[0]) + abs(meta[1]-nodo_actual[1]))


def desplegar_laberinto (maze, camino = None, considerados = None):
    ##Desplejar el mapa 
    plt.imshow(maze, cmap = 'binary')
    ##Desplegue de considerados
    if considerados:
        ##Este for regresa todas las posiciones almacenadas en considerados 
        for i in considerados:
            plt.plot(i[1],i[0],'o', color = 'blue')
            plt.pause(0.1)
    if camino:
        ##Este for regresa todas las posiciones almacenadas en considerados 
        for j in camino:
            plt.plot(j[1],j[0],'o', color = 'red')
            plt.pause(0.1)
    plt.show()

def A_estrella(mapa, punto_inicial, meta):
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
            return camino + [nodo_actual], considerados
        ##ya visitamos el nodo actual
        lista_cerrada[nodo_actual[0], nodo_actual[1]]=1

        for vecinos in movimientos:
            vecino = (nodo_actual[0] + vecinos[0], nodo_actual[1] + vecinos[1])
            ##Corroborar que el vecino esté dentro del mapa y que el vecino sea un nodo transitable y que no haya sido visitado ese nodo previamente
            if ((0 <= vecino[0] < filas) and (0 <= vecino[1] < columnas) and (mapa[vecino[0], vecino[1]] == 0) and (lista_cerrada[vecino[0], vecino[1]] == 0)):
            ##Calcular el valor de la g nuevas, evaluando si es vertical/horizontal o diagonal
                if (abs(vecinos[0]) + abs(vecinos[1])) == 2:
                    g_nuevo = g_actual + 14
                else:
                    g_nuevo = g_actual + 10
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


    ##(Punto_inicial,g , f, camino)
    lista_abierta = [(punto_inicial,0,heuristica(punto_inicial,meta),[])]
    print(len(lista_abierta))
    ##La priemera vez en f, se considera solamente h pues g vale 0
    # liste_abierte  = (nodo, g, f, camino)
    considerados=[]
    filas = np.shape(maze)[0]
    columnas = np.shape(maze)[1]
    lista_cerrada = np.zeros((filas,columnas))

    #lista_cerrada = np.zeros_like(maze)
    while len(lista_abierta)>0:
        menor_f = lista_abierta[0][2]
        nodo_actual,g_actual,f_actual,camino_actual = lista_abierta[0]

        ## Se reccore la posicion 2 de la fila i (distancia Manhattan) de la lista abierta preguntando
        ## cual es el de menor valor 
        indice_menor_f = 0

        for i in range(1,len(lista_abierta)):
            ##Extraer F del nodo a evaluar y lo comparamos con el valor de f_actual
            if lista_abierta[i][2] < menor_f :
                menor_f = lista_abierta[i][2]
                nodo_actual,g_actual,f_actual,camino = lista_abierta[i]
                indice_menor_f = i
        ## Eliminar el nodo de la lista abierta
        lista_abierta = lista_abierta[:indice_menor_f] + lista_abierta[indice_menor_f+1:]
        ##Guardar en considerados el nodo actual
        considerados += [nodo_actual]
        if nodo_actual == meta:
            return camino+[nodo_actual],considerados
        
        lista_cerrada[nodo_actual[0],nodo_actual[1]] = 1

        for vecinos in movimientos:
            vecino = (nodo_actual[0] + vecinos[0], nodo_actual[1] + vecinos[1])
            ##Corroborar que vecino este dentro del mapa  y que el vecino sea un nodo transitable
            ##Y que no se haya visitado ese nodo previamente 
            if (( 0 <= vecino[0] < filas ) and ( 0 <= vecino[1] <columnas) and (maze[vecino[0],vecino[1]] == 0) and (lista_cerrada[vecino[0],vecino[1]] == 0)):
                ##Calcular el valor de las g_nuevas. evauando si es vertical/horizontal
                if(abs(vecinos[0])+abs(vecinos[1])) == 2:
                    g_nuevo = g_actual + 14
                else:
                    g_nuevo = g_actual + 10
                f_nuevo = g_nuevo + heuristica(vecino, meta)
                ## Verificar si el vecino no fue guardado previamente en la fila abierta
                banderita_lista_abierta = False
                for nodo,g,f,camino_actual   in lista_abierta:
                    if nodo == vecino and f <= f_nuevo:
                        banderita_lista_abierta = True
                        break
                if banderita_lista_abierta == False:
                    lista_abierta += [(vecino, g_nuevo, f_nuevo, camino_actual + [vecino])]
    return None, considerados

camino, considerados = A_estrella(maze, punto_inicial, meta)
desplegar_laberinto(maze, camino, considerados)


            

