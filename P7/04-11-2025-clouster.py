import numpy as np
import matplotlib.pyplot as plt
import random

# graficar
def plot_clusters(data, clusters, centroides):
    colors = ['red', 'blue', 'green', 'purple']
    for cluster_index in range(len(clusters)):
        x_points = [clusters[cluster_index][i][0] for i in range(len(clusters[cluster_index]))]
        y_points = [clusters[cluster_index][i][1] for i in range(len(clusters[cluster_index]))]
        plt.scatter(x_points, y_points, color=colors[cluster_index], label=f'Cluster {cluster_index + 1}')

    x_centroids = [centroides[i][0] for i in range(len(centroides))]
    y_centroids = [centroides[i][1] for i in range(len(centroides))]
    plt.scatter(x_centroids, y_centroids, color='black', marker='x', s=100, label='Centroids')

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('K-means Clustering Result')
    plt.legend()
    plt.grid(True)
    plt.show()

# Datos iniciales
data_points = [
    [1.0, 2.0], [1.5, 1.8], [1.2, 2.1], [1.3, 2.3], [1.1, 2.2], [1.4, 2.4],
    [0.8, 1.9], [1.0, 2.2], [1.6, 1.5], [1.4, 1.7], [1.2, 1.9], [1.3, 1.8],
    [1.7, 1.6], [0.9, 2.1], [1.0, 2.3], [1.6, 2.0], [1.4, 2.5], [1.3, 2.0],
    [1.5, 2.1], [1.2, 2.0],
    [5.0, 8.0], [5.5, 8.5], [5.8, 8.1], [5.0, 8.2], [5.4, 8.3], [5.3, 8.0],
    [5.1, 8.4], [5.1, 8.1], [5.2, 8.2], [5.0, 7.9], [5.4, 8.6], [5.7, 8.3],
    [5.8, 8.4], [5.9, 8.2], [6.0, 8.5], [5.8, 8.0], [5.7, 8.1], [5.5, 7.9]
]

### Función para calcular la ditancia euclidiana de los puntos 
def distacia_euclidiana(centroide, dato):
    centroide = np.array(centroide)
    dato = np.array(dato)

    d = (centroide - dato)**2
    print(d)
    distacia = [None]*len(d)
    for i in range(len(d)):
        distacia[i] = np.sqrt(d[i][0] + d[i][1])
    return distacia


def kmeans (data, k, epocas = 2000):
    ##Si el dataset está en formato de lista
    data = np.array(data)
    num_datos, num_caracteristicas = data.shape

    ##Inicialización aleatoria de los centriodes
    indices_aleatorios = [None]*k
    centroides = []

    for i in range(k):
        while True:
            random_centroides_indice = random.randint(0,num_datos)
            duplicado = False
            for j in range(i):
                if indices_aleatorios[j] == random_centroides_indice:
                    duplicado = True
                    break
            if duplicado == True:
                indices_aleatorios[i] = random_centroides_indice
                centroides =centroides + [data[random_centroides_indice]]
                break

    ##Asignar una clase a cada punto
    for epoch in range(epocas):
        asignacion_clouster = [0]*num_datos
        ##Recorrer cada rato para despúes compararlo con CADA centroide 
        for q in range(num_datos):
            ## Un vector que tenga la longitud del dato actula a CADA centride
            dist = [0]*k
            for l in range(k):
                dist[l] = distacia_euclidiana(centroides[l],data[q])
            ##Encontrar el indece de la distancia más pequeña que corresponda a un clouster K
            min_distancia_index = np.argmin(dist)
            asignacion_clouster[q] = min_distancia_index