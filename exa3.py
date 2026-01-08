import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_openml
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
from sklearn.metrics import accuracy_score

pima = fetch_openml(name='diabetes', version=1, as_frame=True)

df = pima.frame 

# El dataset trae la columna "class" como target
df["class"] = df["class"].map({"tested_negative": 0, "tested_positive": 1})

data_points = df[["plas", "mass", "age"]].values   ##Glucosa, BMI, Edad

cluster_reales = df["class"].values

# Normalizacion 
data_points = (data_points - data_points.mean(axis=0)) / data_points.std(axis=0)

# Función para graficar los resultados
def plot_clusters(data, clusters, centroids):
    colors = ['red', 'blue', 'green', 'purple']
    for cluster_index in range(len(clusters)):
        x_points = [clusters[cluster_index][i][0] for i in range(len(clusters[cluster_index]))]
        y_points = [clusters[cluster_index][i][1] for i in range(len(clusters[cluster_index]))]
        plt.scatter(x_points, y_points, color=colors[cluster_index], label=f'Cluster {cluster_index + 1}')
    x_centroids = [centroids[i][0] for i in range(len(centroids))]
    y_centroids = [centroids[i][1] for i in range(len(centroids))]
    plt.scatter(x_centroids, y_centroids, color='black', marker='x', s=100, label='Centroids')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('K-means Clustering Result')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_clusters_3d(data, clusters, centroids):
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['red', 'blue', 'green']
    
    for cluster_index in range(len(clusters)):
        x = [p[0] for p in clusters[cluster_index]]
        y = [p[1] for p in clusters[cluster_index]]
        z = [p[2] for p in clusters[cluster_index]]
        ax.scatter(x, y, z, color=colors[cluster_index], label=f'Cluster {cluster_index+1}')

    cx = [c[0] for c in centroids]
    cy = [c[1] for c in centroids]
    cz = [c[2] for c in centroids]
    
    ax.scatter(cx, cy, cz, color='black', marker='x', s=80, label='Centroides')

    ax.set_xlabel("Glucosa (norm)")
    ax.set_ylabel("BMI (norm)")
    ax.set_zlabel("Edad (norm)")
    plt.title("Clusters con K-means (3 características)")
    plt.legend()
    plt.show()


# Función para calcular la distancia euclidiana
# def distancia_euclidiana(X, x_test):
#     x_test = np.array(x_test)
#     d = (X - x_test) ** 2
#     distancia = [None] * len(d)
#     for i in range(len(d)):
#         distancia[i] = np.sqrt(d[i][0] + d[i][1])
#     return distancia
def distancia_euclidiana(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.sqrt(np.sum((a - b)**2))


# Función de K-means ajustada
def kmeans(data, k, epocas=100):
    data = np.array(data)
    num_datos, num_caracteristicas = data.shape

    # Inicialización de centroides aleatorios
    indices_usados = [-1]*k
    centroides = []
    numero_datos = len(data)
    ## Asignar aleatoriamente los centroides
    for i in range(k):
      while True:
        random_centroides = np.random.randint(0,numero_datos)
        duplicado = False
        for j in range(i):
          if indices_usados[j] == random_centroides:
            duplicado = True
            break
        if duplicado == False:
          indices_usados[i] = random_centroides
          centroides = centroides + [data[random_centroides]]
          break

    for iteration in range(epocas):
        # Paso 1: Asignar cada punto al centroide más cercano
        cluster_assignments = [0] * num_datos
        for i in range(num_datos):
            distances = [0] * k
            for j in range(k):
                #distances[j] = distancia_euclidiana([centroides[j]], data[i])[0]
                distances[j] = distancia_euclidiana(centroides[j], data[i])

            # Encontrar el índice del valor mínimo usando np.argmin
            min_distance_index = np.argmin(distances)
            cluster_assignments[i] = min_distance_index

        # Paso 2: Recalcular los centroides
        new_centroids = []
        for cluster_index in range(k):
            points_in_cluster = []
            for i in range(num_datos):
                if cluster_assignments[i] == cluster_index:
                    points_in_cluster += [data[i]]
            if len(points_in_cluster) > 0:
                cluster_sum = [0] * num_caracteristicas
                for point in points_in_cluster:
                    for j in range(num_caracteristicas):
                        cluster_sum[j] += point[j]
                cluster_mean = [cluster_sum[j] / len(points_in_cluster) for j in range(num_caracteristicas)]
                new_centroids += [cluster_mean]
            else:
                new_centroids += [centroids[cluster_index]]

        # Verificar convergencia manualmente
        converged = True
        for i in range(k):
            for j in range(num_caracteristicas):
                if abs(new_centroids[i][j] - centroides[i][j]) >= 1e-6:
                    converged = False
                    break
            if not converged:
                break

        if converged:
            break

        centroids = new_centroids

    # Agrupar los puntos según los clusters
    clusters = [[] for _ in range(k)]
    for i in range(num_datos):
        clusters[cluster_assignments[i]] = clusters[cluster_assignments[i]] + [data[i].tolist()]

    return centroids, clusters, cluster_assignments

k = 2
centroids, clusters, cluster_predicho = kmeans(data_points, k)
plot_clusters_3d(data_points, clusters, centroids)

# Convertir clusters predichos en arreglo NumPy
cluster_predicho = np.array(cluster_predicho)

## Buscamos un dicc. para clasificas de esta forma {0: 1, 1: 0, 2: 2}, el cluster 0 contiene principalmente la clase 1
cluster_para_clases = {}

## Iteramos sobre los 3  k
for cluster_id in range(k):
    ## Estamos en el cluster actual
    clases_en_cluster = cluster_reales[cluster_predicho == cluster_id]
    
    if len(clases_en_cluster) == 0:
        continue
    
    # Elegir la clase real más común en ese cluster
    clase_mas_frecuente = Counter(clases_en_cluster).most_common(1)[0][0]
    cluster_para_clases[cluster_id] = clase_mas_frecuente

# Convertir predicciones de clusters a clases reales
predicciones_finales = np.array([cluster_para_clases[c] for c in cluster_predicho])

# Accuracy
aproximado = accuracy_score(cluster_reales, predicciones_finales)

print("clase real:", cluster_para_clases)
print("aproxmado del K-means:", aproximado)