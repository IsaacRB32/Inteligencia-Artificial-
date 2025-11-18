import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
from sklearn.metrics import accuracy_score

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)

df["target"] = iris.target

data_points = df[["sepal length (cm)","petal length (cm)","petal width (cm)"]].values 

# Normalización 
data_points = (data_points - data_points.mean(axis=0)) / data_points.std(axis=0)

labels_reales = df["target"].values

def plot_clusters_3d(data, clusters, centroids):
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')

    colors = ['red', 'blue', 'green', 'purple']

    for cluster_index in range(len(clusters)):
        x = [p[0] for p in clusters[cluster_index]]
        y = [p[1] for p in clusters[cluster_index]]
        z = [p[2] for p in clusters[cluster_index]]
        ax.scatter(x, y, z, color=colors[cluster_index], label=f'Cluster {cluster_index + 1}')

    # Centroides
    cx = [c[0] for c in centroids]
    cy = [c[1] for c in centroids]
    cz = [c[2] for c in centroids]
    ax.scatter(cx, cy, cz, color="black", marker="x", s=120, label="Centroids")

    ax.set_xlabel("sepal length (std)")
    ax.set_ylabel("petal length (std)")
    ax.set_zlabel("petal width (std)")
    ax.set_title("K-means Clustering (3D)")
    ax.legend()
    plt.show()


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

k = 3
centroids, clusters, cluster_predicho = kmeans(data_points, k)

plot_clusters(data_points, clusters, centroids)
plot_clusters_3d(data_points, clusters, centroids)

# Convertir clusters predichos en arreglo NumPy
pred = np.array(cluster_predicho)

# Crear un mapeo cluster → clase real
cluster_to_class = {}

for cluster_id in range(k):
    clases_en_cluster = labels_reales[pred == cluster_id]
    
    if len(clases_en_cluster) == 0:
        continue
    
    # Elegir la clase real más común en ese cluster
    clase_mas_frecuente = Counter(clases_en_cluster).most_common(1)[0][0]
    cluster_to_class[cluster_id] = clase_mas_frecuente

# Convertir predicciones de clusters a clases reales
predicciones_finales = np.array([cluster_to_class[c] for c in pred])

# Accuracy
accuracy = accuracy_score(labels_reales, predicciones_finales)

print("Mapeo cluster → clase real:", cluster_to_class)
print("Accuracy del K-means:", accuracy)