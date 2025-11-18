import matplotlib.pyplot as plt
import numpy as np

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

# Datos iniciales
data_points = [
    [1.0, 2.0], [1.5, 1.8], [1.2, 2.1], [1.3, 2.3], [1.1, 2.2], [1.4, 2.4],
    [0.8, 1.9], [1.0, 2.2], [1.6, 1.5], [1.4, 1.7], [1.2, 1.9], [1.3, 1.8],
    [1.7, 1.6], [0.9, 2.1], [1.0, 2.3], [1.6, 2.0], [1.4, 2.5], [1.3, 2.0],
    [1.5, 2.1], [1.2, 2.0],
    [5.0, 8.0], [5.5, 8.5], [5.2, 8.1], [5.0, 8.2], [5.4, 8.3], [5.3, 8.0],
    [5.1, 8.4], [5.6, 8.1], [5.2, 8.2], [5.0, 7.9], [5.4, 8.6], [5.7, 8.3],
    [5.8, 8.4], [5.9, 8.2], [6.0, 8.5], [5.8, 8.0], [5.7, 8.1], [5.5, 7.9]
]

# Función para calcular la distancia euclidiana
def distancia_euclidiana(X, x_test):
    x_test = np.array(x_test)
    d = (X - x_test) ** 2
    distancia = [None] * len(d)
    for i in range(len(d)):
        distancia[i] = np.sqrt(d[i][0] + d[i][1])
    return distancia

# Función de K-means ajustada
def kmeans(data, k, epocas=100):
    data = np.array(data)
    num_datos, num_caracteristicas = data.shape

    # Inicialización de centroides aleatorios

    ## Lista de longuitud igual al número de clousters
    indices_usados = [-1]*k
    ## Lista donde se almacenarán los controides random
    centroides = []
    ## Límite de la muestra
    numero_datos = len(data)

    ## Asignar aleatoriamente los centroides (llenar la lista centroides[])
    for i in range(k):
      ## Aquí se selecciona el candidato aleatorio
      while True:
        ## se selecciona un indice que será un punto de data aleatoreamente
        random_centroides = np.random.randint(0,numero_datos-1)
        duplicado = False
        for j in range(i):
          if indices_usados[j] == random_centroides:
            duplicado = True
            break
        if duplicado == False:
          indices_usados[i] = random_centroides
          centroides = centroides + [data[random_centroides]]
          break
    print(centroides[0])


    for iteration in range(epocas):
        # Paso 1: Asignar cada punto al centroide más cercano
                ## Se cra una lista con la cantiadad num_datos de ceros
        cluster_assignments = [0] * num_datos

        ## Este for llena la lista cluster_assignments en la cual a cada punto de data le asigana a que cluster pertenece 
        for i in range(num_datos):
            ##Estamos en el punto actula

            ##Lista donde se guarda la distancia del punto actula a los k centroides
            distances = [0] * k

            ## Calcula la distancia y la guarda e distances[]
            for j in range(k):
                distances[j] = distancia_euclidiana([centroides[j]], data[i])[0]

            # Encontrar el índice del valor mínimo dentro de distances[] usando np.argmin
            min_distance_index = np.argmin(distances)

            ## mete ese valor mínimo en la posicion i
            cluster_assignments[i] = min_distance_index

        # Paso 2: Recalcular los centroides
        new_centroids = []
        for cluster_index in range(k):
            ##Estamos en el clouster actual

            ## Lista para guardar los puntos que pertenecen al clouster actual
            points_in_cluster = []


            for i in range(num_datos):
                ##Estamos en el dato actual


                ## si el dato actual tiene asignado el clouster actual 
                if cluster_assignments[i] == cluster_index:
                    #Se agrega a la lista de los pertenecientes
                    points_in_cluster += [data[i]]
            
            
            if len(points_in_cluster) > 0:

                ## Lista de longit igual a número de características de ceros(2, [0,0])
                cluster_sum = [0] * num_caracteristicas

                ## Vamos a iter en cada uno de los puntos pertenecietes del couster actual
                for point in points_in_cluster:
                    ## Point actual

                    for j in range(num_caracteristicas):

                        ##Suma todas las cordenadas X y todas las coordenadas Y
                        cluster_sum[j] = cluster_sum[j] + point[j]
                        
                ## Se calcula el promedio y esa es la nueca posicion del centroide
                cluster_mean = [cluster_sum[j] / len(points_in_cluster) for j in range(num_caracteristicas)]
                ## Se agrega el nuevo centroide a la lista 
                new_centroids += [cluster_mean]
            else:
                ## Sie es cluster quesó vacio se mandtiene en la pos anterior
                new_centroids += [centroids[cluster_index]]

        # Verificar convergencia manualmente
            ##suposición, sí convenge
        converged = True
        for i in range(k):
            for j in range(num_caracteristicas):
                ##Compara el nuevo centroide con el anterior
                if abs(new_centroids[i][j] - centroides[i][j]) >= 1e-6:
                    ##Sí se movió
                    converged = False
                    break
            if not converged:
                ##Si uno no converge entonces a no reviso todos los demás
                break
        ## Caso A: la var sigue siendo true
        if converged:
            ## Ninguno se movió, terminó
            break
        
        ## Caso B: Si la var es False porque al menos uno se movió
        ##se actualiza el centroude con la nueva pos y comienza una nueva ronda
        centroids = new_centroids



    # Agrupar los puntos según los clusters
        ## Crea k listas, un por cada cloutser
    clusters = [[] for _ in range(k)]

    for i in range(num_datos):
        ##Selcciona la lista de ese clouster                 ##Obtiene las coordenadas del punto i y las pasa a lista      
        clusters[cluster_assignments[i]] = clusters[cluster_assignments[i]] + [data[i].tolist()]

    return centroids, clusters



# Ejecutar K-means y graficar los resultados
k = 2
centroids, clusters = kmeans(data_points, k)
plot_clusters(data_points, clusters, centroids)