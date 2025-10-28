import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

# Lectura del CSV
df = pd.read_csv("sales_by_year_1995_2025.csv")

# Mostrar las primeras filas
print(df.head())
print(df.info())

# Gráfica original
plt.figure(figsize=(11,5))
plt.plot(df['year'], df['sales_millions_usd'], marker='o')
plt.title("Ventas de coches en Millones de dólares de FolksBaguen")
plt.xlabel("Año")
plt.ylabel("Ventas")
plt.grid(True)
plt.show()

# Vectores
X = np.array([8, 10, 12, 14, 17, 20, 21, 24])
Yr = np.array([32, 38.5, 39.9, 45, 49.7, 54, 54.8, 58])

##Normalizacion
def normalizacion (x):
    X_max = np.max(x)
    X_min = np.min(x)
    return (x-X_min)/(X_max-X_min)

X_norm = normalizacion(X)
Yr_norm = normalizacion(Yr)

print(X_norm)
print(Yr_norm)



# Hiperparámetros
lr = 0.01
T = 0.0000000000001
epsilon = 0.00000000000001
epocas_max = 200000000

def regresion_ride(X, Yr, lr, T, epsilon, epocas_max):
    epochs = 0
    m = len(X)
    b0 = -np.random.rand()
    b1 = np.random.rand()

    J = 0
    Jant = 10
    ECM = []

    while (epochs <= epocas_max and abs(J - Jant) > epsilon):
        Ym = b0 + b1 * X
        Jant = J
        J = (1/(2*m)) * np.sum((Yr - Ym)**2) + T * b1**2
        ECM.append(J)

        b0 = b0 - (lr/m) * np.sum(Ym - Yr)
        b1 = b1 - (lr/m) * np.dot((Ym - Yr), X) + T * b1
        epochs += 1

        print(f"Época Número: {epochs}")
        print(f"b0 = {b0:.4f}, b1 = {b1:.4f}, J = {J:.6f}")

    plt.plot(ECM)
    plt.title("Evolución del ECM")
    plt.xlabel("Épocas")
    plt.ylabel("Error cuadrático medio")
    plt.figure()
    plt.show()

# Ejecutar función
regresion_ride(X_norm, Yr_norm, lr, T, epsilon, epocas_max)
