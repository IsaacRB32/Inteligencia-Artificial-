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
# X = np.array([8, 10, 12, 14, 17, 20, 21, 24])
# Yr = np.array([32, 38.5, 39.9, 45, 49.7, 54, 54.8, 58])

X = df["year"].values
Yr = df["sales_millions_usd"].values

##Normalizacion
def normalizacion (x):
    X_max = np.max(x)
    X_min = np.min(x)
    return (x-X_min)/(X_max-X_min), X_max, X_min

X_norm, X_max, X_min = normalizacion(X)
Yr_norm, Y_max, Y_min = normalizacion(Yr)

print(X_norm)
print(Yr_norm)



# Hiperparámetros
# lr = 0.01
# T = 0.0000000000001
# epsilon = 0.00000000000001
# epocas_max = 2000000

lr = 0.01
T = 1e-13
epsilon = 1e-6
epocas_max = 2000000

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
    plt.show()
    return b0, b1

# Ejecutar función
b0_norm, b1_norm, = regresion_ride(X_norm, Yr_norm, lr, T, epsilon, epocas_max)
##Graficar Ym_normalizado

plt.figure()
plt.scatter(X_norm,Yr_norm, label = 'Datos normalizados', color = 'green')
plt.plot()
plt.plot(X_norm,b0_norm+b1_norm*X_norm, label = 'Modelo Ym_normalizado', color = 'red')
plt.title("Regresion lineal -Normalizada")
plt.xlabel("X_norm")
plt.ylabel("Y_norm")
plt.grid(True)
plt.show()

B1 =  b1_norm*(Y_max-Y_min)/(X_max-X_min)
B0 = (Y_max-Y_min)*b0_norm-B1*X_min+Y_min

print("el valor de B1 es: ", B1)
print("el valor de B0 es: ", B0)


plt.figure()
plt.scatter(X,Yr, label = 'Datos desnormalizados', color = 'green')
plt.plot()
plt.plot(X,B0+B1*X, label = 'Modelo Ym_desnormalizado', color = 'red')
plt.title("Regresion lineal -Normalizada")
plt.xlabel("X_m")
plt.ylabel("Y_m")
plt.grid(True)
plt.show()

### Fase de operación

X_test = float(input("¿Cuál año te gustaría predecir?: "))
Ym_test = B0 + B1*X_test
print(f"La prediccion para el año {X_test} es: {Ym_test}")
