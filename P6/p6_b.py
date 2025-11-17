import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Lectura del CSV
df = pd.read_csv("datos_finanzas.csv")

# Gráfica original

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(df["ingresos_mensuales"], df["gastos_operativos"], df["utilidad_neta"], marker='o')
ax.set_xlabel("Ingresos")
ax.set_ylabel("Gastos operativos")
ax.set_zlabel("Utilidad neta")
ax.set_title("Datos financieros originales")
plt.show()

X1 = df["ingresos_mensuales"].values
X2 = df["gastos_operativos"].values
X3 = df["publicidad_marketing"].values
Yr  = df["utilidad_neta"].values

##  Normalizacion
def normalizacion(x):
    X_max = np.max(x)
    X_min = np.min(x)
    return (x-X_min)/(X_max-X_min), X_max, X_min

X1_norm, X1_max, X1_min = normalizacion(X1)
X2_norm, X2_max, X2_min = normalizacion(X2)
X3_norm, X3_max, X3_min = normalizacion(X3)
Yr_norm, Y_max, Y_min   = normalizacion(Yr)

X = np.column_stack([X1_norm, X2_norm, X3_norm])

## Hiperparámetros
lr = 0.01
T = 1e-13
epsilon = 1e-6
epocas_max = 2000000

def regresion_ride(X, Yr, lr, T, epsilon, epocas_max):
    ##Los pasos empiezan en 0
    epochs = 0
    ## La cantidad de datos que tenemos 
    m = len(X)
    ## Aquí solo inicializamos un un punto random para empezar
    b0 = -np.random.rand()
    b1 = np.random.rand()
    b2 = np.random.rand()
    b3 = np.random.rand()

    ## Función de costo, o sea el error actual 
    J = 0
    ## error anterior 
    ## Inicia en 10 para que sea mucho más grande que el epsilon y se ponga a trabajar
    Jant = 10
    ## Se irán guardando los errores actuales
    ECM = []

    while (epochs <= epocas_max and abs(J - Jant) > epsilon):
        ## Prediccion actual
        Ym = b0 + b1*X[:,0] + b2*X[:,1] + b3*X[:,2]

        Jant = J

        ##--Penalizacion por distacia-- --Penalización por compljidad--
        J = (1/(2*m))*np.sum((Yr - Ym)**2) + T*(b1**2 + b2**2 + b3**2)

        ECM += [J]

        ## Ajusta la altura
        b0 = b0 - (lr/m) * np.sum(Ym - Yr)

        b1 = b1 - ((lr/m)*np.sum((Ym - Yr)*X[:,0]) + T*b1)
        b2 = b2 - ((lr/m)*np.sum((Ym - Yr)*X[:,1]) + T*b2)
        b3 = b3 - ((lr/m)*np.sum((Ym - Yr)*X[:,2]) + T*b3)

        epochs += 1

        print(f"Época Número: {epochs}")
        print(f"b0 = {b0:.4f}, b1 = {b1:.4f}, b2 = {b2:.4f}, b3 = {b3:.4f}, J = {J:.6f}")

    ## Se grafica el ECM
    plt.plot(ECM)
    plt.title("Evolución del ECM")
    plt.xlabel("Épocas")
    plt.ylabel("Error cuadrático medio")
    plt.show()

    return b0, b1, b2, b3

# Ejecutar función
b0_n, b1_n, b2_n, b3_n = regresion_ride(X, Yr_norm, lr, T, epsilon, epocas_max)

##Graficar Ym_normalizado

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X1_norm, X2_norm, Yr_norm, color='green')

Ym_norm_pred = b0_n + b1_n*X1_norm + b2_n*X2_norm + b3_n*X3_norm
ax.plot_trisurf(X1_norm, X2_norm, Ym_norm_pred, alpha=0.5)

ax.set_xlabel("X1 norm")
ax.set_ylabel("X2 norm")
ax.set_zlabel("Y norm")
ax.set_title("Modelo normalizado en 3D")
plt.show()


## Desnormalización de coeficientes
B1 = b1_n*(Y_max-Y_min)/(X1_max-X1_min)
B2 = b2_n*(Y_max-Y_min)/(X2_max-X2_min)
B3 = b3_n*(Y_max-Y_min)/(X3_max-X3_min)
B0 = (Y_max-Y_min)*b0_n - B1*X1_min - B2*X2_min - B3*X3_min + Y_min


## Gráfica desnormalizada
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X1, X2, Yr, color='blue')

Ym_real_pred = B0 + B1*X1 + B2*X2 + B3*X3
ax.plot_trisurf(X1, X2, Ym_real_pred, alpha=0.5)

ax.set_xlabel("Ingresos")
ax.set_ylabel("Gastos")
ax.set_zlabel("Utilidad")
ax.set_title("Modelo desnormalizado en 3D")
plt.show()


### Fase de operación 

ing = float(input("Ingresos mensuales: "))
gas = float(input("Gastos operativos: "))
pub = float(input("Publicidad y marketing: "))

Ym_pred = B0 + B1*ing + B2*gas + B3*pub

print("\n---* Resultado de la Prediccion *---")
print(f"Utilidad neta estimada: {Ym_pred:.2f}")
