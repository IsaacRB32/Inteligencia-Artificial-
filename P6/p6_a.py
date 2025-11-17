import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

# Lectura del CSV
df = pd.read_csv("sales_by_year_1995_2025.csv")

# Mostrar las primeras filas
print(df.head())
print(df.info())

# Gráfica original

##Tamanio de la gráfica
plt.figure(figsize=(11,5))

## plt.plot(x,y,marker='o')
plt.plot(df['year'], df['sales_millions_usd'], marker='o')
## Solo es un titulo
plt.title("Ventas de coches en Millones de dólares de FolksBaguen")
## Le pone el nombre al eje x
plt.xlabel("Año")
## Le pone el nombre al eje y
plt.ylabel("Ventas")
## Activa la cuadricula 
plt.grid(True)
## Mostramos la gráfica en la pantalla 
plt.show()


## df["year"] es solo una serie de pandas
## .values la convierte en un arreglo NumPy
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

    ## Función de costo, o sea el error actual 
    J = 0
    ## error anterior 
    ## Inicia en 10 para que sea mucho más grande que el epsilon y se ponga a trabajar
    Jant = 10
    ## Se irán guardando los errores actuales
    ECM = []

    ## Dos condiciones: 1-. Aun no llega al max de epocas
    ## 2-. Si la mejora es menor que el lim de epsilon
    while (epochs <= epocas_max and abs(J - Jant) > epsilon):
        ## Prediccion actual
        ## genera un arreglo con suposiciones
        Ym = b0 + b1 * X

        Jant = J
        ## Justo es la función de costo
        ## Qué tan mala es la linea actual Ym al compararla con los datos reales 
        ##--Penalizacion por distacia-- --Penalización por compljidad--
        J = (1/(2*m)) * np.sum((Yr - Ym)**2) + T * b1**2
        ## --Costo por error--          --Costo por complejidad--
        #ECM.append(J)
        ##Voy guardando los costos
        ECM += [J]

        #b0 = b0 - (lr/m) * np.sum(Ym - Yr)

        ## Calcular la fuerza y direccion de bajada de b0 (Gradiente)    
        Gradiente = (lr/m) * np.sum(Ym - Yr)

        ## Ajusta la altura
        b0 = b0 - Gradiente
        ## Ajusta la inclinación 
        Penalizacion_Ride = T * b1
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

### Fase de operación (Modificada para incluir Meses)

anio = input("Ingrese el año que desea predecir (ej. 2026): ")
mes = input("Ingrese el número del mes (1=Enero, 12=Diciembre): ")

X_anio = int(anio)
X_mes = int(mes)

if X_mes < 1 or X_mes > 12:
    raise ValueError("El numero de mes debe estar entre 1 y 12.")
    
## Contamos que el mes esta al incio por eso el -1
X_fraction = (X_mes - 1) / 12

## El anio más los decimales de mes
X_test = X_anio + X_fraction

## Se hace la predicción
Ym_test = B0 + B1 * X_test

print("\n---* Resultado de la Prediccion *---")
print(f"Prediccion para: {X_anio}, Mes {X_mes}")
print(f"El modelo usa el valor de tiempo: {X_test:.4f}")
print(f"La predicción de ventas para esa fecha es de: {Ym_test:.4f} millones de USD.")