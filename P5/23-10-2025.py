import numpy as np 
import matplotlib.pyplot as plt


##Definicion del vector de 
X = np.array([8, 10, 12, 14, 17, 20, 21, 24])
Yr = np.array([32, 38.5, 39.9, 45, 49.7, 54, 54.8, 58])

##Definir parámetros/hiperparámetros 
lr = 0.005
T = 0.0000000000001
epsilon = 0.000001
epocas_max = 20000
epochs = 0

## Número de muestras
m = len(X)
#Ym = np.zeros(m)

## Inicializacion de coeficientes Beta
b0 = -(np.random.random()) ##Se recomienda que b0 empieze en negativos
b1 = np.random.random()

J = 10 ##va a calcular j en cada epoca
Jant = 0
ECM = [] ##Este lo guarda


##########################Fase de entrenamiento#################################
while (epochs <= epocas_max and abs(J - Jant) > epsilon):

    Ym = b0 + b1*X
    Jant = J

    J = (1/(2*m))*np.sum((Yr-Ym)**2) + T*b1**2
    ECM += [J]

    ##Descenso de gradiente
    b0 = b0 - (lr/m)*np.sum(Ym-Yr)
    b1 = b1 - (lr/m)*np.dot((Ym-Yr), X) + T*b1

    epochs = epochs + 1
    print(f"Época Número: {epochs}")
    print(f"El valor de b0 es igual a: {b0} y el de b1 {b1}")

plt.plot(ECM)

###################################################################################