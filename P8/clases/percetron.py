import numpy as np
import matplotlib.pyplot as plt
##Características
X= np.array([[0,0],[0,1],[1,0],[1,1]])

##Número de muestras
m = len(X)
##Clases
Yd = np.array([0,0,0,1])
Yobt = np.zeros(m)
##Definición de parámetros/hiperparametros
lr= 0.1
epoch_max = 1000
epsilon = 0.00000000000001
##Guardar el historial de todos los errores
ECM = []
##Inicialización de pesos
    ##Este es el sesgo(bias)
w0 = -np.random.rand()
    ##Vector de pesos
W = np.random.rand((2))


#########################################################333
## Funciones de activación (Escalón)
############################################################
def escalon(z):
  if z>=0:
    return 1
  return 0

## Logistica
def sigmoide(z):
  return 1/(1+np.exp(-z))
## ReLU
def ReLU(z):
  return max(0,z)
###########################################################

epoch =0
J = 10
Janterior = 0
print(f"W0 = {w0} y W={W}")

##Si J llega a cero significa que ya no es necesario que siga aprendiendo
while(epoch <= epoch_max and J>0 ):
  ##Defini un ciclo for para ir muestra a muestra
  print(f"El número de época es: {epoch}")
  Janterior= J
  ##Calculamos que tan mal le fue al anterior (ECM)
  J = (1/(2*m))*np.sum((Yd-Yobt)**2)
  ECM += [J]

  for i in range(m):
    print(f"Muestra: {i}")
    ##Calcular el valor de activación
    z = w0 + np.sum(W*X[i,:])
    ## Pasar z a la función de activación
    Yobt[i] = escalon(z)    
    print(Yobt)
    if Yobt[i] != Yd[i]:
      ##Entrenamiento
      w0 = w0 - (lr/m)*np.sum(Yobt-Yd)
      W = W - (lr/m)*np.sum(Yobt-Yd)*X[i,:]
      print(f"W0 = {w0} y W={W}")
  epoch = epoch +1

plt.plot(ECM)
print(f"W0 = {w0} y W={W}")