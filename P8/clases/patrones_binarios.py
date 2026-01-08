import numpy as np
import matplotlib.pyplot as plt

A = np.array([
 [0,0,1,1,1,1,1,0,0,0],
 [0,1,1,0,0,0,1,1,0,0],
 [1,1,0,0,0,0,0,1,1,0],
 [1,1,0,0,0,0,0,1,1,0],
 [1,1,1,1,1,1,1,1,1,0],
 [1,1,0,0,0,0,0,1,1,0],
 [1,1,0,0,0,0,0,1,1,0],
 [1,1,0,0,0,0,0,1,1,0],
 [1,1,0,0,0,0,0,1,1,0],
 [0,0,0,0,0,0,0,0,0,0],
])

X = np.array([
 [1,0,0,0,0,0,0,0,0,1],
 [0,1,0,0,0,0,0,0,1,0],
 [0,0,1,0,0,0,0,1,0,0],
 [0,0,0,1,0,0,1,0,0,0],
 [0,0,0,0,1,1,0,0,0,0],
 [0,0,0,0,1,1,0,0,0,0],
 [0,0,0,1,0,0,1,0,0,0],
 [0,0,1,0,0,0,0,1,0,0],
 [0,1,0,0,0,0,0,0,1,0],
 [1,0,0,0,0,0,0,0,0,1],
])

##Función que agrega ruido a las muestras (sal y pimienta)
def agregar_ruido(patron, prob_ruido=0.1):
    ##Se crea una copia para no modificar el original
    ruidoso = patron.copy()
    ##Crea una matrix 10x10 aleatoria 
    ruido = np.random.rand(*patron.shape) < prob_ruido
    ##Se hace flip
    ruidoso[ruido] = 1 - ruidoso[ruido]
    return ruidoso

np.random.seed(0)
##Generación de variantes ruidosas (con etiquetas)
A1, A2, A3, A4, A5 = A, agregar_ruido(A, 0.05), agregar_ruido(A, 0.1), agregar_ruido(A, 0.15), agregar_ruido(A, 0.2)
X1, X2, X3, X4, X5 = X, agregar_ruido(X, 0.05), agregar_ruido(X, 0.1), agregar_ruido(X, 0.15), agregar_ruido(X, 0.20)

##Crea 10 ejes (2 filas × 5 columnas)
fig, axs = plt.subplots(2, 5, figsize=(10, 7))
fig.suptitle("Patrones de entrenamiento (A y X con ruido)", fontsize=14)

patrones = [A1, A2, A3, A4, A5, X1, X2, X3, X4, X5]
titulos = [
    "A original", "A ruido 5%", "A ruido 10%", "A ruido 15%", "A ruido 20%",
    "X original", "X ruido 5%", "X ruido 10%", "X ruido 15%",  "X ruido 20%"
]
for i, ax in enumerate(axs.flat):
    ax.imshow(patrones[i], cmap='binary')
    ax.set_title(titulos[i])
    ax.axis('off')

plt.tight_layout()
plt.show()

#####################################Matriz Características###############################################
##Cantidad de de pixeles que tiene la imagen
n = A.size 
##Crea una caja vacía 
matriz = np.zeros((len(patrones),n))

# # El bucle recorre las 10 imágenes una por una:
# # Toma la imagen i (por ejemplo, la "A original").
# # La estira a 100 píxeles.
# # La guarda en la fila i de la nueva matriz.
for i in range(len(patrones)):
                ##Desenrrolla la matriz
  matriz[i]=np.array(patrones[i].flatten())
## Clases/Etiquetas 
# # Índice, Imagen,  Yd(Valor deseado),Significado
# # 0 al 4,A1 hasta A5,     1        ,"""Si ves esto, es una A"""
# # 5 al 9,X1 hasta X5,     0        ,"""Si ves esto, es una X"""
Yd =np.array([1,1,1,1,1,0,0,0,0,0])

############################################################################3
##Número de muestras
m = len(matriz)
##Clases
Yobt = np.zeros(m)
##Definición de parámetros/hiperparametros
lr= 0.1
epoch_max = 1000
epsilon = 0.00000000000001
##Guardar el historial de todos los errores
ECM = []
##Inicialización de pesos
np.random.seed(42)  # semilla para pesos (separada del ruido)
w0 = -np.random.rand()
W = np.random.rand((n))

#########################################################333
## Funciones de activación
##Escalón

def escalon(z):
  if z>0:
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
##X = matriz
X_train = matriz

while(epoch <= epoch_max and J>0 ):
  ##Defini un ciclo for para ir muestra a muestra
  print(f"El número de época es: {epoch}")
  Janterior= J
  J = (1/(2*m)) * np.sum((Yd - Yobt)**2)
  ECM += [J]
  for i in range(m):
    print(f"Muestra: {i}")
    ##Calcular el valor de activación
    z = w0 + np.sum(W*X_train[i,:])
    ## Pasar z a la función de activación
    Yobt[i] = escalon(z)
    print(Yobt)
    if Yobt[i] != Yd[i]:
      ##Entrenamiento
      w0 = w0 - (lr/m)*np.sum(Yobt-Yd)
      W = W - (lr/m)*np.sum(Yobt-Yd)*X_train[i,:]
      print(f"W0 = {w0} y W={W}")
  epoch = epoch +1
plt.plot(ECM)

#######################3
##FASE DE OPERACION
##########################
np.random.seed(1)
A11, A22, A33, A44, A55= A, agregar_ruido(A, 0.05), agregar_ruido(A, 0.1), agregar_ruido(A, 0.15), agregar_ruido(A, 0.2)
X11, X22, X33, X44, X55 =  X, agregar_ruido(X, 0.05), agregar_ruido(X, 0.1), agregar_ruido(X, 0.15), agregar_ruido(X, 0.20)

fig, axs = plt.subplots(2, 5, figsize=(10, 7))
fig.suptitle("Patrones de test (A y X con ruido)", fontsize=14)

patrones_test = [A11, A22, A33, A44, A55, X11, X22, X33, X44, X55]
titulos = [
    "A original", "A ruido 5%", "A ruido 10%", "A ruido 15%", "A ruido 20%",
    "X original", "X ruido 5%", "X ruido 10%", "X ruido 15%",  "X ruido 20%"
]

for i, ax in enumerate(axs.flat):
    ax.imshow(patrones_test[i], cmap='binary')
    ax.set_title(titulos[i])
    ax.axis('off')

plt.tight_layout()
plt.show()

matriz_test = np.zeros((len(patrones_test),n))
m = len(matriz_test)
for i in range(len(patrones_test)):
  matriz_test[i]=np.array(patrones_test[i].flatten())
## Clases/Etiquetas

Yobt_test = np.zeros(m)
for i in range(m):
  print(f"Test: {i}")
  ##Calcular el valor de activación
  z = w0 + np.sum(W*matriz_test[i,:])
  ## Pasar z a la función de activación
  Yobt_test[i] = escalon(z)
print(Yobt_test)

Yd_test = np.array([1,1,1,1,1,0,0,0,0,0])
acc = np.mean(Yobt_test.astype(int) == Yd_test)
print("Accuracy:", acc)
