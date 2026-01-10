#OR EX
import numpy as np
import matplotlib.pyplot as plt
##Learning rate 
delta = 0.4
#W0=-4
X0=1
it = 0
ECM= 10
epocas=50
error=1
#W = np.array([1,0.9])
epoca = 0

#Yobt = np.array([-1,-1,1,1])
Yobt = np.zeros(4, dtype=float)

#Entrada
X = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])

Yd = np.array([0,1,1,0])
# Listas para almacenar datos para la gráfica
epoca_list = []
error_list = []
#Inicializar la semilla para crear valores random
np.random.seed(40)

##El número de neuronas de entrada será igual al número de columnas de x = 2 entradas (x1, x2)
neuronas_entrada= X.shape[1] #1 porque queremos la columna es decir las entradas
neuronas_salida= 1

####################
##Conecta la capa de entrada con la capa oculta
####################

W_entrada= -np.random.rand(neuronas_entrada,neuronas_entrada) #fila,columa
W0_entrada= -1*np.random.rand(1,neuronas_entrada)#1 porque solo pedimos una fila, con el numero de entradas

####################
##Conecta la capa oculta con la capa de salida
####################
W_salida=-np.random.rand(neuronas_salida,neuronas_entrada)#fila,columna
W0_salida=-1*np.random.rand(1,neuronas_salida)


def escalon(z):
 return np.where(z >=0,1,0)

def bipolar(z):
  return np.where(z >=0,1,-1)

def sigmoide(z):
  return 1/(1+(np.exp(-z)))


while(ECM >= 0.2 and epoca<=epocas):

    for j in range (0, X.shape[0]): #evalua casos
        z_entrada = np.dot(W_entrada.T,X[j,:]) + W0_entrada*X0 #primera capa
        ##Resultado de la capa oculta
        y_entrada = sigmoide(z_entrada) #primera capa

        z_salida =  np.dot(W_salida,y_entrada.T) + W0_salida * X0
        ##Calcula su decisión final y aplica la sigmoide
        Yobt[j] = sigmoide(z_salida)

        if abs(Yd[j] - Yobt[j]) > 0.1: 
            error_actual = Yd[j] - Yobt[j]
            
            W_salida = W_salida + delta * error_actual * y_entrada
            W0_salida = W0_salida + delta * error_actual
            
            W_entrada = W_entrada + delta * np.outer(X[j,:], error_actual)
            W0_entrada = W0_entrada + delta * error_actual.T


    ECM = (1/2)*np.sum((Yd-Yobt)**2)

    print(ECM)
    it += 1
    epoca_list.append(it)
    print("Epoca:", it)
    print("error:", ECM)
    error_list.append(ECM)
    epoca = epoca +1
    print("Epoca: ", epoca)
    print("Y obt de la epoca: ", Yobt)

print("Y obt final: ", Yobt)


# Graficar el error durante las épocas
plt.plot(epoca_list, error_list, marker='o')
plt.title('Error durante las épocas')
plt.xlabel('Época')
plt.ylabel('Error')
plt.grid(True)
plt.show()