import numpy as np
import matplotlib.pyplot as plt

A = np.array([
 [0,0,1,1,1,1,1,0,0,0],
 [0,1,1,0,0,0,1,1,0,0],
 [1,1,0,0,0,0,0,1,1,0],
 [1,1,0,0,0,0,0,1,1,0],
 [1,1,1,1,1,1,1,1,1,0],
 [1,1,0,0,0,0,0,0,0,0],
 [1,1,0,0,0,0,0,0,0,0],
 [0,1,1,0,0,0,0,1,1,0],
 [0,0,1,1,1,1,1,1,1,0],
 [0,0,0,1,1,1,1,0,0,0],
])

Ximg = np.array([
 [0,0,1,1,1,1,1,1,0,0],
 [1,1,1,1,1,1,1,1,1,0],
 [1,1,0,0,0,0,0,0,1,1],
 [1,1,0,0,0,0,0,0,1,1],
 [0,0,0,0,0,0,0,1,1,0],
 [0,0,0,0,0,0,1,1,0,0],
 [0,0,0,0,0,1,1,0,0,0],
 [0,0,0,0,1,1,0,0,0,0],
 [0,0,0,1,1,1,1,1,1,1],
 [0,0,1,1,1,1,1,1,1,1],
])

delta = 0.04
X0 = 1
it = 0
ECM = 10
epocas = 50
epoca = 0

def agregar_ruido_propio(patron, prob_ruido=0.1):
    ruidoso = patron.copy()

    ## sal/pimienta
    ruido = np.random.rand(*patron.shape) < prob_ruido
    ruidoso[ruido] = 1 - ruidoso[ruido]

    ## Oclusión por bloque
    if np.random.rand() < prob_ruido:  # a mayor ruido, más probable el bloque
        h, w = ruidoso.shape
        ## Aquí se calcula el tamaño del bloque y aseguramos que minimo sea de 1 
        tam = max(1, int(np.ceil(prob_ruido * 10))) 
        ##Altura del bloque (El 4 está ahí para que el bloque nunca sea mayor que 3×3 y no destruya la letra)
        bh = np.random.randint(1, min(4, tam + 1)) #-> Puede ser un alto de máximo 3 bloques
        ##Anchura del bloque
        bw = np.random.randint(1, min(4, tam + 1)) #-> Puede ser un ancho de máximo 3 bloques
        ##y0 y x0 eligen el lugar donde empieza el bloque, cuidando que no se salga de la imagen
        y0 = np.random.randint(0, h - bh + 1)
        x0 = np.random.randint(0, w - bw + 1)
        ##Vemosi le metemos 1's o 0's 
        val = 0 if np.random.rand() < 0.5 else 1
        ruidoso[y0:y0+bh, x0:x0+bw] = val

    ## Corrimiento pequeño
    if np.random.rand() < prob_ruido:
        dy = np.random.randint(-1, 2)  ## -1,0,1
        dx = np.random.randint(-1, 2)
        ruidoso = np.roll(ruidoso, shift=(dy, dx), axis=(0, 1))

    return ruidoso

np.random.seed(0)
A1, A2, A3, A4, A5 = A, agregar_ruido_propio(A, 0.1), agregar_ruido_propio(A, 0.15), agregar_ruido_propio(A, 0.2), agregar_ruido_propio(A, 0.25)
X1, X2, X3, X4, X5 = Ximg, agregar_ruido_propio(Ximg, 0.1), agregar_ruido_propio(Ximg, 0.15), agregar_ruido_propio(Ximg, 0.2), agregar_ruido_propio(Ximg, 0.2)

patrones = [A1, A2, A3, A4, A5, X1, X2, X3, X4, X5]
titulos = [
    "A original", "A ruido 5%", "A ruido 10%", "A ruido 15%", "A ruido 20%",
    "X original", "X ruido 5%", "X ruido 10%", "X ruido 15%",  "X ruido 20%"
]

fig, axs = plt.subplots(2, 5, figsize=(10, 7))
fig.suptitle("Patrones de entrenamiento (A y X) - Multicapa sin backprop", fontsize=14)
for i, ax in enumerate(axs.flat):
    ax.imshow(patrones[i], cmap='binary')
    ax.set_title(titulos[i])
    ax.axis('off')
plt.tight_layout()
plt.show()


n = A.size  # 100
X = np.zeros((len(patrones), n), dtype=float)
for i in range(len(patrones)):
    X[i, :] = patrones[i].flatten()

Yd = np.array([1,1,1,1,1,0,0,0,0,0], dtype=float)
Yobt = np.zeros(len(X), dtype=float)

epoca_list = []
error_list = []

np.random.seed(40)


neuronas_entrada = X.shape[1]
neuronas_salida = 1

W_entrada = -np.random.rand(neuronas_entrada, neuronas_entrada)
W0_entrada = -1 * np.random.rand(1, neuronas_entrada)

W_salida = -np.random.rand(neuronas_salida, neuronas_entrada)
W0_salida = -1 * np.random.rand(1, neuronas_salida)

def escalon(z):
 return np.where(z >=0,1,0)

def bipolar(z):
  return np.where(z >=0,1,-1)

def sigmoide(z):
  return 1/(1+(np.exp(-z)))


while (ECM >= 0.2 and epoca <= epocas):

    for j in range(0, X.shape[0]):
        z_entrada = np.dot(W_entrada.T, X[j, :]) + W0_entrada * X0
        y_entrada = sigmoide(z_entrada)

        z_salida = (np.dot(W_salida, y_entrada.T) + W0_salida * X0).item()
        Yobt[j] = sigmoide(z_salida)

        if abs(Yd[j] - Yobt[j]) > 0.1:
            error_actual = Yd[j] - Yobt[j]

            # Ajuste salida
            W_salida = W_salida + delta * error_actual * y_entrada
            W0_salida = W0_salida + delta * error_actual

            W_entrada = W_entrada + delta * np.outer(X[j, :], error_actual)
            W0_entrada = W0_entrada + delta * error_actual

    ECM = (1/2) * np.sum((Yd - Yobt) ** 2)

    print(ECM)
    it += 1
    epoca_list.append(it)
    error_list.append(ECM)
    epoca = epoca + 1
    print("Epoca:", epoca)
    print("Y obt de la epoca:", Yobt)

print("Y obt final:", Yobt)

plt.plot(epoca_list, error_list, marker='o')
plt.title('Error durante las épocas (Multicapa sin backprop)')
plt.xlabel('Época')
plt.ylabel('Error')
plt.grid(True)
plt.show()

##FASE DE OPOERACION 
np.random.seed(1)
A11, A22, A33, A44, A55 = A, agregar_ruido_propio(A, 0.05), agregar_ruido_propio(A, 0.1), agregar_ruido_propio(A, 0.15), agregar_ruido_propio(A, 0.2)
X11, X22, X33, X44, X55 = Ximg, agregar_ruido_propio(Ximg, 0.05), agregar_ruido_propio(Ximg, 0.1), agregar_ruido_propio(Ximg, 0.15), agregar_ruido_propio(Ximg, 0.20)

patrones_test = [A11, A22, A33, A44, A55, X11, X22, X33, X44, X55]

fig, axs = plt.subplots(2, 5, figsize=(10, 7))
fig.suptitle("Patrones de test (A y X) - Multicapa sin backprop", fontsize=14)
for i, ax in enumerate(axs.flat):
    ax.imshow(patrones_test[i], cmap='binary')
    ax.set_title(titulos[i])
    ax.axis('off')
plt.tight_layout()
plt.show()

X_test = np.zeros((len(patrones_test), n), dtype=float)
for i in range(len(patrones_test)):
    X_test[i, :] = patrones_test[i].flatten()

Yd_test = np.array([1,1,1,1,1,0,0,0,0,0], dtype=float)
Yobt_test = np.zeros(len(X_test), dtype=float)

for j in range(0, X_test.shape[0]):
    z_entrada = np.dot(W_entrada.T, X_test[j, :]) + W0_entrada * X0
    y_entrada = sigmoide(z_entrada)
    z_salida = (np.dot(W_salida, y_entrada.T) + W0_salida * X0).item()
    Yobt_test[j] = sigmoide(z_salida)

pred = (Yobt_test >= 0.5).astype(int)
acc = np.mean(pred == Yd_test.astype(int))
print("Yobt_test (sigmoide):", Yobt_test)
print("Pred (>=0.5):", pred)
print("Accuracy:", acc)
