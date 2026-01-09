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

X = np.array([
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
        dy = np.random.randint(-1, 2)  # -1,0,1
        dx = np.random.randint(-1, 2)
        ruidoso = np.roll(ruidoso, shift=(dy, dx), axis=(0, 1))

    return ruidoso


np.random.seed(0)
A1, A2, A3, A4, A5 = A, agregar_ruido_propio(A, 0.05), agregar_ruido_propio(A, 0.1), agregar_ruido_propio(A, 0.15), agregar_ruido_propio(A, 0.2)
X1, X2, X3, X4, X5 = X, agregar_ruido_propio(X, 0.05), agregar_ruido_propio(X, 0.1), agregar_ruido_propio(X, 0.15), agregar_ruido_propio(X, 0.20)

fig, axs = plt.subplots(2, 5, figsize=(10, 7))
fig.suptitle("Patrones de entrenamiento (A y X con ruido propio)", fontsize=14)

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

n = A.size
matriz = np.zeros((len(patrones), n))
for i in range(len(patrones)):
    matriz[i] = np.array(patrones[i].flatten())

Yd = np.array([1,1,1,1,1,0,0,0,0,0])

m = len(matriz)
Yobt = np.zeros(m)
lr = 0.1
epoch_max = 1000
ECM = []

np.random.seed(42)
w0 = -np.random.rand()
W = np.random.rand((n))

def escalon(z):
    if z > 0:
        return 1
    return 0

epoch = 0
J = 10

X_train = matriz

while (epoch <= epoch_max and J > 0):
    print(f"El número de época es: {epoch}")
    J = (1/(2*m)) * np.sum((Yd - Yobt)**2)
    ECM += [J]

    for i in range(m):
        print(f"Muestra: {i}")
        z = w0 + np.sum(W * X_train[i, :])
        Yobt[i] = escalon(z)
        print(Yobt)

        if Yobt[i] != Yd[i]:
            w0 = w0 - (lr/m) * np.sum(Yobt - Yd)
            W  = W  - (lr/m) * np.sum(Yobt - Yd) * X_train[i, :]
            print(f"W0 = {w0} y W={W}")

    epoch = epoch + 1

plt.figure()
plt.plot(ECM)
plt.title("ECM (Perceptrón)")
plt.xlabel("Época")
plt.ylabel("Error")
plt.grid(True)
plt.show()

np.random.seed(1)
A11, A22, A33, A44, A55 = A, agregar_ruido_propio(A, 0.05), agregar_ruido_propio(A, 0.1), agregar_ruido_propio(A, 0.15), agregar_ruido_propio(A, 0.2)
X11, X22, X33, X44, X55 = X, agregar_ruido_propio(X, 0.05), agregar_ruido_propio(X, 0.1), agregar_ruido_propio(X, 0.15), agregar_ruido_propio(X, 0.20)

fig, axs = plt.subplots(2, 5, figsize=(10, 7))
fig.suptitle("Patrones de test (A y X con ruido propio)", fontsize=14)

patrones_test = [A11, A22, A33, A44, A55, X11, X22, X33, X44, X55]
for i, ax in enumerate(axs.flat):
    ax.imshow(patrones_test[i], cmap='binary')
    ax.set_title(titulos[i])
    ax.axis('off')

plt.tight_layout()
plt.show()

matriz_test = np.zeros((len(patrones_test), n))
for i in range(len(patrones_test)):
    matriz_test[i] = np.array(patrones_test[i].flatten())

Yobt_test = np.zeros(len(matriz_test))
for i in range(len(matriz_test)):
    z = w0 + np.sum(W * matriz_test[i, :])
    Yobt_test[i] = escalon(z)

print("Yobt_test:", Yobt_test)

Yd_test = np.array([1,1,1,1,1,0,0,0,0,0])
acc = np.mean(Yobt_test.astype(int) == Yd_test)
print(f"Exactitud de: {acc*100}%")
