import numpy as np
import matplotlib.pyplot as plt

# ======================================
# 1. CREACIÓN DE PATRONES BASE (20x20)
# ======================================

N = 20  # tamaño 20x20

def letra_A_20():
    A = np.zeros((N, N))
    A[2:4, 8:12] = 1
    A[4:18, 4:6] = 1
    A[4:18, 14:16] = 1
    A[2:4, 6:9] = 1
    A[2:4, 11:14] = 1
    A[10:12, 4:16] = 1
    return A

def letra_B_20():
    B = np.zeros((N, N))
    B[2:18, 4:6] = 1
    B[2:4, 4:14] = 1
    B[9:11, 4:14] = 1
    B[16:18, 4:14] = 1
    B[4:9, 14:16] = 1
    B[11:16, 14:16] = 1
    return B

def letra_C_20():
    C = np.zeros((N, N))
    C[4:16, 4:6] = 1
    C[2:4, 6:16] = 1
    C[16:18, 6:16] = 1
    C[3:5, 5:7] = 1
    C[15:17, 5:7] = 1
    C[4:6, 15:17] = 1
    C[14:16, 15:17] = 1
    return C

def letra_D_20():
    D = np.zeros((N, N))
    D[2:18, 4:6] = 1
    D[2:4, 4:12] = 1
    D[16:18, 4:12] = 1
    D[4:16, 14:16] = 1
    D[3:5, 12:15] = 1
    D[15:17, 12:15] = 1
    return D

def letra_E_20():
    E = np.zeros((N, N))
    E[2:18, 4:6] = 1
    E[2:4, 4:16] = 1
    E[9:11, 4:14] = 1
    E[16:18, 4:16] = 1
    return E

def letra_F_20():
    F = np.zeros((N, N))
    F[2:18, 4:6] = 1
    F[2:4, 4:16] = 1
    F[9:11, 4:14] = 1
    return F

def ruido(patron, porcentaje=0.1):
    r = patron.copy()
    n = int(porcentaje * N * N)
    for _ in range(n):
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        r[i, j] = 1 - r[i, j]
    return r


# ======================================
# 2. GENERACIÓN DE DATOS
# ======================================

np.random.seed(1)

patrones_base = [letra_A_20(), letra_B_20(), letra_C_20(),
                 letra_D_20(), letra_E_20(), letra_F_20()]

clases = ["A","B","C","D","E","F"]
labels = np.eye(len(clases))  # one-hot 6x6

X_train, Y_train = [], []
X_test, Y_test = [], []

train_imgs = []
test_imgs = []

for p, y in zip(patrones_base, labels):
    for _ in range(10):     # entrenamiento: 10 por patrón base (inciso)
        r = ruido(p)
        train_imgs.append(r)
        X_train.append(r.flatten())
        Y_train.append(y)
    for _ in range(2):      # prueba: 2 NUEVOS por patrón base (inciso)
        r = ruido(p)
        test_imgs.append(r)
        X_test.append(r.flatten())
        Y_test.append(y)

X_train = np.array(X_train, dtype=float)
Y_train = np.array(Y_train, dtype=float)
X_test  = np.array(X_test,  dtype=float)
Y_test  = np.array(Y_test,  dtype=float)


# ======================================
# 3. VISUALIZACIÓN DE PATRONES
# ======================================

# --- Patrones base ---
plt.figure(figsize=(10,3))
for i, p in enumerate(patrones_base):
    plt.subplot(1, 6, i+1)
    plt.imshow(p, cmap="gray")
    plt.title("Base: " + clases[i])
    plt.axis("off")
plt.suptitle("Patrones base 20x20 (6 letras)")
plt.show()

# --- Patrones ruidosos de entrenamiento (10 por clase) ---
plt.figure(figsize=(12,7))
for i in range(len(train_imgs)):
    plt.subplot(6, 10, i+1)  # 6 clases x 10 patrones
    plt.imshow(train_imgs[i], cmap="gray")
    plt.axis("off")
plt.suptitle("Patrones ruidosos de entrenamiento (10 por clase)")
plt.show()

# --- Patrones ruidosos de prueba (2 por clase) ---
plt.figure(figsize=(12,3))
for i in range(len(test_imgs)):
    plt.subplot(1, 12, i+1)  # 12 = 6*2
    plt.imshow(test_imgs[i], cmap="gray")
    plt.axis("off")
plt.suptitle("Patrones ruidosos de prueba (2 por clase)")
plt.show()


# ======================================
# 4. RED NEURONAL (3 CAPAS)
# ======================================

eta = 0.25
epocas = 2500

n_in  = N * N        # 400 (20x20)
n_h1  = 80           # topología adecuada (ajustada a 400 entradas y 6 clases)
n_h2  = 40
n_out = 6            # 6 letras

W1 = np.random.randn(n_in, n_h1) * 0.1
B1 = np.zeros(n_h1)

W2 = np.random.randn(n_h1, n_h2) * 0.1
B2 = np.zeros(n_h2)

W3 = np.random.randn(n_h2, n_out) * 0.1
B3 = np.zeros(n_out)

def sigmoide(z):
    return 1/(1+np.exp(-z))

def d_sigmoide_z(z):
    s = sigmoide(z)
    return s*(1-s)


# ======================================
# 5. ENTRENAMIENTO
# ======================================

errores = []

for ep in range(epocas):

    ECM = 0.0

    for p in range(len(X_train)):

        z1 = np.dot(X_train[p], W1) + B1
        y1 = sigmoide(z1)

        z2 = np.dot(y1, W2) + B2
        y2 = sigmoide(z2)

        z3 = np.dot(y2, W3) + B3
        y = sigmoide(z3)

        e = Y_train[p] - y
        ECM += np.sum(e**2)

        delta3 = e * d_sigmoide_z(z3)
        delta2 = d_sigmoide_z(z2) * np.dot(W3, delta3)
        delta1 = d_sigmoide_z(z1) * np.dot(W2, delta2)

        W3 += eta * np.outer(y2, delta3)
        B3 += eta * delta3

        W2 += eta * np.outer(y1, delta2)
        B2 += eta * delta2

        W1 += eta * np.outer(X_train[p], delta1)
        B1 += eta * delta1

    errores.append(0.5 * ECM)
    if ep % 50 == 0:
        print("Epoca:", ep, "ECM:", 0.5*ECM)


# ======================================
# 6. ERROR DE ENTRENAMIENTO
# ======================================

plt.plot(errores)
plt.xlabel("Época")
plt.ylabel("Error cuadrático medio")
plt.title("Convergencia del entrenamiento")
plt.grid()
plt.show()


# ======================================
# 7. MAPAS DE ACTIVACIÓN
# ======================================

def forward_map(x):
    z1 = np.dot(x, W1) + B1
    y1 = sigmoide(z1)

    z2 = np.dot(y1, W2) + B2
    y2 = sigmoide(z2)

    z3 = np.dot(y2, W3) + B3
    y = sigmoide(z3)

    return y1, y2, y

y1, y2, y = forward_map(X_test[0])

plt.figure(figsize=(10,3))

plt.subplot(1,3,1)
plt.imshow(y1.reshape(8,10), cmap="hot")  # 80 -> 8x10
plt.title("Activación Oculta 1")
plt.colorbar()

plt.subplot(1,3,2)
plt.imshow(y2.reshape(5,8), cmap="hot")   # 40 -> 5x8
plt.title("Activación Oculta 2")
plt.colorbar()

plt.subplot(1,3,3)
plt.bar(clases, y)
plt.title("Salida")
plt.ylim(0,1)

plt.suptitle("Mapas de activación (patrón de prueba)")
plt.show()


# ======================================
# 8. CLASIFICACIÓN FINAL
# ======================================

print("\nRESULTADOS DE CLASIFICACIÓN\n")

for i in range(len(X_test)):
    _, _, y = forward_map(X_test[i])
    np.set_printoptions(suppress=True, precision=3)
    print("Esperado:", clases[np.argmax(Y_test[i])],
          "Clasificado:", clases[np.argmax(y)],
          "Salida:", y)
