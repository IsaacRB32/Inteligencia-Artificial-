import numpy as np
import matplotlib.pyplot as plt

# ======================================
# 1. CREACIÓN DE PATRONES BASE (30x30)
# ======================================

def letra_A():
    A = np.zeros((30,30))
    for i in range(30):
        A[i, 15-i//2:15+i//2+1] = 1
    A[15:18,:] = 1
    return A

def letra_Z():
    Z = np.zeros((30,30))
    Z[0,:] = 1
    Z[-1,:] = 1
    for i in range(30):
        Z[i,29-i] = 1
    return Z

def letra_X():
    X = np.zeros((30,30))
    for i in range(30):
        X[i,i] = 1
        X[i,29-i] = 1
    return X

def ruido(patron, porcentaje=0.1):
    r = patron.copy()
    n = int(porcentaje * 30 * 30)
    for _ in range(n):
        i = np.random.randint(0,30)
        j = np.random.randint(0,30)
        r[i,j] = 1 - r[i,j]
    return r

# ======================================
# 2. GENERACIÓN DE DATOS
# ======================================

np.random.seed(1)

patrones_base = [letra_A(), letra_Z(), letra_X()]
clases = ["A","Z","X"]
labels = [[1,0,0],[0,1,0],[0,0,1]]

X_train, Y_train = [], []
X_test, Y_test = [], []

train_imgs = []
test_imgs = []

for p, y in zip(patrones_base, labels):
    for _ in range(9):     # entrenamiento
        r = ruido(p)
        train_imgs.append(r)
        X_train.append(r.flatten())
        Y_train.append(y)
    for _ in range(3):     # prueba (NUEVOS)
        r = ruido(p)
        test_imgs.append(r)
        X_test.append(r.flatten())
        Y_test.append(y)

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

# ======================================
# 3. VISUALIZACIÓN DE PATRONES
# ======================================

# --- Patrones base ---
plt.figure(figsize=(8,3))
for i, p in enumerate(patrones_base):
    plt.subplot(1,3,i+1)
    plt.imshow(p, cmap="gray")
    plt.title("Base: " + clases[i])
    plt.axis("off")
plt.suptitle("Patrones base 30x30")
plt.show()

# --- Patrones ruidosos de entrenamiento ---
plt.figure(figsize=(9,9))
for i in range(len(train_imgs)):
    plt.subplot(3,9,i+1)
    plt.imshow(train_imgs[i], cmap="gray")
    plt.axis("off")
plt.suptitle("Patrones ruidosos de entrenamiento (9 por clase)")
plt.show()

# --- Patrones ruidosos de prueba ---
plt.figure(figsize=(9,3))
for i in range(len(test_imgs)):
    plt.subplot(1,9,i+1)
    plt.imshow(test_imgs[i], cmap="gray")
    plt.axis("off")
plt.suptitle("Patrones ruidosos de prueba (3 por clase)")
plt.show()

# ======================================
# 4. RED NEURONAL (3 CAPAS)
# ======================================

eta = 0.25 ##Learning Rate
epocas = 2500 

##Entrada
n_in = 900 ##Entran 900 neuronas (30x30 pixeles)
##Capas ocultas (Se va reduciendo tipo enbudo)
n_h1 = 60 ##Primera capa de abstracción
n_h2 = 30 ##Segunda capa de abstracción
##Salida
n_out = 3 ##Queremos clasificar 3 letras, cada neurona representa una probabilidad para una de esas letras

W1 = np.random.randn(n_in, n_h1) * 0.1  # Puente de Entrada (900) a Capa Oculta 1 (60)
B1 = np.zeros(n_h1)                    # Ajuste (bias) para las 60 neuronas de la Capa 1

W2 = np.random.randn(n_h1, n_h2) * 0.1 # Puente de Capa Oculta 1 (60) a Capa Oculta 2 (30)
B2 = np.zeros(n_h2)                    # Ajuste (bias) para las 30 neuronas de la Capa 2

W3 = np.random.randn(n_h2, n_out) * 0.1 # Puente de Capa Oculta 2 (30) a Salida Final (3)
B3 = np.zeros(n_out)                   # Ajuste (bias) para las 3 neuronas de Salida

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

    ##Dentro de cada epoca analiza uno por uno los 27 patrones de entrenamiento
    for p in range(len(X_train)):
        #----------------------------------
        ##Forward Propagation
        #----------------------------------

        ## La Primera Capa (De Píxeles a Rasgos)
        z1 = np.dot(X_train[p], W1) + B1
        y1 = sigmoide(z1)
        ## La Segunda Capa (De Rasgos a Formas)
        z2 = np.dot(y1, W2) + B2
        y2 = sigmoide(z2)
        ## La Capa de Salida (La Decisión Final)
        z3 = np.dot(y2, W3) + B3
        y = sigmoide(z3)

        ##sistema de calificación
        e = Y_train[p] - y
        ECM += np.sum(e**2)

        #----------------------------------
        ##Backpropagation
        #----------------------------------
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
    if ep % 20 == 0:
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
plt.imshow(y1.reshape(6,10), cmap="hot")
plt.title("Activación Oculta 1")
plt.colorbar()

plt.subplot(1,3,2)
plt.imshow(y2.reshape(5,6), cmap="hot")
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
    print("Esperado:", clases[np.argmax(Y_test[i])],
          "Clasificado:", clases[np.argmax(y)],
          "Salida:", y)