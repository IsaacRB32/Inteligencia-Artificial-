import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# ======================================
# 1. CARGA DE DATASET (>=4 características, >=4 clases)
# ======================================

np.random.seed(1)

digits = load_digits() ##Cargamos el dataste 

##Desenrollando y pasando a flotantes las 
X = digits.data.astype(float)        # (n_samples, 64)
##Son las clases reales, por eso se necesitan que sean enteros
y_int = digits.target.astype(int)    # clases 0..9

# Normalización simple para sigmoide: pixeles de 0..16 a 0..1 (Desvanecimiento del Gradiente)
X = X / 16.0

clases = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']  # 10 clases: "0"..."9"
n_out = len(clases)  # 10

# One-hot,acomodador automático de 1,797 imágenes
Y = np.eye(n_out)[y_int]

# Split train/test (80/20) 
##Vamos a barajear ya que vienen en orden 
idx = np.random.permutation(len(X))
##El corte ya que con 80% vamos a entrenar 
cut = int(0.8 * len(X))
## train
train_idx = idx[:cut]
## test
test_idx = idx[cut:]

X_train = X[train_idx]  ##entradas para entrenar
Y_train = Y[train_idx] ##targets (one-hot) para entrenar
X_test  = X[test_idx] ##entradas nuevas para evaluar
Y_test  = Y[test_idx] ##targets reales (one-hot) para comparar

# ======================================
# 2. VISUALIZACIÓN DE MUESTRAS
# ======================================

plt.figure(figsize=(10, 5))
##Los primeros 10
for i in range(10):
    plt.subplot(2, 5, i+1)
    ##Los dobla 
    plt.imshow(X_train[i].reshape(8, 8), cmap="gray")
    ##Imprime el numero de la posición del vector One-Hot
    plt.title("Base: " + clases[int(np.argmax(Y_train[i]))])
    plt.axis("off")
plt.suptitle("Muestras del dataset Digits (8x8)")
plt.show()

# ======================================
# 4. RED NEURONAL (3 CAPAS) 
# ======================================

eta = 0.25
epocas = 250  # ajustado porque ahora hay MUCHOS patrones (dataset real)

n_in = X_train.shape[1]   # 64 características

# Topología adecuada para 64 features y 10 clases (embudo)
n_h1 = 32
n_h2 = 16

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

    # Barajar patrones por época (mejora convergencia sin cambiar tu lógica)
    orden = np.random.permutation(len(X_train))

    for p in orden:
        # Forward
        z1 = np.dot(X_train[p], W1) + B1
        y1 = sigmoide(z1)

        z2 = np.dot(y1, W2) + B2
        y2 = sigmoide(z2)

        z3 = np.dot(y2, W3) + B3
        y = sigmoide(z3)

        # Error
        e = Y_train[p] - y
        ECM += np.sum(e**2)

        # Backprop
        delta3 = e * d_sigmoide_z(z3)
        delta2 = d_sigmoide_z(z2) * np.dot(W3, delta3)
        delta1 = d_sigmoide_z(z1) * np.dot(W2, delta2)

        # Update
        W3 += eta * np.outer(y2, delta3)
        B3 += eta * delta3

        W2 += eta * np.outer(y1, delta2)
        B2 += eta * delta2

        W1 += eta * np.outer(X_train[p], delta1)
        B1 += eta * delta1

    errores.append(0.5 * ECM)

    if ep % 10 == 0:
        print("Epoca:", ep, "ECM:", 0.5*ECM)

# ======================================
# 6. ERROR DE ENTRENAMIENTO
# ======================================

plt.plot(errores)
plt.xlabel("Época")
plt.ylabel("Error cuadrático medio")
plt.title("Convergencia del entrenamiento (Digits)")
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
plt.imshow(y1.reshape(4,8), cmap="hot")   # 32 -> 4x8
plt.title("Activación Oculta 1")
plt.colorbar()

plt.subplot(1,3,2)
plt.imshow(y2.reshape(4,4), cmap="hot")   # 16 -> 4x4
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

print("\nRESULTADOS DE CLASIFICACIÓN (TEST)\n")
np.set_printoptions(suppress=True, precision=3)

for i in range(10):  # muestra 10 ejemplos
    _, _, y_pred = forward_map(X_test[i])
    print("Esperado:", clases[np.argmax(Y_test[i])],
          "Clasificado:", clases[np.argmax(y_pred)],
          "Salida:", y_pred)

# ======================================
# 9. ingresar el usuario nuevas características
# ======================================
# # El usuario dibuja en una cuadrícula 32x32.
# # Internamente se reduce a 8x8 (promedio por bloques 4x4),
# # se normaliza igual que el dataset Digits y se clasifica.

N_draw = 32
N_target = 8
block = N_draw // N_target  # 32 / 8 = 4

draw_grid = np.zeros((N_draw, N_draw), dtype=float)

fig, ax = plt.subplots(figsize=(6, 6))

img = ax.imshow(
    draw_grid,
    cmap="gray",
    vmin=0, vmax=1,
    interpolation="nearest",
    resample=False,
    extent=(0, N_draw, N_draw, 0)
)

ax.set_aspect("equal")
ax.set_title(
    "Dibuja un dígito (32x32)\n"
    "Arrastrar clic izq: pintar | Arrastrar clic der: borrar\n"
    "p: predecir | c: limpiar | q: salir"
)

ax.set_xticks(np.arange(0, N_draw+1, 1))
ax.set_yticks(np.arange(0, N_draw+1, 1))
ax.grid(True, linewidth=0.3)
ax.set_xticklabels([])
ax.set_yticklabels([])

np.set_printoptions(suppress=True, precision=3)

# ------------------------------
# Variables de estado del mouse
# ------------------------------
pintando = False
borrando = False

def actualizar():
    img.set_data(draw_grid)
    fig.canvas.draw_idle()

def downsample_32_to_8(g32):
    """
    Convierte 32x32 -> 8x8 promediando bloques 4x4.
    Escala a 0..16 para parecerse al rango original Digits.
    """
    g8 = np.zeros((N_target, N_target), dtype=float)
    for i in range(N_target):
        for j in range(N_target):
            bloque = g32[i*block:(i+1)*block, j*block:(j+1)*block]
            g8[i, j] = bloque.mean()
    return g8 * 16.0  # 0..16

def paint_at(event):
    if event.inaxes != ax or event.xdata is None or event.ydata is None:
        return

    j = int(np.floor(event.xdata))
    i = int(np.floor(event.ydata))

    if 0 <= i < N_draw and 0 <= j < N_draw:
        if pintando:
            draw_grid[i, j] = 1.0
            actualizar()
        elif borrando:
            draw_grid[i, j] = 0.0
            actualizar()

def on_press(event):
    global pintando, borrando
    if event.inaxes != ax:
        return

    if event.button == 1:      # clic izquierdo
        pintando = True
        borrando = False
    elif event.button == 3:    # clic derecho
        borrando = True
        pintando = False

    paint_at(event)  # pinta/borrra también el primer punto

def on_release(event):
    global pintando, borrando
    pintando = False
    borrando = False

def on_move(event):
    paint_at(event)

def on_key(event):
    if event.key == "c":
        draw_grid[:, :] = 0.0
        ax.set_title(
            "Dibuja y presiona 'p'.\n"
            "Arrastrar clic izq: pintar | Arrastrar clic der: borrar\n"
            "p: predecir | c: limpiar | q: salir"
        )
        actualizar()

    elif event.key == "p":
        # 1) Reducir 32x32 a 8x8
        g8 = downsample_32_to_8(draw_grid)

        # 2) Normalizar igual que el entrenamiento (0..16 -> 0..1)
        x_user = g8.flatten() / 16.0

        # 3) Clasificar
        _, _, y_user = forward_map(x_user)
        pred = clases[int(np.argmax(y_user))]

        ax.set_title(
            f"Clasificado: {pred} | Salida: {np.round(y_user, 3)}\n"
            "Arrastrar clic izq: pintar | Arrastrar clic der: borrar\n"
            "p: predecir | c: limpiar | q: salir"
        )
        actualizar()

        # Mostrar lo que realmente entra a la red (8x8)
        plt.figure(figsize=(3, 3))
        plt.imshow(g8, cmap="gray", interpolation="nearest", resample=False)
        plt.title("Entrada real a la red (8x8)")
        plt.axis("off")
        plt.show()

    elif event.key == "q":
        plt.close(fig)

# Conexiones de eventos
fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("button_release_event", on_release)
fig.canvas.mpl_connect("motion_notify_event", on_move)
fig.canvas.mpl_connect("key_press_event", on_key)

plt.show()
