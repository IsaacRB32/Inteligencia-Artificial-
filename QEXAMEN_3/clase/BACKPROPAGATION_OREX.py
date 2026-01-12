import numpy as np
import matplotlib.pyplot as plt

# Parámetros
delta = 0.3
epocas = 3500
ECM = 10

# Datos XOR
X = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
Yd = np.array([0,1,1,0])

# Inicialización
np.random.seed(40)

n_in = 2
n_h = 2
n_out = 1

W1 = -np.random.rand(n_in, n_h)
B1 = -np.random.rand(n_h)

W2 = -np.random.rand(n_h)
B2 = -np.random.rand(1)

def sigmoide(z):
    return 1.0 / (1.0 + np.exp(-z))

def d_sigmoide_z(z):
    s = sigmoide(z)
    return s * (1.0 - s)

# Para graficar
epoca_list = []
error_list = []

Yobt = np.zeros(4)
epoca = 0

while ECM >= 0.2 and epoca < epocas:

    for p in range(4):

        # =====================
        # FORWARD
        # =====================

        z1 = np.dot(X[p], W1) + B1          # (2,)
        y1 = sigmoide(z1)                   # (2,)

        z2 = np.dot(y1, W2) + B2[0]         # escalar
        Yobt[p] = sigmoide(z2)

        # =====================
        # BACKPROP
        # =====================

        error = Yd[p] - Yobt[p]
        delta2 = error * d_sigmoide_z(z2)   # escalar

        delta1 = delta2 * W2 * d_sigmoide_z(z1)  # (2,)

        # =====================
        # ACTUALIZACIÓN
        # =====================

        W2 = W2 + delta * delta2 * y1
        B2[0] = B2[0] + delta * delta2

        W1 = W1 + delta * np.outer(X[p], delta1)
        B1 = B1 + delta * delta1

    # =====================
    # ERROR CUADRÁTICO MEDIO
    # =====================

    ECM = 0.5 * np.sum((Yd - Yobt)**2)

    epoca += 1
    epoca_list.append(epoca)
    error_list.append(ECM)

    print("Epoca:", epoca)
    print("Error:", ECM)
    print("Y obt:", Yobt)

print("Salida final:", Yobt)

plt.plot(epoca_list, error_list, marker='o')
plt.xlabel("Época")
plt.ylabel("Error")
plt.grid()
plt.show()