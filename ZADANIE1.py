import numpy as np
import matplotlib.pyplot as plt

K = 100000
r = 0.4
x_75 = 10
czas = (75, 120)

def Gompertz(N, t, r, K):
    return r * N * np.log(K / N)

def Verhulst(N, t, r, K):
    return r * N * ((K - N) / K)

def rozniczkowanie(funkcja, x_75, czas, r, K):
    dt = 0.01
    t = np.arange(75, 120, dt)
    N = np.zeros_like(t)
    N[0] = x_75

    for i in range(1, len(t)):
        N[i] = N[i-1] + dt * funkcja(N[i-1], t[i-1], r, K)

    return t, N

tg, Ng = rozniczkowanie(Gompertz, x_75, czas, r, K)

tv, Nv = rozniczkowanie(Verhulst, x_75, czas, r, K)

plt.plot(tg, Ng, label="Model Gompertz'a")
plt.plot(tv, Nv, label="Model Verhulst'a")
plt.xlabel('t')
plt.ylabel('Objętość')
plt.legend()
plt.grid(True)
plt.show()