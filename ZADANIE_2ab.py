import numpy as np
import matplotlib.pyplot as plt

e = np.array([1.25, 0.5])
g = np.array([0.5, 0.2])
h = np.array([0.1, 0.2])

e_b = np.array([5, 5])
g_b = np.array([4, 8])
h_b = np.array([1, 4])

t = np.linspace(0, 10, 1000)

warunki_poczatkowe = np.array([3, 4])

def model_wspolzawodnictwa(t, y0, e, g, h):
    N1, N2 = y0
    dN1dt = (e[0] - g[0] * (h[0] * N1 + h[1] * N2)) * N1
    dN2dt = (e[1] - g[1] * (h[0] * N1 + h[1] * N2)) * N2
    return np.array([dN1dt, dN2dt])

def runge_kutta4(funkcja, y0, t, parametry):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0

    for i in range(1, n):
        dt = t[i] - t[i-1]
        k1 = funkcja(t[i-1], y[i-1], * parametry)
        k2 = funkcja(t[i-1] + dt / 2, y[i-1] + dt / 2 * k1, * parametry)
        k3 = funkcja(t[i-1] + dt / 2, y[i-1] + dt / 2 * k2, * parametry)
        k4 = funkcja(t[i-1] + dt, y[i-1] + dt * k3, * parametry)

        y[i] = y[i-1] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return y

wartosci_a = runge_kutta4(model_wspolzawodnictwa, warunki_poczatkowe, t, (e, g, h))
wartosci_b = runge_kutta4(model_wspolzawodnictwa, warunki_poczatkowe, t, (e_b, g_b, h_b))

plt.figure(figsize=(8, 5))
plt.plot(t, wartosci_a[:, 0], label='N1')
plt.plot(t, wartosci_a[:, 1], linestyle = 'dashed', label='N2')
plt.title('Podpunkt a)')
plt.xlabel('Czas')
plt.ylabel('Populacja')
plt.legend()
plt.show()

plt.plot(t, wartosci_b[:, 0], label='N1')
plt.plot(t, wartosci_b[:, 1], linestyle = 'dashed',label='N2')
plt.title('Podpunkt b)')
plt.xlabel('Czas')
plt.ylabel('Populacja')
plt.legend()
plt.grid(True)

plt.show()