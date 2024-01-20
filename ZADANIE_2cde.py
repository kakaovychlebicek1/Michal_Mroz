import numpy as np
import matplotlib.pyplot as plt

e = [0.8, 0.4]
g = [1, 0.5]
h = [0.3, 0.4]

warunki_poczatkowe_c = [4, 8]
warunki_poczatkowe_d = [8, 8]
warunki_poczatkowe_e = [12, 8]

czas = np.linspace(0, 10, 1000)

def model_wspolzawodnictwa(t, y0, e, g, h):
    N1, N2 = y0
    dN1dt = (e[0] - g[0] * (N1 + h[1] * N2)) * N1
    dN2dt = (e[1] - g[1] * (N2 + h[0] * N1)) * N2
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

punkty_na_wykresie = np.array([[4,8], [8,8], [12,8]])

wartosci_c = runge_kutta4(model_wspolzawodnictwa, warunki_poczatkowe_c, czas, (e, g, h))
wartosci_d = runge_kutta4(model_wspolzawodnictwa, warunki_poczatkowe_d, czas, (e, g, h))
wartosci_e = runge_kutta4(model_wspolzawodnictwa, warunki_poczatkowe_e, czas, (e, g, h))

plt.figure(figsize=(8, 6))
plt.plot(wartosci_c[:, 0], wartosci_c[:, 1], label='N1,N2 = (4,8)')
plt.plot(wartosci_d[:, 0], wartosci_d[:, 1], label='N1,N2 = (8,8)')
plt.plot(wartosci_e[:, 0], wartosci_e[:, 1], label='N1,N2 = (12,8)')

plt.scatter(punkty_na_wykresie[:, 0], punkty_na_wykresie[:, 1], color='black')
plt.title('Portret Fazowy')
plt.xlabel('Czas')
plt.ylabel('Populacja')
plt.legend()
plt.grid(True)
plt.show()
