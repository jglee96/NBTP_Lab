import numpy as np
import layerDBR
import matplotlib.pyplot as plt

# Real world environnment
N_layer = 11
# nh = 2.6811  # TiO2 at 400 nm (Siefke)
# nl = 1.4701  # SiO2 at 400 nm (Malitson)
nh = 1.47 # Veroclear at 800 nm
nl = 1 # Air

minwave = 100
maxwave = 1200
wavestep = 5
wavelength = np.array([np.arange(minwave, maxwave, wavestep)])
tarwave = 600

Tstate = np.array([
    int(tarwave/(4*nh)), int(tarwave/(4*nl)),
    int(tarwave/(4*nh)), int(tarwave/(4*nl)),
    int(tarwave/(4*nh)), int(tarwave/(4*nl)),
    int(tarwave/(4*nh)), int(tarwave/(4*nl)),
    int(tarwave/(4*nh)), int(tarwave/(4*nl)),
    int(tarwave/(4*nh))])
TR = layerDBR.calR(Tstate, N_layer, wavelength, nh, nl, True)
Ts = layerDBR.reward(TR, tarwave, 25, wavelength)
Twidth = layerDBR.calFWHM(TR, wavelength, tarwave)

x = np.reshape(wavelength, wavelength.shape[1])
plt.figure(2)
plt.subplot(2, 1, 1)
plt.plot(x, TR)
print(Tstate)
print(Ts, Twidth)

lx = np.arange(N_layer)
plt.subplot(2, 1, 2)
plt.bar(lx, Tstate, width=1, color='blue')
plt.show()