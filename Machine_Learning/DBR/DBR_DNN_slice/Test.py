import time
import sliceDBR
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

Nslice = 7
epsi = 2.6**2
eps0 = 1.4**2

minwave = 200
maxwave = 800
wavestep = 5
wavelength = np.array([np.arange(minwave,maxwave,wavestep)])
tarwave = 400

Tstate = np.array([int(tarwave/(4*np.sqrt(epsi))),
                    int(tarwave/(4*np.sqrt(eps0))),
                    int(tarwave/(4*np.sqrt(epsi))),
                    int(tarwave/(4*np.sqrt(eps0))),
                    int(tarwave/(4*np.sqrt(epsi))),
                    int(tarwave/(4*np.sqrt(eps0))),
                    int(tarwave/(4*np.sqrt(epsi)))])

sct = time.time()
TR = sliceDBR.calR(Tstate,Nslice,wavelength,epsi,eps0,True)
print("--- Custom Function %s seconds ---" %(time.time() - sct))
spt = time.time()
tTR = sliceDBR.tcalR(Tstate,Nslice,wavelength,epsi,eps0,True)
print("--- Imported Function %s seconds ---" %(time.time() - spt))

x = np.reshape(wavelength,wavelength.shape[1])
plt.figure(2)
plt.subplot(2,1,1)
plt.plot(x,TR)

plt.subplot(2,1,2)
plt.plot(x,tTR)
plt.show()