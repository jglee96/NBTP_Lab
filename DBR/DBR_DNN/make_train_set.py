import numpy as np
import DBR
import random
from datetime import datetime
from dbr_dnn_20190414 import Ngrid, dx, epsi, eps0, minwave, maxwave, wavestep, wavelength, tarwave, statefilename, Rfilename, Nsample

for n in range(Nsample):
    state = np.random.randint(2,size=Ngrid)
    R = DBR.calR(state,Ngrid,wavelength,dx,epsi,eps0)

    with open(statefilename, "a") as sf:
        np.savetxt(sf,state,fmt='%d')
    with open(Rfilename, "a") as Rf:
        np.savetxt(Rf,R,fmt='%.5f')

    if (n+1)%100 == 0:
        print(n+1,' saved')

print('*****Train Set Prepared*****')