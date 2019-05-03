import numpy as np
import sliceDBR
# import tensorflow as tf
from dbr_dnn_20190503 import Nslice, nh, nl, wavelength, tarwave, statefilename, Rfilename, Nsample

FPATH = 'D:/NBTP_Lab/DBR/DBR_DNN_slice'

lbound = (tarwave/(4*nh))*0.5
ubound = (tarwave/(4*nl))*1.5

print('tarwave: {}, nh: {:.3f}, nl: {:.3f}'.format(tarwave, nh, nl))
print('lbound: {:.3f}, ubound: {:.3f}'.format(lbound, ubound))

for i in range(6):
    sname = FPATH + statefilename + '_' + str(i) + '.csv'
    Rname = FPATH + Rfilename + '_' + str(i) + '.csv'
    n = 0
    while n < Nsample:
        state = np.random.randint(
            low=int(lbound), high=int(ubound), size=Nslice)
        R = sliceDBR.calR(state, Nslice, wavelength, nh, nl, True)
        state = np.reshape(state, (1, Nslice))
        R = np.reshape(R, (1, wavelength.shape[1]))

        with open(sname, "a") as sf:
            np.savetxt(sf, state, fmt='%d', delimiter=',')
        with open(Rname, "a") as Rf:
            np.savetxt(Rf, R, fmt='%.5f', delimiter=',')
        n = n+1

        if (n+1) % 100 == 0:
            print(n+1, ' saved')

print('*****Train Set Prepared*****')