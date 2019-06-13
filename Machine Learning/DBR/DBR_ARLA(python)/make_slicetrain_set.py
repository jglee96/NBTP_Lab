import numpy as np
import pixelDBR
# import tensorflow as tf
from dbr_arla_20190517 import N_pixel, dx, nh, nl, wavelength, tarwave, Nsample

PATH = 'D:/NBTP_Lab/DBR/DBR_ARLA(python)'
TRAIN_PATH = PATH + '/trainset' + '/01'

print('N_pixel: {}, nh: {:.3f}, nl: {:.3f}'.format(N_pixel, nh, nl))

for i in range(10):
    sname = TRAIN_PATH + '/state_' + str(i) + '.csv'
    Rname = TRAIN_PATH + '/R_' + str(i) + '.csv'
    n = 0
    for n in range(Nsample):
        state = np.random.randint(2, size=N_pixel)

        R = pixelDBR.calR(state, dx, N_pixel, wavelength, nh, nl)
        state = np.reshape(state, (1, N_pixel))
        R = np.reshape(R, (1, wavelength.shape[1]))

        with open(sname, "a") as sf:
            np.savetxt(sf, state, fmt='%d', delimiter=',')
        with open(Rname, "a") as Rf:
            np.savetxt(Rf, R, fmt='%.5f', delimiter=',')

        if (n+1) % 1000 == 0:
            print('{}th {}step'.format(i+1, n+1))

print('*****Train Set Prepared*****')