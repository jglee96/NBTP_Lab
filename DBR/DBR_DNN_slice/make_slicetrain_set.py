import numpy as np
import sliceDBR
# import tensorflow as tf
from dbr_dnn_20190421 import Nslice, nh, nl, minwave, maxwave, wavestep, wavelength, tarwave, statefilename, Rfilename, Nsample, Nfile

FPATH = 'D:/NBTP_Lab/DBR/DBR_DNN_slice'

lbound = (tarwave/(4*nh))*0.5
hbound = (tarwave/(4*nl))*1.5

for i in range(29):
    sname = FPATH + statefilename + '_' + str(i+22) + '.txt'
    Rname = FPATH + Rfilename + '_' + str(i+22) + '.txt'
    n = 0
    while n < Nsample:
        state = np.random.randint(low=int(lbound),high=int(hbound),size=Nslice,dtype=int)
        R = sliceDBR.calR(state,Nslice,wavelength,nh,nl,True)

        with open(sname, "a") as sf:
            np.savetxt(sf,state,fmt='%d')
        with open(Rname, "a") as Rf:
            np.savetxt(Rf,R,fmt='%.5f')
        n = n+1

        if (n+1)%100 == 0:
            print(n+1,' saved')

print('*****Train Set Prepared*****')