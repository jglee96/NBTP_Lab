import numpy as np
import sliceDBR
import random
# import tensorflow as tf
from datetime import datetime
from dbr_dnn_20190421 import Nslice, epsi, eps0, minwave, maxwave, wavestep, wavelength, tarwave, statefilename, Rfilename, Nsample

FPATH = 'D:/NBTP_Lab/DBR/DBR_DNN_slice'

# X = tf.placeholder(tf.float32, [len(wavelength[0])])
# Y = tf.ones([len(wavelength[0])], tf.float32)
# loss = tf.losses.mean_squared_error(Y, X)

for i in range(7):
    sname = FPATH + statefilename + '_' + str(i+3) + '.txt'
    Rname = FPATH + Rfilename + '_' + str(i+3) + '.txt'
    n = 0
    while n < Nsample:
        state = np.random.randint(int(0.5*tarwave),size=Nslice)
        R = sliceDBR.calR(state,Nslice,wavelength,epsi,eps0,True)

        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer()) # Initialize Tensorflow variables
        #     rloss = sess.run(loss, feed_dict={X: R})
        #     if rloss > 0.05:
        #         with open(sname, "a") as sf:
        #             np.savetxt(sf,state,fmt='%d')
        #         with open(Rname, "a") as Rf:
        #             np.savetxt(Rf,R,fmt='%.5f')
        #         n = n+1

        with open(sname, "a") as sf:
            np.savetxt(sf,state,fmt='%d')
        with open(Rname, "a") as Rf:
            np.savetxt(Rf,R,fmt='%.5f')
        n = n+1

        if (n+1)%100 == 0:
            print(n+1,' saved')

print('*****Train Set Prepared*****')