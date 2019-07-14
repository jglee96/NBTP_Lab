import numpy as np
import DBR
import random
import tensorflow as tf
from datetime import datetime
from dbr_dnn_20190414 import Ngrid, dx, epsi, eps0, minwave, maxwave, wavestep, wavelength, tarwave, statefilename, Rfilename, Nsample

FPATH = 'D:/NBTP_Lab/DBR/DBR_DNN'

X = tf.placeholder(tf.float32, [len(wavelength[0])])
Y = tf.ones([len(wavelength[0])], tf.float32)
loss = tf.losses.mean_squared_error(Y, X)

for i in range(10):
    sname = FPATH + statefilename + '_' + str(i) + '.txt'
    Rname = FPATH + Rfilename + '_' + str(i) + '.txt'
    n = 0
    while n < Nsample:
        state = np.random.randint(2,size=Ngrid)
        R = DBR.calR(state,Ngrid,wavelength,dx,epsi,eps0)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer()) # Initialize Tensorflow variables
            rloss = sess.run(loss, feed_dict={X: R})
            if rloss > 0.05:
                with open(sname, "a") as sf:
                    np.savetxt(sf,state,fmt='%d')
                with open(Rname, "a") as Rf:
                    np.savetxt(Rf,R,fmt='%.5f')
                n = n+1

        if (n+1)%100 == 0:
            print(n+1,' saved')

print('*****Train Set Prepared*****')