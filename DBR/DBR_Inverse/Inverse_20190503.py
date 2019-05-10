import numpy as np
import tensorflow as tf
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import tmm_copy
# import sliceDBR
from datetime import datetime

# DBR model
Nslice = 7
nh = 2.6811  # TiO2 at 400 nm (Siefke)
nl = 1.4701  # SiO2 at 400 nm (Malitson)

minwave = 200
maxwave = 800
wavestep = 5
wavelength = np.array([np.arange(minwave, maxwave, wavestep)])
tarwave = 400

# Constants defining our neural network
INPUT_SIZE = Nslice
OUTPUT_SIZE = wavelength.shape[1]

# Base data
lbound = (tarwave/(4*nh))*0.5
ubound = (tarwave/(4*nl))*1.5
thickness = np.arange(start=lbound, stop=ubound)
print("======== Design Information ========")
print('tarwave: {}, nh: {:.3f}, nl: {:.3f}'.format(tarwave, nh, nl))
print('lbound: {:.3f}, ubound: {:.3f}'.format(lbound, ubound))


def gen_spect_file():
        spect = np.zeros((1, wavelength.shape[0]), dtype=int)
        minrange = 600
        maxrange = 650
        minidx = np.where(wavelength == minrange)[0][0]
        maxidx = np.where(wavelength == maxrange)[0][0]

        spect[0, minidx:maxidx+1] = 1
        filename = 'C:/Users/owner/Documents/LJG/DBR_Inverse/data/spec_('+str(minrange)+'_'+str(maxrange)+').csv'
        np.savetxt(filename, spect, delimiter=',')
        print('======== Spectrum File Saved ========')


def getdesignData(filename):
        design_filename = filename + '.csv'
        design = pd.read_csv(design_filename, header=None)
        design = design.values

        return design


# Define custom py_func which takes also a grad op as argument:
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


def main():
        # Load base data
        init_list_rand = tf.constant(
                np.random.rand(1, INPUT_SIZE)*(ubound - lbound) + lbound,
                dtype=tf.float32)
        x = tf.get_variable(name='b', initializer=init_list_rand)
        y = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE])

        minLimit = lbound
        maxLimit = ubound
        x = tf.maximum(x, minLimit)
        x = tf.minimum(x, maxLimit)

        # Calculate Trasfer Matrix with tf
        # n_list = np.empty(INPUT_SIZE)
        # for i in range(Nslice):
        #         if (i % 2) == 0:
        #                 n_list[i] = nh
        #         else:
        #                 n_list[i] = nl
        # n_list = np.concatenate(([1], n_list, [1]))
        # n_list = np.reshape(n_list, (1, INPUT_SIZE + 2))
        # kz_list = 2 * np.pi * n_list / np.transpose(wavelength)  # (wavelength, n_list)
        # num_layer = INPUT_SIZE + 2

        # n_list = tf.constant(n_list, dtype=tf.float32)
        # kz_list = tf.constant(kz_list, dtype=tf.float32)

        # d_list = tf.concat(
        #         [tf.zeros([1, 1]), x, tf.zeros([1, 1])], 1)
        # d_list = tf.reshape(d_list, shape=[-1])
        # delta = tf.multiply(kz_list, d_list)  # (wavelength, n_list)

        # t_list = np.zeros(shape=(num_layer, num_layer), dtype=complex)
        # r_list = np.zeros(shape=(num_layer, num_layer), dtype=complex)
        # for i in range(num_layer-1):
        #         t_list[i, i+1] = tmm_copy.interface_t(
        #                 pol, n_list[i], n_list[i+1],
        #                 th_list[i], th_list[i+1])
        #         r_list[i, i+1] = tmm_copy.interface_r(
        #                 pol, n_list[i], n_list[i+1],
        #                 th_list[i], th_list[i+1])
        # M_list = np.zeros(shape=(num_layer, 2, 2), dtype=complex)

        n_list = np.empty(INPUT_SIZE)
        for i in range(Nslice):
                if (i % 2) == 0:
                        n_list[i] = nh
                else:
                        n_list[i] = nl
        n_list = np.concatenate(([1], n_list, [1]))
        n_list = np.reshape(n_list, (1, INPUT_SIZE + 2))
        kz_list = 2 * np.pi * n_list / np.transpose(wavelength)  # (wavelength, n_list)
        num_layer = INPUT_SIZE + 2

        n_list = tf.constant(n_list, dtype=tf.float32)
        kz_list = tf.constant(kz_list, dtype=tf.float32)

        d_list = tf.concat(
                [tf.zeros([1, 1]), x, tf.zeros([1, 1])], 1)
        d_list = tf.reshape(d_list, shape=[-1])
        delta = tf.multiply(kz_list, d_list)  # (wavelength, n_list)

        temp = tf.Variable(
                initial_value=[[1.0+0.0j, 0.0+0.0j],
                               [0.0+0.0j, 1.0+0.0j]],
                trainable=False,
                dtype=tf.complex)
        Btot = [tf.Variable(
                initial_value=[[1.0+0.0j, 0.0+0.0j],
                               [0.0+0.0j, 1.0+0.0j]],
                trainable=False,
                dtype=tf.complex) for _ in range(OUTPUT_SIZE)]
        R = []
        for w in range(OUTPUT_SIZE):
                for i in range(INPUT_SIZE):
                        Bt = 0.5 * tf.Variable(
                                initial_value=[[(1 + (tf.gather_nd(n_list, [i + 1]) / tf.gather_nd(n_list, [i]))) * tf.exp(-1j * tf.gather_nd(delta, [w, i])),
                                                (1 - (tf.gather_nd(n_list, [i + 1]) / tf.gather_nd(n_list, [i]))) * tf.exp(1j * tf.gather_nd(delta, [w, i]))],
                                               [(1 - (tf.gather_nd(n_list, [i + 1]) / tf.gather_nd(n_list, [i]))) * tf.exp(-1j * tf.gather_nd(delta, [w, i])),
                                                (1 + (tf.gather_nd(n_list, [i + 1]) / tf.gather_nd(n_list, [i]))) * tf.exp(1j * tf.gather_nd(delta, [w, i]))]],
                                trainable=False, dtype=tf.complex)
                        Btot[w] = tf.matmul(Btot, Bt)
                R.append(tf.square(tf.gather_nd(Btot[w], [1, 0]) / tf.gather_nd(Btot[w], [0, 0])))
        # Backward propagation
        # This will select all the values that we want.
        topval = tf.abs(tf.matmul(y, tf.transpose(tf.abs(yhat))))
        # topval = tf.reduce_mean(tf.matmul(y,tf.transpose(tf.abs(yhat))))
        # This will get the values that we do not want
        botval = tf.abs(tf.matmul((1-y), tf.transpose(tf.abs(yhat))))
        # botval = tf.reduce_mean(
        #         tf.matmul(tf.abs(y-1), tf.transpose(tf.abs(yhat))))
        cost = botval/topval
        # global_step = tf.Variable(0, trainable=False)
        # learning_rate = tf.train.exponential_decay(
        #         1E-3,global_step,1000,0.7, staircase=False)
        # optimizer = tf.train.RMSPropOptimizer(
        #         learning_rate=learning_rate).minimize(
        #                 cost, global_step=global_step, var_list=[x])
        optimizer = tf.train.AdamOptimizer(
                learning_rate=1E-2).minimize(cost, var_list=[x])

        # get design data where we want
        design_name = 'C:/Users/owner/Documents/LJG/DBR_Inverse/data/gen_spect(500_550)'
        design_y = getdesignData(design_name)
        numEpoch = 50000
        start_time = time.time()
        print("========                         Iterations started                  ========")
        with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for n in range(numEpoch):
                        sess.run(optimizer, feed_dict={y: design_y})
                        if (n+1) % 500 == 0:
                                loss = sess.run(
                                        cost, feed_dict={y: design_y})[0][0]
                                print('Step: {}, Loss: {:.5f}, X: {}'.format((n+1), loss, x.eval()))
        print("========Iterations completed in : {:.3f} ========".format((time.time()-start_time)))

if __name__ == "__main__":
        train = True
        genspect = False
        if train:  # trin model
                main()
        elif genspect:
                gen_spect_file()