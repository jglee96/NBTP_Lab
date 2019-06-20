import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import pixelDBR
from datetime import datetime
from tensorflow.python.framework import ops

N_pixel = 100
dx = 5
nh = 2.6811  # TiO2 at 400 nm (Siefke)
nl = 1.4701  # SiO2 at 400 nm (Malitson)

minwave = 200
maxwave = 800
wavestep = 5
wavelength = np.array([np.arange(minwave, maxwave, wavestep)])
tarwave = 500
bandwidth = 40

# Base data
th =tarwave/(4*nh)
tl = tarwave/(4*nl)
print("======== Design Information ========")
print('tarwave: {}, nh: {:.3f}, nl: {:.3f}'.format(tarwave, nh, nl))
print('Th: {:.2f}, Tl: {:.2f}'.format(th, tl))

INPUT_SIZE = N_pixel
OUTPUT_SIZE = len(wavelength[0])
batch_size = 64

Nfile = 5
Nsample = 10000
PATH = 'D:/NBTP_Lab/Machine Learning/DBR/DBR_gaussianloglikelihood_lossfunction'
TRAIN_PATH = PATH + '/trainset/01'

def getData():
    # Load Training Data
    print("========      Load Data     ========")
    Xarray = []
    Yarray = []
    for nf in range(Nfile):
        sname = TRAIN_PATH + '/state_' + str(nf) + '.csv'
        Xtemp = pd.read_csv(sname, header=None)
        Xtemp = Xtemp.values
        Xarray.append(Xtemp)

        Rname = TRAIN_PATH + '/R_' + str(nf) + '.csv'
        Ytemp = pd.read_csv(Rname, header=None)
        Ytemp = Ytemp.values
        Yarray.append(Ytemp)

    sX = np.concatenate(Xarray)
    sY = np.concatenate(Yarray)

    return sX, sY

def nll_gaussian(y_pred_mean, y_pred_sd, y_test):
    square = tf.square(y_pred_mean - y_test)
    ms = tf.add(tf.divide(square, 2*y_pred_sd), tf.log(2*np.pi*y_pred_sd)/2)
    # reduce to sclar
    ms = tf.reduce_sum(ms)

    return ms

def main():
    # Load training data
    sX, sY = getData()

    ## Define NN ##
    # Define Input Lyaers
    X = tf.placeholder(tf.float32, [None, INPUT_SIZE], name="input_x")
    net = tf.nn.relu(X)

    # Define hidden Layers
    Hidden_Layer = [200, 200, 200, 200]

    for hnn in Hidden_Layer:
        net = tf.layers.dense(
            net, hnn, activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.contrib.layers.xavier_initializer())
    
    net = tf.layers.dense(
        net, OUTPUT_SIZE, activation=None,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        bias_initializer=tf.contrib.layers.xavier_initializer())
    Rpred = net
    Y = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE], name="output_y")

    # Rpred_sigma = tf.exp(Rpred)
    # loss = nll_gaussian(Rpred, Rpred_sigma, Y)
    square = tf.square(Rpred - Y)
    loss = tf.reduce_sum(tf.divide(square, 2))
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    train = optimizer.minimize(loss)

    with tf.Session() as sess:
        # Init variable of NN
        sess.run(tf.global_variables_initializer())
        
        # Training
        for n in range(int(Nsample*Nfile/batch_size)):
            input_X = np.reshape(sX[n*batch_size:(n+1)*batch_size], [batch_size, INPUT_SIZE])
            output_Y = np.reshape(sY[n*batch_size:(n+1)*batch_size], [batch_size, OUTPUT_SIZE])
            feed = {X: input_X, Y: output_Y}
            sess.run(train, feed_dict=feed)
        
        # Save
        model_name = PATH + "/model/FCDNN_" + datetime.now().strftime("%Y%m%d%H")+'.ckpt'
        saver = tf.train.Saver()
        saver.save(sess, model_name)

        # Testing
        Tstate = np.random.randint(2, size=N_pixel)
        TR = pixelDBR.calR(Tstate, dx, N_pixel, wavelength, nh, nl)
        tX = np.reshape(Tstate, [-1, INPUT_SIZE])
        tY = np.reshape(TR, [-1, OUTPUT_SIZE])
        NR = sess.run(Rpred, feed_dict={X: tX})
        Tloss = sess.run(loss, feed_dict={X: tX, Y: tY})

        print("LOSS: ", Tloss)
        x = np.reshape(wavelength, wavelength.shape[1])
        TR = np.reshape(TR, (120,))
        NR = np.reshape(NR, (120,))
        plt.figure(1)
        plt.subplot(2, 1, 1)
        plt.plot(x, TR)

        plt.subplot(2, 1, 2)
        plt.plot(x, NR)
        plt.show()


def gen_spect_file():
        spect = np.zeros((1, wavelength.shape[1]), dtype=int)
        minrange = 480
        maxrange = 520
        
        minidx = np.where(wavelength == minrange)[1][0]
        maxidx = np.where(wavelength == maxrange)[1][0]

        spect[0, minidx:maxidx+1] = 1
        filename = PATH + "/SpectFile(200,800)_500.csv"
        np.savetxt(filename, spect, delimiter=',')
        print('======== Spectrum File Saved ========')

def binaryRound(x):
    """
    Rounds a tensor whose values are in [0,1] to a tensor with values in {0, 1},
    using the straight through estimator for the gradient.
    """
    g = tf.get_default_graph()

    with ops.name_scope("BinaryRound") as name:
        with g.gradient_override_map({"Round": "Identity"}):
            return tf.round(x, name=name)


def DBS_OptimizeSimulation():
    trained_saver = tf.train.import_meta_graph(PATH + "/model/FCDNN_2019061420.ckpt.meta")
    trained_graph = tf.get_default_graph()

    X = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE], name="input_x")

    W0 = trained_graph.get_tensor_by_name('dense/kernel:0')
    b0 = trained_graph.get_tensor_by_name('dense/bias:0')
    L0 = tf.nn.relu(tf.matmul(X, W0) + b0)

    W1 = trained_graph.get_tensor_by_name('dense_1/kernel:0')
    b1 = trained_graph.get_tensor_by_name('dense_1/bias:0')
    L1 = tf.nn.relu(tf.matmul(L0, W1) + b1)

    W2 = trained_graph.get_tensor_by_name('dense_2/kernel:0')
    b2 = trained_graph.get_tensor_by_name('dense_2/bias:0')
    L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

    W3 = trained_graph.get_tensor_by_name('dense_3/kernel:0')
    b3 = trained_graph.get_tensor_by_name('dense_3/bias:0')
    L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)

    W4 = trained_graph.get_tensor_by_name('dense_4/kernel:0')
    b4 = trained_graph.get_tensor_by_name('dense_4/bias:0')
    Y = tf.add(tf.matmul(L3, W4), b4) # No activation function

    with tf.Session() as sess:
        trained_saver.restore(sess, PATH + "/model/FCDNN_2019061420.ckpt")
        for idx in range(1):
            optimized_x = np.random.randint(2, size=(1, N_pixel))
            optimized_reward = 0
            for n in range(500):
                x = optimized_x
                for i in range(N_pixel):
                    # Toggle
                    x[0][i] = abs(x[0][i] - 1)
                    R = sess.run(Y, feed_dict={X: x})
                    R = np.reshape(R, newshape=(1, wavelength.shape[1]))
                    pre_reward = pixelDBR.reward(R, tarwave, wavelength, bandwidth)[0]
                    if pre_reward > optimized_reward:
                        optimized_x = x
                        optimized_reward = pre_reward
                    else:
                        x[0][i] = abs(x[0][i] - 1)
                    
                    # Swap
                    temp_x = x
                    temp = x[0][i]
                    x[0][i] = x[0][i-1]
                    x[0][i-1] = temp
                    R = sess.run(Y, feed_dict={X: x})
                    R = np.reshape(R, newshape=(1, wavelength.shape[1]))
                    pre_reward = pixelDBR.reward(R, tarwave, wavelength, bandwidth)[0]
                    if pre_reward > optimized_reward:
                        optimized_x = x
                        optimized_reward = pre_reward
                    else:
                        x = temp_x
                if (n % 100) == 0:
                    print("{}th case, {}th epoch, reward: {:.2f}".format(idx, n, optimized_reward))
        optimized_R = sess.run(Y, feed_dict={X: optimized_x})
    print("Optimized result: {:.2f}".format(optimized_reward))

    wavelength_x = np.reshape(wavelength, wavelength.shape[1])
    optimized_R = np.reshape(optimized_R, wavelength.shape[1])
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(wavelength_x, optimized_R)

    pixel_x = np.arange(N_pixel)
    optimized_x = np.reshape(optimized_x, N_pixel)
    plt.subplot(2, 1, 2)
    plt.bar(pixel_x, optimized_x, width=1, color="black")
    plt.show()


def Ratio_OptimizeSimulation():
    trained_saver = tf.train.import_meta_graph(PATH + "/model/FCDNN_2019061420.ckpt.meta")
    trained_graph = tf.get_default_graph()

    init_list_rand = tf.constant(np.random.randint(2, size=(1, N_pixel)), dtype=tf.float32)
    X = tf.get_variable(name='b', initializer=init_list_rand)
    Xint = binaryRound(X)
    Xint = tf.clip_by_value(Xint, clip_value_min=0, clip_value_max=1)
    Y = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE], name="output_y")

    W0 = trained_graph.get_tensor_by_name('dense/kernel:0')
    b0 = trained_graph.get_tensor_by_name('dense/bias:0')
    L0 = tf.nn.relu(tf.matmul(Xint, W0) + b0)

    W1 = trained_graph.get_tensor_by_name('dense_1/kernel:0')
    b1 = trained_graph.get_tensor_by_name('dense_1/bias:0')
    L1 = tf.nn.relu(tf.matmul(L0, W1) + b1)

    W2 = trained_graph.get_tensor_by_name('dense_2/kernel:0')
    b2 = trained_graph.get_tensor_by_name('dense_2/bias:0')
    L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

    W3 = trained_graph.get_tensor_by_name('dense_3/kernel:0')
    b3 = trained_graph.get_tensor_by_name('dense_3/bias:0')
    L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)

    W4 = trained_graph.get_tensor_by_name('dense_4/kernel:0')
    b4 = trained_graph.get_tensor_by_name('dense_4/bias:0')
    Yhat = tf.add(tf.matmul(L3, W4), b4) # No activation function

    Inval = tf.matmul(Y, tf.transpose(Yhat))
    Outval = tf.matmul((1-Y), tf.transpose(Yhat))
    cost = Inval / Outval
    optimizer = tf.train.AdamOptimizer(learning_rate=3E-3).minimize(cost, var_list=[X])

    design_y = pd.read_csv(PATH + "/SpectFile(200,800)_500.csv", header=None)
    design_y = design_y.values

    with tf.Session() as sess:
        trained_saver.restore(sess, PATH + "/model/FCDNN_2019061420.ckpt")
        # sess.run(tf.global_variables_initializer())
        sess.run(tf.variables_initializer([X]))
        for n in range(30000):
            sess.run(optimizer, feed_dict={Y: design_y})
            if (n % 1000) == 0:
                temp_x = np.reshape(Xint.eval().astype(int), newshape=N_pixel)
                temp_R = np.reshape(pixelDBR.calR(temp_x, dx, N_pixel, wavelength, nh, nl), newshape=(1, wavelength.shape[1]))
                temp_reward = pixelDBR.reward(temp_R, tarwave, wavelength, bandwidth)
                print("{}th epoch, reward: {:.2f}".format(n, temp_reward[0]))
        optimized_x = np.reshape(Xint.eval().astype(int), newshape=N_pixel)
        optimized_R = np.reshape(pixelDBR.calR(optimized_x, dx, N_pixel, wavelength, nh, nl), newshape=(1, wavelength.shape[1]))
        optimized_reward = pixelDBR.reward(optimized_R, tarwave, wavelength, bandwidth)
    print("Optimized result: {:.2f}".format(optimized_reward[0]))
    print(optimized_x)

    wavelength_x = np.reshape(wavelength, wavelength.shape[1])
    optimized_R = np.reshape(optimized_R, wavelength.shape[1])
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(wavelength_x, optimized_R)

    pixel_x = np.arange(N_pixel)
    plt.subplot(2, 1, 2)
    plt.bar(pixel_x, optimized_x, width=1, color="black")
    plt.show()

if __name__ == "__main__":
    # main()
    # gen_spect_file()
    Ratio_OptimizeSimulation()
    # DBS_OptimizeSimulation()
