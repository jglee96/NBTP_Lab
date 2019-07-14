import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import pixelDBR

N_pixel = 100
dx = 5
nh = 2.6811  # TiO2 at 400 nm (Siefke)
nl = 1.4701  # SiO2 at 400 nm (Malitson)

minwave = 200
maxwave = 800
wavestep = 5
wavelength = np.array([np.arange(minwave, maxwave, wavestep)])
tarwave = 400
bandwidth = 50

# Base data
th =tarwave/(4*nh)
tl = tarwave/(4*nl)
print("======== Design Information ========")
print('tarwave: {}, nh: {:.3f}, nl: {:.3f}'.format(tarwave, nh, nl))
print('Th: {:.2f}, Tl: {:.2f}'.format(th, tl))

INPUT_SIZE = N_pixel
OUTPUT_SIZE = len(wavelength[0])
batch_size = 100

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
    net = tf.layers.dense(
        X, 200, activation=None,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        bias_initializer=tf.contrib.layers.xavier_initializer())

    # Define hidden Layers
    Hidden_Layer = [200, 200, 200, 200]

    for hnn in Hidden_Layer:
        res1 = tf.layers.dense(
            net, hnn, activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.contrib.layers.xavier_initializer())
        res2 = tf.layers.dense(
            res1, hnn, activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.contrib.layers.xavier_initializer())
        net = tf.nn.relu(res2 + net)
    
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

if __name__ == "__main__":
    main()
