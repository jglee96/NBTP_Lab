import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

N_pixel = 20
INPUT_SIZE = N_pixel * N_pixel
OUTPUT_SIZE = 50

PATH = 'D:/NBTP_Lab/Machine_Learning/2Dsplitter/1x2_splitter_prediction'
TRAIN_PATH = PATH + '/trainset/11'


def getData():
    # Load Training Data
    print("========      Load Data     ========")

    sname = TRAIN_PATH + '/index_total.csv'
    Xtemp = pd.read_csv(sname, header=None, delimiter=",")
    sX = Xtemp.values

    port1_name = TRAIN_PATH + '/PORT1result_total.csv'
    port1 = pd.read_csv(port1_name, header=None, delimiter=",")
    P1 = port1.values

    port2_name = TRAIN_PATH + '/PORT2result_total.csv'
    port2 = pd.read_csv(port2_name, header=None, delimiter=",")
    P2 = port2.values

    return sX, P1, P2


def shuffle_data(X, P):
    Nsample = P.shape[0]
    x = np.arange(P.shape[0])
    np.random.shuffle(x)

    X = X[x, :]
    P = P[x, :]

    return X, P


def main(n_batch, lr_rate, beta1, beta2, n_hidden):

    # Load training data
    sX, P1, P2 = getData()
    Nsample = P1.shape[0]

    Nr = 0.9
    Nlearning = int(Nr*Nsample)
    Ntest = Nsample - Nlearning

    testX = sX[Nlearning:, :]
    testP = P2[Nlearning:, :]

    trainX = sX[0:Nlearning, :]
    trainP = P2[0:Nlearning, :]
    trainX_total = trainX
    trainP_total = trainP
    n_copy = 100
    for i in range(n_copy):
        trainX, trainP = shuffle_data(trainX, trainP)
        trainX_total = np.concatenate((trainX_total, trainX), axis=0)
        trainP_total = np.concatenate((trainP_total, trainP), axis=0)

    # build network
    X = tf.placeholder(tf.float32, [None, INPUT_SIZE])
    Y = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])

    net = X
    for i, n in enumerate(n_hidden):
        with tf.name_scope('dense'+str(i)):
            net = tf.layers.dense(net, n, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="dense"+str(i))

    with tf.name_scope('Yhat'):
        net = tf.layers.dense(net, OUTPUT_SIZE, activation=tf.nn.sigmoid, name="Yhat")
    Yhat = net

    loss = tf.reduce_mean(tf.abs(Y - Yhat))
    train = tf.train.AdamOptimizer(learning_rate=lr_rate).minimize(loss)

    with tf.Session() as sess:
        # tensorboard
        # Initializer Tensorflow Variables
        sess.run(tf.global_variables_initializer())
        # Train
        loss_hist = []
        for n in range(int(Nlearning*n_copy/n_batch)):
            feed_trainX = np.reshape(
                trainX_total[n*n_batch:(n+1)*n_batch, :], [n_batch, INPUT_SIZE])
            feed_trainY = np.reshape(
                trainP_total[n*n_batch:(n+1)*n_batch, :], [n_batch, OUTPUT_SIZE])
            feed_train = {X: feed_trainX, Y: feed_trainY}
            sess.run(train, feed_dict=feed_train)
            # log
            if n % 10 == 0:
                loss_hist.append(sess.run(loss, feed_dict=feed_train))
                print(n, 'trained')

        # Test
        test_loss = []
        for n in range(int(Ntest/n_batch)):
            feed_testX = np.reshape(
                testX[n*n_batch:(n+1)*n_batch, :], [n_batch, INPUT_SIZE])
            feed_testY = np.reshape(
                testP[n*n_batch:(n+1)*n_batch, :], [n_batch, OUTPUT_SIZE])
            feed_test = {X: feed_testX, Y: feed_testY}
            test_loss.append(sess.run(loss, feed_dict=feed_test))
        Ytest = np.reshape(sess.run(Yhat, feed_dict={X: np.reshape(testX[99, :], [1, INPUT_SIZE])}), OUTPUT_SIZE)
        plt.figure(1)
        plt.subplot(2, 1, 1)
        plt.plot(Ytest)
        plt.subplot(2, 1, 2)
        plt.plot(np.reshape(testP[99, :], OUTPUT_SIZE))

        plt.figure(2)
        plt.subplot(2, 1, 1)
        plt.plot(loss_hist)
        plt.subplot(2, 1, 2)
        plt.plot(test_loss)
        plt.show()

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Physics Net Training")
    parser.add_argument("--n_batch", type=int, default=100)
    parser.add_argument("--lr_rate", type=float, default=1E-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--n_hidden", default=[200, 200, 200, 200])
    args = parser.parse_args()
    dict = vars(args)

    for key,value in dict.items():
        if (dict[key]=="False"):
            dict[key] = False
        elif dict[key]=="True":
            dict[key] = True
        try:
            if dict[key].is_integer():
                dict[key] = int(dict[key])
            else:
                dict[key] = float(dict[key])
        except:
            pass
    print (dict)

    kwargs = {  
            'n_batch':dict['n_batch'],
            'lr_rate':dict['lr_rate'],
            'beta1':dict['beta1'],
            'beta2':dict['beta2'],
            'n_hidden':dict['n_hidden']
            }

    main(**kwargs)