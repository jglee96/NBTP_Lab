import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

N_pixel = 20

PATH = 'D:/NBTP_Lab/Machine_Learning/2Dsplitter/1x2_splitter_prediction'
TRAIN_PATH = PATH + '/trainset/03'


def getData():
    # Load Training Data
    print("========      Load Data     ========")

    sname = TRAIN_PATH + '/index.csv'
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


def main(n_batch, lr_rate, beta1, beta2):

    # Load training data
    sX, P1, P2 = getData()
    Nsample = P1.shape[0]
    INPUT_SIZE = sX.shape[1]
    OUTPUT_SIZE = P1.shape[1]

    Nr = 0.8
    Nlearning = int(Nr*Nsample)
    Ntest = Nsample - Nlearning

    testX = sX[Nlearning:, :]
    testP = P2[Nlearning:, :]

    trainX = sX[0:Nlearning, :]
    trainP = P2[0:Nlearning, :]
    trainX_total = trainX
    trainP_total = trainP
    n_copy = 70
    for i in range(n_copy):
        trainX, trainP = shuffle_data(trainX, trainP)
        trainX_total = np.concatenate((trainX_total, trainX), axis=0)
        trainP_total = np.concatenate((trainP_total, trainP), axis=0)

    # build network
    X = tf.placeholder(tf.float32, [None, INPUT_SIZE])

    ### Input Layer
    # input: ? * 20 * 10 * 1
    # output: ? * 20 * 10 * 1
    net = tf.reshape(tensor=X, shape=[-1, N_pixel, int(N_pixel/2), 1], name='input_reshape')

    ### Hidden Layer
    # input: ? * 20 * 10 * 1
    # output: ? * 20 * 10 * 16
    conv = tf.layers.conv2d(
        inputs=net, filters=16, kernel_size=[3, 3], padding="SAME", strides=1)
    bm = tf.layers.batch_normalization(inputs=conv)
    relu = tf.nn.relu(bm)

    # input: ? * 20 * 10 * 1
    # output: ? * 20 * 10 * 16
    res1 = residual_block(relu, 16, False)
    res2 = residual_block(res1, 16, False)
    res3 = residual_block(res2, 16, False)
    res4 = residual_block(res3, 16, False)
    res5 = residual_block(res4, 16, False)

    # input: ? * 20 * 10 * 16
    # output: ? * 10 * 5 * 32
    res6 = residual_block(res5, 32, True)
    res7 = residual_block(res6, 32, False)
    res8 = residual_block(res7, 32, False)
    res9 = residual_block(res8, 32, False)
    res10 = residual_block(res9, 32, False)

    # input: ? * 10 * 5 * 32
    # output: ? * 5 * 3 * 64
    res11 = residual_block(res10, 64, True)
    res12 = residual_block(res11, 64, False)
    res13 = residual_block(res12, 64, False)
    res14 = residual_block(res13, 64, False)
    res15 = residual_block(res14, 64, False)

    ### Global Average Pooling
    # input: ? * 5 * 5 * 64
    # output: ? * 1 * 1 * 64
    gap = tf.reduce_mean(res15, [1, 2], keep_dims=True)

    ### Output Layers
    # input: ? * 1 * 1 * 64
    # output: ? * 1 * 1 * 64
    shape = gap.get_shape().as_list()
    dimension = shape[1] * shape[2] * shape[3]
    flat = tf.reshape(gap, [-1, dimension])

    Yhat = tf.layers.dense(flat, OUTPUT_SIZE, activation=None, name="Yhat")

    Y = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])
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


def residual_block(X, num_filter, chg_dim):
    stride = 1

    if chg_dim:
        stride = 2
        pool1 = tf.layers.max_pooling2d(inputs=X, strides=2, pool_size=[1, 1])
        pad1 = tf.pad(pool1, [[0, 0], [0, 0], [0, 0], [int(num_filter/4), int(num_filter/4)]])
        shortcut = pad1
    else:
        shortcut = X

    conv1 = tf.layers.conv2d(
        inputs=X, filters=num_filter, kernel_size=[3, 3], padding='SAME',
        strides=stride, kernel_initializer=tf.contrib.layers.xavier_initializer())
    bm1 = tf.layers.batch_normalization(inputs=conv1)
    relu1 = tf.nn.relu(bm1)

    conv2 = tf.layers.conv2d(
        inputs=relu1, filters=num_filter, kernel_size=[3, 3],
        padding='SAME', strides=1, kernel_initializer=tf.contrib.layers.xavier_initializer())
    bm2 = tf.layers.batch_normalization(inputs=conv2)

    X_output = tf.nn.relu(bm2 + shortcut)

    return X_output


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Physics Net Training")
    parser.add_argument("--n_batch", type=int, default=100)
    parser.add_argument("--lr_rate", type=float, default=1E-5)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    args = parser.parse_args()
    dict = vars(args)

    for key, value in dict.items():
        if (dict[key] == "False"):
            dict[key] = False
        elif dict[key] == "True":
            dict[key] = True
        try:
            if dict[key].is_integer():
                dict[key] = int(dict[key])
            else:
                dict[key] = float(dict[key])
        except:
            pass
    print(dict)

    kwargs = {
            'n_batch': dict['n_batch'],
            'lr_rate': dict['lr_rate'],
            'beta1': dict['beta1'],
            'beta2': dict['beta2']
            }

    main(**kwargs)