import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

N_pixel = 20
INPUT_SIZE = N_pixel * N_pixel
OUTPUT_SIZE = 50

PATH = 'D:/NBTP_Lab/Machine_Learning/2Dsplitter/2Dsplitter_tensorboard'
TRAIN_PATH = PATH + '/trainset/11'


def getData(): # 03
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

    port3_name = TRAIN_PATH + '/PORT3result_total.csv'
    port3 = pd.read_csv(port3_name, header=None, delimiter=",")
    P3 = port3.values

    Nsample = P1.shape[0]
    x = np.arange(P1.shape[0])
    np.random.shuffle(x)

    sX = sX[x, :]
    P1 = P1[x, :]
    P2 = P2[x, :]
    P3 = P3[x, :]

    return sX, P1, P2, P3, Nsample


def shuffle_data(X, P2):
    Nsample = P2.shape[0]
    x = np.arange(P2.shape[0])
    np.random.shuffle(x)

    X = X[x, :]
    # P1 = P1[x, :]
    P2 = P2[x, :]
    # P3 = P3[x, :]

    return X, P2


def main(n_batch, lr_rate, beta1, beta2):

    # Load training data
    # sX, _, sP2, sP3, wavelength, Nsample = getData()
    sX, sP1, sP2, sP3, Nsample = getData()

    Nr = 0.9
    Nlearning = int(Nr*Nsample)
    Ntest = int((1-Nr)*Nsample)

    trainX = sX[0:Nlearning, :]
    # trainP1 = sP1[0:Nlearning, :]
    trainP2 = sP2[0:Nlearning, :]
    # trainP3 = sP3[0:Nlearning, :]
    trainX_total = trainX
    # trainP1_total = trainP1
    trainP2_total = trainP2
    # trainP3_total = trainP3
    n_copy = 100
    for i in range(n_copy):
        trainX, trainP2 = shuffle_data(trainX, trainP2)
        trainX_total = np.concatenate((trainX_total, trainX), axis=0)
        # trainP1_total = np.concatenate((trainP1_total, trainP1), axis=0)
        trainP2_total = np.concatenate((trainP2_total, trainP2), axis=0)
        # trainP3_total = np.concatenate((trainP3_total, trainP3), axis=0)

    # build network
    X = tf.placeholder(tf.float32, [None, INPUT_SIZE])

    ### Input Layer
    # input: ? * 20 * 20 * 1
    # output: ? * 20 * 20 * 1
    net = tf.reshape(tensor=X, shape=[-1, N_pixel, N_pixel, 1], name='input_reshape')

    ### Hidden Layer
    # input: ? * 20 * 20 * 1
    # output: ? * 20 * 20 * 16
    conv = tf.layers.conv2d(
        inputs=net, filters=16, kernel_size=[3, 3], padding="SAME", strides=1)
    bm = tf.layers.batch_normalization(inputs=conv)
    relu = tf.nn.relu(bm)

    # input: ? * 20 * 20 * 1
    # output: ? * 20 * 20 * 16
    res1 = residual_block(relu, 16, False)
    res2 = residual_block(res1, 16, False)
    res3 = residual_block(res2, 16, False)
    res4 = residual_block(res3, 16, False)
    res5 = residual_block(res4, 16, False)

    # input: ? * 20 * 20 * 16
    # output: ? * 10 * 10 * 32
    res6 = residual_block(res5, 32, True)
    res7 = residual_block(res6, 32, False)
    res8 = residual_block(res7, 32, False)
    res9 = residual_block(res8, 32, False)
    res10 = residual_block(res9, 32, False)

    # input: ? * 10 * 10 * 32
    # output: ? * 5 * 5 * 64
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
    # loss = tf.reduce_mean(tf.square(Y - Yhat))
    loss = tf.reduce_mean(tf.abs(Y - Yhat))
    testloss = tf.reduce_mean(tf.abs(Y - Yhat))
    loss_hist = tf.summary.scalar('loss', loss)
    train = tf.train.AdamOptimizer(learning_rate=lr_rate).minimize(loss)

    with tf.Session() as sess:
        # tensorboard
        net.writer = tf.summary.FileWriter(PATH + '/logs/'+datetime.now().strftime("%Y%m%d%H%M"))
        net.writer.add_graph(sess.graph)
        # Initializer Tensorflow Variables
        sess.run(tf.global_variables_initializer())
        # Train
        for n in range(int(Nlearning*n_copy/n_batch)):
            feed_trainX = np.reshape(
                trainX_total[n*n_batch:(n+1)*n_batch, :], [n_batch, INPUT_SIZE])
            feed_trainY = np.reshape(
                trainP2_total[n*n_batch:(n+1)*n_batch, :], [n_batch, OUTPUT_SIZE])
            feed_train = {X: feed_trainX, Y: feed_trainY}
            sess.run(train, feed_dict=feed_train)
            # log
            if n % 10 == 0:
                merged_summary = tf.summary.merge([loss_hist])
                summary = sess.run(merged_summary, feed_dict=feed_train)
                net.writer.add_summary(summary, global_step=n)
                print(n, 'trained')

        # Test
        test_loss = []
        for n in range(int(Ntest/n_batch)):
            feed_testX = np.reshape(
                sX[n_batch*int(Nlearning/n_batch) + n*n_batch:n_batch*int(Nlearning/n_batch) + (n+1)*n_batch, :],
                [n_batch, INPUT_SIZE])
            feed_testY = np.reshape(
                sP2[n_batch*int(Nlearning/n_batch) + n*n_batch:n_batch*int(Nlearning/n_batch) + (n+1)*n_batch, :],
                [n_batch, OUTPUT_SIZE])
            feed_test = {X: feed_testX, Y: feed_testY}
            test_loss.append(sess.run(testloss, feed_dict=feed_test))
        print("Test absolute lss : ", sum(test_loss)/len(test_loss))
        Ytest = np.reshape(sess.run(Yhat, feed_dict={X: np.reshape(sX[99, :], [1, INPUT_SIZE])}), OUTPUT_SIZE)
        plt.figure(1)
        plt.subplot(2, 1, 1)
        plt.plot(Ytest)
        plt.subplot(2, 1, 2)
        plt.plot(np.reshape(sP2[99, :], OUTPUT_SIZE))
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
    parser.add_argument("--lr_rate", type=float, default=1E-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.7)
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
            'beta2':dict['beta2']
            }

    main(**kwargs)