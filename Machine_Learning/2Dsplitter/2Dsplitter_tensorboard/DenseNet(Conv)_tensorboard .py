import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from tflearn.layers.conv import global_avg_pool
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
    net = tf.reshape(tensor=X, shape=[-1, N_pixel, N_pixel, 1], name='input_reshape')

    Yhat = Dense_net(X=net, num_filter=10, nb_blocks=[6])
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


def bottleneck_layer(X, num_filter, scope):
    with tf.name_scope(scope):
        net = tf.layers.batch_normalization(inputs=X)
        net = tf.nn.relu(net)
        net = tf.layers.conv2d(
            inputs=net, filters=4*num_filter, kernel_size=[1, 1], padding='SAME',
            strides=1, kernel_initializer=tf.contrib.layers.xavier_initializer())

        net = net = tf.layers.batch_normalization(inputs=net)
        net = tf.nn.relu(net)
        net = tf.layers.conv2d(
            inputs=net, filters=4*num_filter, kernel_size=[3, 3], padding='SAME',
            strides=1, kernel_initializer=tf.contrib.layers.xavier_initializer())

        return net


def transition_layer(X, scope):
    with tf.name_scope(scope):
        net = tf.layers.batch_normalization(inputs=X)
        net = tf.nn.relu(net)

        in_channel = X.shape[-1].value
        net = tf.layers.conv2d(
            inputs=net, filters=int(in_channel*0.5), kernel_size=[1, 1], padding='SAME',
            strides=2, kernel_initializer=tf.contrib.layers.xavier_initializer())
        net = tf.layers.average_pooling2d(inputs=net, pool_size=[2, 2], strides=2, padding='VALID')

        return net


def dense_block(X, num_filter, nb_layers, layer_name):
    with tf.name_scope(layer_name):
        layers_concat = list()
        layers_concat.append(X)

        net = bottleneck_layer(X, num_filter, scope=layer_name+'_bottleN_' + str(0))

        layers_concat.append(net)

        for i in range(nb_layers - 1):
            net = tf.concat(layers_concat, axis=3)

        return net


def Dense_net(X, num_filter, nb_blocks):
    conv = tf.layers.conv2d(
        inputs=X, filters=2*num_filter, kernel_size=[3, 3], padding='SAME',
        strides=2, kernel_initializer=tf.contrib.layers.xavier_initializer())
    net = tf.layers.max_pooling2d(inputs=X, strides=2, pool_size=[3, 3], padding='VALID')

    for idx, i in enumerate(nb_blocks):
        net = dense_block(X=net, num_filter=num_filter, nb_layers=i, layer_name='dense_'+str(idx))
        net = transition_layer(net, scope='trans_'+str(idx))

    net = tf.layers.batch_normalization(inputs=net)
    net = tf.nn.relu(net)
    net = global_avg_pool(net, name='Global_avg_pooling')
    net = tf.contrib.layers.flatten(net)
    net = tf.layers.dense(inputs=net, units=OUTPUT_SIZE, name='linear')

    return net


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