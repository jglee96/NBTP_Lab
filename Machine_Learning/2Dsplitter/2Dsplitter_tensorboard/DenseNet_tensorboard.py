import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

N_pixel = 20
INPUT_SIZE = N_pixel * N_pixel
OUTPUT_SIZE = 50

PATH = 'D:/LJG/2Dsplitter'
TRAIN_PATH = PATH + '/trainset/10'


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
    P2 = P2[x, :]

    return X, P2


def main(n_batch, lr_rate, beta1, beta2, n_hidden, n_block):

    # Load training data
    sX, sP1, sP2, sP3, Nsample = getData()

    Nr = 0.9
    Nlearning = int(Nr*Nsample)
    Ntest = int((1-Nr)*Nsample)

    trainX = sX[0:Nlearning, :]
    trainP2 = sP2[0:Nlearning, :]
    trainX_total = trainX
    trainP2_total = trainP2

    n_copy = 100
    for i in range(n_copy):
        trainX, trainP2 = shuffle_data(trainX, trainP2)
        trainX_total = np.concatenate((trainX_total, trainX), axis=0)
        trainP2_total = np.concatenate((trainP2_total, trainP2), axis=0)

    with tf.device('/GPU:0'):
        # build network
        X = tf.placeholder(tf.float32, [None, INPUT_SIZE])
        Y = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])

    with tf.device('/GPU:0'):
        net = tf.layers.dense(X, n_hidden[0], activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="transe")
    net_hist = []
    for idx, n in enumerate(n_block):
        dense = []
        dense.append(net)
        for i in range(n):
            with tf.name_scope('dense'+str(idx)+'_'+str(i)):
                with tf.device('/GPU:0'):
                    net = tf.layers.dense(net, n_hidden[idx], activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='dense'+str(idx)+'_'+str(i))
                dense.append(net)
                net_hist.append(tf.summary.histogram("activations"+str(idx)+'_'+str(i), net))
                dense_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "dense"+str(idx)+'_'+str(i))
                net_hist.append(tf.summary.histogram("weights"+str(idx)+'_'+str(i), dense_vars[0]))
                net_hist.append(tf.summary.histogram("biases"+str(idx)+'_'+str(i), dense_vars[1]))
        with tf.device('/GPU:0'):
            net = tf.concat(dense, axis=0)
            net = tf.nn.relu(net)

    with tf.name_scope('Yhat'):
        with tf.device('/GPU:0'):
            Yhat = tf.layers.dense(net, OUTPUT_SIZE, activation=None, name="Yhat")
        net_hist.append(tf.summary.histogram("activations_Yhat", Yhat))
        dense_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Yhat")
        net_hist.append(tf.summary.histogram("weights_Yhat", dense_vars[0]))
        net_hist.append(tf.summary.histogram("biasess_Yhat", dense_vars[1]))

    with tf.device('/GPU:0'):
        # loss = tf.reduce_mean(tf.square(Y - Yhat))
        loss = tf.reduce_mean(tf.abs(Y - Yhat))
        testloss = tf.reduce_mean(tf.abs(Y - Yhat))
        train = tf.train.AdamOptimizer(learning_rate=lr_rate).minimize(loss)
    loss_hist = tf.summary.scalar('loss', loss)

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
            if n%10 == 0:
                merged_summary = tf.summary.merge([loss_hist] + net_hist)
                summary =sess.run(merged_summary, feed_dict=feed_train)
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

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Physics Net Training")
    parser.add_argument("--n_batch", type=int, default=100)
    parser.add_argument("--lr_rate", type=float, default=1E-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.7)
    parser.add_argument("--n_hidden", default=[100, 100, 100])
    parser.add_argument("--n_block", default=[6, 12, 48])
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
            'n_hidden':dict['n_hidden'],
            'n_block':dict['n_block']
            }

    main(**kwargs)