import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from scipy.stats import pearsonr
from datetime import datetime

N_pixel = 20
INPUT_SIZE = N_pixel * int(N_pixel/2)
OUTPUT_SIZE = 26

PATH = 'D:/NBTP_Lab/Machine_Learning/2Dsplitter/1x2_splitter'
TRAIN_PATH = PATH + '/trainset/06'
Nfile = 12


def getData(mode):
    # Load Training Data
    print("========      Load Data     ========")

    if mode == 'pack':
        sname = TRAIN_PATH + '/index.csv'
        Xtemp = pd.read_csv(sname, header=None, delimiter=",")
        sX = Xtemp.values

        port1_name = TRAIN_PATH + '/PORT1result.csv'
        port1 = pd.read_csv(port1_name, delimiter=",")
        P1 = port1.values

        port2_name = TRAIN_PATH + '/PORT2result.csv'
        port2 = pd.read_csv(port2_name, delimiter=",")
        P2 = port2.values
    elif mode == 'unpack':
        for n in range(Nfile):
            try:
                sname = TRAIN_PATH + '/' + str(n) + '_index.txt'
                Xintarray = []
                Xtemp = pd.read_csv(sname, header=None, delimiter=",")
                Xstrarray = Xtemp.values
                for j in range(Xstrarray.shape[0]):
                    temp = Xstrarray[j][0]
                    temp = list(map(int, temp))
                    Xintarray.append(temp)
                tempX = np.asarray(Xintarray)
            except:
                sname = TRAIN_PATH + '/' + str(n) + '_index.csv'
                Xtemp = pd.read_csv(sname, header=None, delimiter=",")
                tempX = Xtemp.values

            port1_name = TRAIN_PATH + '/' + str(n) + '_PORT1result.csv'
            port1 = pd.read_csv(port1_name, delimiter=",")
            tempP1 = port1.values

            port2_name = TRAIN_PATH + '/' + str(n) + '_PORT2result.csv'
            port2 = pd.read_csv(port2_name, delimiter=",")
            tempP2 = port2.values

            if n == 0:
                sX = tempX
                P1 = tempP1
                P2 = tempP2
            else:
                sX = np.concatenate((sX, tempX), axis=0)
                P1 = np.concatenate((P1, tempP1), axis=0)
                P2 = np.concatenate((P2, tempP2), axis=0)

    return sX, P1, P2


def shuffle_data(X, P1):
    Nsample = P1.shape[0]
    x = np.arange(P1.shape[0])
    np.random.shuffle(x)

    X = X[x, :]
    P1 = P1[x, :]

    return X, P1


def corr_coef(x, y):

    corr, _ = pearsonr(x, y)

    return corr


def main(n_batch, lr_rate, beta1, beta2):

    # Load training data
    sX, _, P2 = getData(mode='unpack')
    Nsample = sX.shape[0]

    Nr = 0.9
    Nlearning = int(Nr*Nsample)
    Ntest = Nsample - Nlearning

    sX, P2 = shuffle_data(sX, P2)
    testX = sX[Nlearning:, :]
    testY = P2[Nlearning:, :]

    trainX = sX[0:Nlearning, :]
    trainY = P2[0:Nlearning, :]
    trainX_total = trainX
    trainY_total = trainY
    n_copy = 5
    for i in range(n_copy):
        trainX_total = np.concatenate((trainX_total, trainX), axis=0)
        trainY_total = np.concatenate((trainY_total, trainY), axis=0)
    trainX, trainY = shuffle_data(trainX, trainY)

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
    # input: ? * 5 * 3 * 64
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
    loss = tf.reduce_mean(tf.square(Y - Yhat))
    loss_hist = tf.summary.scalar('loss', loss)
    train = tf.train.AdamOptimizer(learning_rate=lr_rate).minimize(loss)

    with tf.Session() as sess:
        # Initializer Tensorflow Variables
        sess.run(tf.global_variables_initializer())

        # Train
        train_loss = []
        for n in range(int(Nlearning*n_copy/n_batch)):
            feed_trainX = np.reshape(
                trainX_total[n*n_batch:(n+1)*n_batch, :], [n_batch, INPUT_SIZE])
            feed_trainY = np.reshape(
                trainY_total[n*n_batch:(n+1)*n_batch, :], [n_batch, OUTPUT_SIZE])
            feed_train = {X: feed_trainX, Y: feed_trainY}
            temp, _ = sess.run([loss, train], feed_dict=feed_train)
            train_loss.append(temp)
            if n % 10 == 0:
                print(n, "th train")

        # Test
        test_loss = []
        cc_x = []
        cc_y = []
        test_batch = int(n_batch/20)
        for n in range(int(Ntest/test_batch)):
            feed_testX = np.reshape(
                testX[n*test_batch:(n+1)*test_batch, :], [test_batch, INPUT_SIZE])
            feed_testY = np.reshape(
                testY[n*test_batch:(n+1)*test_batch, :], [test_batch, OUTPUT_SIZE])
            feed_test = {X: feed_testX, Y: feed_testY}
            test_loss.append(sess.run(loss, feed_dict=feed_test))
            cc_x.append(np.mean(testY[n*test_batch]))
            cc_y.append(np.mean(sess.run(Yhat, feed_dict={X: np.reshape(testX[n*test_batch], [1, INPUT_SIZE])})))

        # Example test
        tloss, test = sess.run([loss, Yhat], feed_dict={
            X: np.reshape(testX[99, :], [1, INPUT_SIZE]),
            Y: np.reshape(testY[99, :], [1, OUTPUT_SIZE])})
        test = np.reshape(test, OUTPUT_SIZE)
        plt.figure(1)
        plt.subplot(2, 1, 1)
        plt.plot(test)
        # plt.ylim(0.0, 0.5)
        plt.subplot(2, 1, 2)
        plt.plot(np.reshape(testY[99, :], OUTPUT_SIZE))
        # plt.ylim(0.0, 0.5)
        print(tloss)

        plt.figure(2)
        plt.plot(train_loss)
        # with open(PATH + '/Training_loss(ResNet).csv', 'w') as lossfile:
        #     np.savetxt(lossfile, train_loss, delimiter=',', fmt='%.5f')

        plt.figure(3)
        plt.plot(test_loss)
        # with open(PATH + '/Test_loss(ResNet).csv', 'w') as lossfile:
        #     np.savetxt(lossfile, test_loss, delimiter=',', fmt='%.5f')

        plt.figure(4)
        plt.scatter(cc_x, cc_y)
        corr = corr_coef(cc_x, cc_y)
        print(corr)
        # with open(PATH + '/Corr(ResNet).csv', 'w') as lossfile:
        #     np.savetxt(lossfile, np.reshape(cc_x, [1, len(cc_x)]), delimiter=',', fmt='%.5f')
        # with open(PATH + '/Corr(ResNet).csv', 'a') as lossfile:
        #     np.savetxt(lossfile, np.reshape(cc_y, [1, len(cc_y)]), delimiter=',', fmt='%.5f')

        test_loss_mean = sum(test_loss) / len(test_loss)
        print(test_loss_mean)
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
        inputs=X, filters=num_filter, kernel_size=[4, 4], padding='SAME',
        strides=stride, kernel_initializer=tf.contrib.layers.xavier_initializer())
    bm1 = tf.layers.batch_normalization(inputs=conv1)
    relu1 = tf.nn.relu(bm1)

    conv2 = tf.layers.conv2d(
        inputs=relu1, filters=num_filter, kernel_size=[4, 4],
        padding='SAME', strides=1, kernel_initializer=tf.contrib.layers.xavier_initializer())
    bm2 = tf.layers.batch_normalization(inputs=conv2)

    X_output = tf.nn.relu(bm2 + shortcut)

    return X_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Physics Net Training")
    parser.add_argument("--n_batch", type=int, default=100)
    parser.add_argument("--lr_rate", type=float, default=1E-3)
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