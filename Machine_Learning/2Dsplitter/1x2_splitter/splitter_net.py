import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from splitter_core import *

N_pixel = 20
INPUT_SIZE = N_pixel * int(N_pixel/2)
OUTPUT_SIZE = 26

PATH = 'D:/NBTP_Lab/Machine_Learning/2Dsplitter/1x2_splitter'
TRAIN_PATH = PATH + '/trainset/05'


def binaryRound(x):
    """
    Rounds a tensor whose values are in [0,1] to a tensor with values in {0, 1},
    using the straight through estimator for the gradient.
    ref: https://r2rt.com/binary-stochastic-neurons-in-tensorflow.html
    """
    g = tf.get_default_graph()

    with ops.name_scope("BinaryRound") as name:
        with g.gradient_override_map({"Round": "Identity"}):
            return tf.round(x, name=name)


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
            sname = TRAIN_PATH + '/' + str(n) + '_index.txt'
            Xintarray = []
            Xtemp = pd.read_csv(sname, header=None, delimiter=",")
            Xstrarray = Xtemp.values
            for j in range(Xstrarray.shape[0]):
                temp = Xstrarray[j][0]
                temp = list(map(int, temp))
                Xintarray.append(temp)
            tempX = np.asarray(Xintarray)
            # file error check
            if n == 0:
                pre_len_check = Xstrarray.shape[0]
            len_check = Xstrarray.shape[0]
            if pre_len_check != len_check:
                print(n, len_check)
            pre_len_check = len_check

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


def Ratio_Optimization(
    output_folder, weight_name_save, n_batch, lr_rate, beta1, beta2,
    decay_steps, lr_decay, momentum, nesterov, num_layers, reg_beta,
    RNnum_block, n_hidden, Dense_list):

    init_list_rand = tf.constant(np.random.randint(2, size=(1, INPUT_SIZE)), dtype=tf.float32)
    X = tf.get_variable(name='b', initializer=init_list_rand)
    Xint = binaryRound(X)
    Xint = tf.clip_by_value(Xint, clip_value_min=0, clip_value_max=1)

    weights, biases = load_weights(output_folder, weight_name_save, num_layers)
    Yhat = FCDNN_forwardprop(Xint, weights, biases, num_layers)

    # cost = 1 / (tf.reduce_mean(Yhat))
    cost = 1 - tf.reduce_min(Yhat)

    # Optimizer
    optimizer = tf.train.AdamOptimizer(
        learning_rate=1E-3).minimize(cost, var_list=[X])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for n in range(100000):
            sess.run(optimizer)
            if (n % 100) == 0:
                check_cost = sess.run(cost)
                check_Yhat = np.mean(sess.run(Yhat))
                print("{}th epoch, cost: {:.4f}, mean: {:.2f}".format(n, check_cost, check_Yhat))
        optimized_x = np.reshape(Xint.eval().astype(int), newshape=(N_pixel, int(N_pixel/2)))
        optimized_cost = sess.run(cost)
        optimized_Y = sess.run(Yhat)
    print("Optimized result: {:.4f}".format(optimized_cost))

    print('[', end='')
    for i,  v1 in enumerate(optimized_x):
        for j, v2 in enumerate(v1):
            if j == (int(N_pixel/2) - 1) and i != (N_pixel-1):
                print(str(v2) + ';')
            elif j == (int(N_pixel/2) - 1) and i == (N_pixel-1):
                print(str(v2) + '];')
            else:
                print(str(v2) + ',', end='')

    wavelength_x = np.arange(OUTPUT_SIZE)
    optimized_Y = np.reshape(optimized_Y, OUTPUT_SIZE)

    plt.figure(1)
    plt.plot(wavelength_x, optimized_Y)

    plt.figure(2)
    plt.imshow(optimized_x, cmap='gray')
    plt.show()


def main(
    output_folder, weight_name_save, n_batch, lr_rate, beta1, beta2,
    decay_steps, lr_decay, momentum, nesterov, num_layers, reg_beta,
    RNnum_block, n_hidden, Dense_list):

    # Load training data
    sX, _, P2 = getData(mode='pack')
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
    n_copy = 10
    for i in range(n_copy):
        trainX_total = np.concatenate((trainX_total, trainX), axis=0)
        trainY_total = np.concatenate((trainY_total, trainY), axis=0)
    trainX, trainY = shuffle_data(trainX, trainY)

    ## Define NN ##
    X = tf.placeholder(tf.float32, [None, INPUT_SIZE])
    Y = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])
    weights = []
    biases = []

    ## FCDNN init
    for i in range(0, num_layers):
        if i == 0:
            weights.append(init_weights((INPUT_SIZE, n_hidden)))
        else:
            weights.append(init_weights((n_hidden, n_hidden)))
        biases.append(init_bias(n_hidden))
    weights.append(init_weights((n_hidden, OUTPUT_SIZE)))
    biases.append(init_bias(OUTPUT_SIZE))

    ## Forward propagation
    Yhat = FCDNN_forwardprop(X, weights, biases, num_layers)

    ## Optimization
    loss = tf.reduce_mean(tf.square(Y-Yhat))
    train = tf.train.AdamOptimizer(
            learning_rate=lr_rate, beta1=beta1, beta2=beta2).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Training
        train_loss = []
        for n in range(int(Nlearning*n_copy/n_batch)):
            input_X = np.reshape(
                trainX_total[n*n_batch:(n+1)*n_batch, :], [n_batch, INPUT_SIZE])
            output_Y = np.reshape(
                trainY_total[n*n_batch:(n+1)*n_batch, :], [n_batch, OUTPUT_SIZE])
            feed = {X: input_X, Y: output_Y}
            _, temp = sess.run([train, loss], feed_dict=feed)
            train_loss.append(temp)

        # Save
        FCDNN_save_weights(weights, biases, output_folder, weight_name_save, num_layers)

        # Test
        test_loss = []
        test_batch = int(n_batch/20)
        for n in range(int(Ntest/test_batch)):
            input_X = np.reshape(testX[n*test_batch:(n+1)*test_batch], [test_batch, INPUT_SIZE])
            output_Y = np.reshape(testY[n*test_batch:(n+1)*test_batch], [test_batch, OUTPUT_SIZE])
            feed = {X: input_X, Y: output_Y}
            temp = sess.run(loss, feed_dict=feed)
            test_loss.append(temp)

        # Example test
        test = np.reshape(sess.run(Yhat, feed_dict={X: np.reshape(testX[99, :], [1, INPUT_SIZE])}), OUTPUT_SIZE)
        plt.figure(1)
        plt.subplot(2, 1, 1)
        plt.plot(test)
        # plt.ylim(0.0, 0.5)
        plt.subplot(2, 1, 2)
        plt.plot(np.reshape(testY[99, :], OUTPUT_SIZE))
        # plt.ylim(0.0, 0.5)

        plt.figure(2)
        plt.plot(train_loss)

        plt.figure(3)
        plt.plot(test_loss)

        test_loss_mean = sum(test_loss) / len(test_loss)
        print(test_loss_mean)
        plt.show()


if __name__=="__main__":
    # for lr_rate in [1E-2, 5E-3, 1E-3, 5E-4, 1E-4]:
    #     for beta1 in [0.9, 0.7, 0.5, 0.3]:
    #         for beta2 in [0.999, 0.9, 0.7, 0.5, 0.3]:
        # for lr_decay in [0.9, 0.5, 0.2]:
        #     for decay_steps in [10, 50, 100, 500]:
        #         for momentum in [0.9, 0.7, 0.5, 0.3, 0.1]:
        #             for nesterov in [True, False]:
    parser = argparse.ArgumentParser(
        description="Physics Net Training")
    parser.add_argument("--output_folder",type=str, default='D:/NBTP_Lab/Machine_Learning/2Dsplitter/1x2_splitter/NN_parameter')
    parser.add_argument("--weight_name_save", type=str, default="")
    parser.add_argument("--n_batch", type=int, default=100)
    parser.add_argument("--lr_rate", type=float, default=1E-3) # 4-0.0055
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--lr_decay", type=float, default=0.9) # 4-0.9
    parser.add_argument("--decay_steps", type=float, default=50)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--nesterov", type=bool, default=True)
    parser.add_argument("--num_layers", default=4)
    parser.add_argument("--reg_beta", type=float, default=0.01)
    parser.add_argument("--RNnum_block", default=5)
    parser.add_argument("--n_hidden", default=100)
    parser.add_argument("--Dense_list", default=[4, 4, 4])
    args = parser.parse_args()
    dict = vars(args)
    print(dict)

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
            'output_folder': dict['output_folder'],
            'weight_name_save': dict['weight_name_save'],
            'n_batch': dict['n_batch'],
            'lr_rate': dict['lr_rate'],
            'beta1': dict['beta1'],
            'beta2': dict['beta2'],
            'lr_decay': dict['lr_decay'],
            'decay_steps': dict['decay_steps'],
            'momentum': dict['momentum'],
            'nesterov': dict['nesterov'],
            'num_layers': int(dict['num_layers']),
            'reg_beta': dict['reg_beta'],
            'RNnum_block': int(dict['RNnum_block']),
            'n_hidden': int(dict['n_hidden']),
            'Dense_list': dict['Dense_list']
        }

    # main(**kwargs)
    Ratio_Optimization(**kwargs)