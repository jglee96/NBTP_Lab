import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from splitter_core import *
from SFOptimizer import SFOptimizer, SFDamping

N_pixel = 20
INPUT_SIZE = N_pixel * N_pixel
OUTPUT_SIZE = 100

PATH = 'D:/NBTP_Lab/Machine Learning/2D splitter/2D splitter_DNN'
TRAIN_PATH = PATH + '/trainset/03'

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


def getData(): # 03
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
    

def main(
    output_folder, weight_name_save, n_batch, lr_rate,
    lr_decay, num_layers, RNnum_block, n_hidden,
    Dense_list, NN_mode, Optimizer_mode):

    # Load training data
    sX, _, sP2, sP3, Nsample = getData()
    sX = np.reshape(sX, (-1, N_pixel, N_pixel, 1))

    Nr = 0.9
    Nlearning = int(Nr*Nsample)
    Ntest = int((1-Nr)*Nsample)

    ## Define NN ##
    g1 = tf.Graph()  # port2 NN
    P2weights = []
    P2biases = []

    g2 = tf.Graph()  # port3 NN
    P3weights = []
    P3biases = []

    with g1.as_default() as g:
        with g.name_scope("g1") as scope:
            X2 = tf.placeholder(tf.float32, [None, N_pixel, N_pixel, 1]) # grayscale (1, 0)
            P2 = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])

            if NN_mode == 0:
                ## CNN init
                P2weights.append(init_weights(3*3*1*32))
                P2weights.append(init_weights(3*3*32*64))
                P2weights.append(init_weights(3*3*64*128))
                P2weights.append(init_weights(3*3*128*256))
                P2weights.append(init_weights(256*128))
                P2weights.append(init_weights(128*100))
                num_layers = 6

                ## Forward propagation
                P2hat = forwardprop(X2, P2weights, P2biases)
            else:
                print("Not Yet")
            
            ## Optimization
            # P2loss = tf.reduce_mean(tf.abs(P2-P2hat))
            P2loss = tf.reduce_mean(tf.square((P2-P2hat)))
            if Optimizer_mode == 0:
                P2train = tf.train.AdamOptimizer(learning_rate=lr_rate, beta1=0.5).minimize(P2loss)
            elif Optimizer_mode == 1:
                global_step = tf.Variable(0, trainable=False)
                learning_rate = tf.train.exponential_decay(
                    lr_rate, global_step, 10, lr_decay, staircase=True)
                P2train = tf.train.RMSPropOptimizer(
                    learning_rate=learning_rate).minimize(
                        P2loss, global_step=global_step)
            elif Optimizer_mode == 2:
                var_list = P2weights + P2biases
                P2train = SFOptimizer(
                    var_list, krylov_dimension=len(var_list),
                    damping_type=SFDamping.marquardt, dtype=tf.float32).minimize(P2loss)
            elif Optimizer_mode == 3:
                global_step = tf.Variable(0, trainable=False)
                learning_rate = tf.train.exponential_decay(
                    lr_rate, global_step, 10, lr_decay, staircase=True)
                P2train = tf.train.GradientDescentOptimizer(
                    learning_rate=learning_rate).minimize(
                        P2loss, global_step=global_step)

    with g2.as_default() as g:
        with g.name_scope("g2") as scope:
            X3 = tf.placeholder(tf.float32, [None, N_pixel, N_pixel, 1])
            P3 = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])

            if NN_mode == 0:
                ## CNN init
                P3weights.append(init_weights(3*3*1*32))
                P3weights.append(init_weights(3*3*32*64))
                P3weights.append(init_weights(3*3*64*128))
                P3weights.append(init_weights(3*3*128*256))
                P3weights.append(init_weights(256*128))
                P3weights.append(init_weights(128*100))
                num_layers = 6

                ## Forward propagation
                P3hat = forwardprop(X3, P3weights, P3biases)

            else:
                print("Not Yet")

            ## Optimization
            # P3loss = tf.reduce_mean(tf.abs(P3-P3hat))
            P3loss = tf.reduce_mean(tf.square((P3-P3hat)))
            if Optimizer_mode == 0:
                P3train = tf.train.AdamOptimizer(learning_rate=lr_rate, beta1=0.5).minimize(P3loss)
            elif Optimizer_mode == 1:
                global_step = tf.Variable(0, trainable=False)
                learning_rate = tf.train.exponential_decay(
                    lr_rate, global_step, 10, lr_decay, staircase=True)
                P3train = tf.train.RMSPropOptimizer(
                    learning_rate=learning_rate).minimize(
                        P3loss, global_step=global_step)
            elif Optimizer_mode == 2:
                var_list = P3weights + P3biases
                P3train = SFOptimizer(
                    var_list, krylov_dimension=len(var_list),
                    damping_type=SFDamping.marquardt, dtype=tf.float32).minimize(P3loss)
            elif Optimizer_mode == 3:
                global_step = tf.Variable(0, trainable=False)
                learning_rate = tf.train.exponential_decay(
                    lr_rate, global_step, 10, lr_decay, staircase=True)
                P3train = tf.train.GradientDescentOptimizer(
                    learning_rate=learning_rate).minimize(
                        P3loss, global_step=global_step)

    with tf.Session(graph=g1) as sess:
        sess.run(tf.global_variables_initializer())

        # Training
        P2trainloss = []
        for n in range(int(Nlearning/n_batch)):
            input_X = np.reshape(sX[n*n_batch:(n+1)*n_batch, :], [n_batch, N_pixel, N_pixel, 1])
            output_P2 = np.reshape(sP2[n*n_batch:(n+1)*n_batch, :], [n_batch, OUTPUT_SIZE])
            feed2 = {X2: input_X, P2: output_P2}
            sess.run(P2train, feed_dict=feed2)
            P2trainloss.append(sess.run(P2loss, feed_dict=feed2))

        # Save
        if NN_mode == 0:
            save_weights(P2weights, P2biases, output_folder + "/P2", weight_name_save, num_layers)
        else:
            print("Not Yet")
        
        P2testloss = []
        # Test
        for n in range(int(Ntest/n_batch)):
            input_X = np.reshape(
                sX[n_batch*int(Nlearning/n_batch) + n*n_batch:n_batch*int(Nlearning/n_batch) + (n+1)*n_batch, :],
                [n_batch, N_pixel, N_pixel, 1])
            output_P2 = np.reshape(
                sP2[n_batch*int(Nlearning/n_batch) + n*n_batch:n_batch*int(Nlearning/n_batch) + (n+1)*n_batch, :],
                [n_batch, OUTPUT_SIZE])
            feed2 = {X2: input_X, P2: output_P2}
            P2testloss.append(sess.run(P2loss, feed_dict=feed2))
        P2test = np.reshape(sess.run(P2hat, feed_dict={X2: np.reshape(sX[99, :], [1, N_pixel, N_pixel, 1])}), OUTPUT_SIZE)
        plt.figure(1)
        plt.subplot(2, 1, 1)
        plt.plot(P2test)
        plt.subplot(2, 1, 2)
        plt.plot(np.reshape(sP2[99, :], OUTPUT_SIZE))

    with tf.Session(graph=g2) as sess:
        sess.run(tf.global_variables_initializer())

        # Training
        P3trainloss = []
        for n in range(int(Nlearning/n_batch)):
            input_X = np.reshape(
                sX[n*n_batch:(n+1)*n_batch, :], [n_batch, N_pixel, N_pixel, 1])
            output_P3 = np.reshape(
                sP3[n*n_batch:(n+1)*n_batch, :], [n_batch, OUTPUT_SIZE])
            feed3 = {X3: input_X, P3: output_P3}
            sess.run(P3train, feed_dict=feed3)
            P3trainloss.append(sess.run(P3loss, feed_dict=feed3))

        # Save
        if NN_mode == 0:
            save_weights(P3weights, P3biases, output_folder + "/P3", weight_name_save, num_layers)
        else:
            print("Not Yet")
        
        P3testloss = []
        # Test
        for n in range(int(Ntest/n_batch)):
            input_X = np.reshape(
                sX[n_batch*int(Nlearning/n_batch) + n*n_batch:n_batch*int(Nlearning/n_batch) + (n+1)*n_batch, :],
                [n_batch, N_pixel, N_pixel, 1])
            output_P3 = np.reshape(
                sP3[n_batch*int(Nlearning/n_batch) + n*n_batch:n_batch*int(Nlearning/n_batch) + (n+1)*n_batch, :],
                [n_batch, OUTPUT_SIZE])
            feed3 = {X3: input_X, P3: output_P3}
            P3testloss.append(sess.run(P3loss, feed_dict=feed3))
        P3test = np.reshape(sess.run(P3hat, feed_dict={X3: np.reshape(sX[99, :], [1, N_pixel, N_pixel, 1])}), OUTPUT_SIZE)
        plt.figure(2)
        plt.subplot(2, 1, 1)
        plt.plot(P3test)
        plt.subplot(2, 1, 2)
        plt.plot(np.reshape(sP3[99, :], OUTPUT_SIZE))

    plt.figure(3)
    plt.plot(P2trainloss)
    plt.plot(P3trainloss)
    
    plt.figure(4)
    plt.plot(P2testloss)
    plt.plot(P3testloss)
    print(P2testloss)
    print(P3testloss)
    plt.show()


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Physics Net Training")
    parser.add_argument("--output_folder",type=str, default='D:/NBTP_Lab/Machine Learning/2D splitter/2D splitter_CNN/NN_parameter')
    parser.add_argument("--weight_name_save", type=str, default="")
    parser.add_argument("--n_batch", type=int, default=100)
    parser.add_argument("--lr_rate", type=float, default=1E-3)
    parser.add_argument("--lr_decay", type=float, default=0.9)
    parser.add_argument("--num_layers", default=10)
    parser.add_argument("--RNnum_block", default=5)
    parser.add_argument("--n_hidden", default=250)
    parser.add_argument("--Dense_list", default=[8, 8, 8, 8, 8, 8, 8, 8])
    parser.add_argument("--NN_mode", default=0) # 0: CNN
    parser.add_argument("--Optimizer_mode", default=0) # 0: Adam, 1: RMSProp, 2: SFO, 3: SGD

    args = parser.parse_args()
    dict = vars(args)
    print(dict)

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
            'output_folder':dict['output_folder'],
            'weight_name_save':dict['weight_name_save'],
            'n_batch':dict['n_batch'],
            'lr_rate':dict['lr_rate'],
            'lr_decay':dict['lr_decay'],
            'num_layers':int(dict['num_layers']),
            'RNnum_block':int(dict['RNnum_block']),
            'n_hidden':int(dict['n_hidden']),
            'Dense_list':dict['Dense_list'],
            'NN_mode':int(dict['NN_mode']),
            'Optimizer_mode':int(dict['Optimizer_mode'])
            }

    main(**kwargs)
    # Ratio_Optimization(**kwargs)