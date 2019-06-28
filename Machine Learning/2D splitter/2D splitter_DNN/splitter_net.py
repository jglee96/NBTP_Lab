import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from splitter_core import *

N_pixel = 20
INPUT_SIZE = N_pixel * N_pixel
OUTPUT_SIZE = 100

tarwave = 500
bandwidth = 40

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

'''
def getData(): # 01, 02
    # Load Training Data
    print("========      Load Data     ========")

    Xintarray = []

    sname = TRAIN_PATH + '/index.csv'
    Xtemp = pd.read_csv(sname, header=None, dtype=object)
    Xstrarray = Xtemp.values
    for i in range(Nsample):
        temp = Xstrarray[i][0]
        # temp = temp.replace(str(i+1), "", 1)
        temp = list(map(int, temp))
        Xintarray.append(np.array(temp))

    port1_name = TRAIN_PATH + '/PORT1result.csv'
    port1 = pd.read_csv(port1_name, header=None, delimiter=",")
    P1 = port1.values

    port2_name = TRAIN_PATH + '/PORT2result.csv'
    port2 = pd.read_csv(port2_name, header=None, delimiter=",")
    P2 = port2.values

    port3_name = TRAIN_PATH + '/PORT3result.csv'
    port3 = pd.read_csv(port3_name, header=None, delimiter=",")
    P3 = port3.values

    sX = np.asarray(Xintarray)

    wavelength = P1[0,:]
    P1 = np.delete(P1, 0, 0)
    P2 = np.delete(P2, 0, 0)
    P3 = np.delete(P3, 0, 0)

    return sX, P1, P2, P3, wavelength
'''
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

    wavelength = P1[0,:]
    Nsample = P1.shape[0]
    P1 = np.delete(P1, 0, 0)

    return sX, P1, P2, P3, wavelength, Nsample


def Ratio_Optimization(
    output_folder, weight_name_save, n_batch, lr_rate,
    num_layers, RNnum_block, n_hidden, DNN_mode):

    init_list_rand = tf.constant(np.random.randint(2, size=(1, INPUT_SIZE)), dtype=tf.float32)
    X = tf.get_variable(name='b', initializer=init_list_rand)
    Xint = binaryRound(X)
    Xint = tf.clip_by_value(Xint, clip_value_min=0, clip_value_max=1)
    Y = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE], name="output_y")

    P2weights, P2biases = load_weights(output_folder + "/P2", weight_name_save, num_layers)
    P2hat = FCDNN_forwardprop(Xint, P2weights, P2biases, num_layers)
    P2hat = ResNet_forwardprop(Xint, P2weights, P2biases, num_layers)
    P3weights, P3biases = load_weights(output_folder + "/P3", weight_name_save, num_layers)    
    P3hat = FCDNN_forwardprop(Xint, P3weights, P3biases, num_layers)
    P3hat = ResNet_forwardprop(Xint, P3weights, P3biases, num_layers)

    P2Inval = tf.reduce_mean(tf.matmul(Y, tf.transpose(P2hat)))
    P2Outval = tf.reduce_mean(tf.matmul((1-Y), tf.transpose(P2hat)))
    P3Inval = tf.reduce_mean(tf.matmul(Y, tf.transpose(P3hat)))
    P3Outval = tf.reduce_mean(tf.matmul((1-Y), tf.transpose(P3hat)))
    # P2 = P2Outval / P2Inval
    # P3 = P3Outval / P3Inval
    P2 = 1 / P2Inval
    P3 = 1 / P3Inval

    cost = tf.sqrt(P2*P2 + P3*P3)
    # cost = P2 + P3
    # Optimizer
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        lr_rate, global_step, 1000, 0.9, staircase=False)
    # optimizer = tf.train.AdamOptimizer(
    #     learning_rate=learning_rate).minimize(cost, var_list=[X])
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=learning_rate).minimize(
            cost, global_step=global_step, var_list=[X])

    design_y = np.zeros(shape=(1, OUTPUT_SIZE))
    design_y[0, 45:55] = 1.0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for n in range(10000):
            sess.run(optimizer, feed_dict={Y: design_y})
            if (n % 100) == 0:
                temp_cost = sess.run(cost, feed_dict={Y: design_y})
                temp_P2 = sess.run(P2, feed_dict={Y: design_y})
                print("{}th epoch, cost: {:.4f}, P2: {:.2f}".format(n, temp_cost, temp_P2))
        optimized_x = np.transpose(np.reshape(Xint.eval().astype(int), newshape=(N_pixel, N_pixel)))
        optimized_cost = sess.run(cost, feed_dict={Y: design_y})
        optimized_P2 = sess.run(P2hat)
        optimized_P3 = sess.run(P3hat)
    print("Optimized result: {:.4f}".format(optimized_cost))
    
    wavelength_x = np.arange(OUTPUT_SIZE)
    optimized_P2 = np.reshape(optimized_P2, OUTPUT_SIZE)
    optimized_P3 = np.reshape(optimized_P3, OUTPUT_SIZE)

    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(wavelength_x, optimized_P2)
    plt.subplot(2, 1, 2)
    plt.plot(wavelength_x, optimized_P3)

    plt.figure(2)
    plt.imshow(optimized_x, cmap='gray')
    plt.show()
    

def main(
    output_folder, weight_name_save, n_batch, lr_rate,
    lr_decay, huber_delta, num_layers, RNnum_block, n_hidden,
    Dense_list, DNN_mode):

    # Load training data
    sX, _, sP2, sP3, wavelength, Nsample = getData()

    Nr = 0.4
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
            X2 = tf.placeholder(tf.float32, [None, INPUT_SIZE])
            P2 = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])

            if DNN_mode == 0:
                ## FCDNN init
                for i in range(0, num_layers):
                    if i == 0:
                        P2weights.append(init_weights((INPUT_SIZE, n_hidden)))
                    else:
                        P2weights.append(init_weights((n_hidden, n_hidden)))
                    P2biases.append(init_bias(n_hidden))
                P2weights.append(init_weights((n_hidden, OUTPUT_SIZE)))
                P2biases.append(init_bias(OUTPUT_SIZE))

                ## Forward propagation
                P2hat = FCDNN_forwardprop(X2, P2weights, P2biases, num_layers)
            elif DNN_mode == 1:
                ## ResNet init
                # input
                P2weights.append(init_weights((INPUT_SIZE, n_hidden)))
                P2biases.append(init_bias(n_hidden))
                for i in range(0,RNnum_block):
                    # 1st
                    P2weights.append(init_weights((n_hidden, n_hidden)))
                    P2biases.append(init_bias(n_hidden))
                    # 2nd
                    P2weights.append(init_weights((n_hidden, n_hidden)))
                    P2biases.append(init_bias(n_hidden))
                # output
                P2weights.append(init_weights((n_hidden, OUTPUT_SIZE)))
                P2biases.append(init_bias(OUTPUT_SIZE))

                ## Forward propagation
                P2hat = ResNet_forwardprop(X2, P2weights, P2biases, RNnum_block)
            elif DNN_mode == 2:
                ## DenseNet init
                # input
                P2weights.append(init_weights((INPUT_SIZE, n_hidden)))
                P2biases.append(init_bias(n_hidden))
                for n in Dense_list:
                    for i in range(n):
                        P2weights.append(init_weights((n_hidden, n_hidden)))
                        P2biases.append(init_bias(n_hidden))
                # output
                P2weights.append(init_weights((n_hidden, OUTPUT_SIZE)))
                P2biases.append(init_bias(OUTPUT_SIZE))

                ## Forward propagation
                P2hat = DenseNet_forwardprop(X2, P2weights, P2biases, Dense_list)
            
            ## Optimization
            P2loss = tf.reduce_mean(tf.square(P2-P2hat))
            # P2loss = tf.losses.huber_loss(labels=P2, predictions=P2hat, delta=huber_delta)
            # P2train = tf.train.AdamOptimizer(learning_rate=lr_rate).minimize(P2loss)
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(
                lr_rate, global_step, 100, lr_decay, staircase=False)
            P2train = tf.train.RMSPropOptimizer(
                learning_rate=learning_rate).minimize(
                    P2loss, global_step=global_step)

    with g2.as_default() as g:
        with g.name_scope("g2") as scope:
            X3 = tf.placeholder(tf.float32, [None, INPUT_SIZE])
            P3 = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])

            if DNN_mode == 0:
                ## FCDNN init
                for i in range(0, num_layers):
                    if i == 0:
                        P3weights.append(init_weights((INPUT_SIZE, n_hidden)))
                    else:
                        P3weights.append(init_weights((n_hidden, n_hidden)))
                    P3biases.append(init_bias(n_hidden))
                P3weights.append(init_weights((n_hidden, OUTPUT_SIZE)))
                P3biases.append(init_bias(OUTPUT_SIZE))

                ## Forward propagation
                P3hat = FCDNN_forwardprop(X3, P3weights, P3biases, num_layers)

            elif DNN_mode == 1:
                ## ResNet init
                # input
                P3weights.append(init_weights((INPUT_SIZE, n_hidden)))
                P3biases.append(init_bias(n_hidden))
                for i in range(0,RNnum_block):
                    # 1st
                    P3weights.append(init_weights((n_hidden, n_hidden)))
                    P3biases.append(init_bias(n_hidden))
                    # 2nd
                    P3weights.append(init_weights((n_hidden, n_hidden)))
                    P3biases.append(init_bias(n_hidden))
                # output
                P3weights.append(init_weights((n_hidden, OUTPUT_SIZE)))
                P3biases.append(init_bias(OUTPUT_SIZE))

                ## Forward propagation
                P3hat = ResNet_forwardprop(X3, P3weights, P3biases, RNnum_block)
            elif DNN_mode == 2:
                ## DenseNet init
                # input
                P3weights.append(init_weights((INPUT_SIZE, n_hidden)))
                P3biases.append(init_bias(n_hidden))
                for n in Dense_list:
                    for i in range(n):
                        P3weights.append(init_weights((n_hidden, n_hidden)))
                        P3biases.append(init_bias(n_hidden))
                # output
                P3weights.append(init_weights((n_hidden, OUTPUT_SIZE)))
                P3biases.append(init_bias(OUTPUT_SIZE))

                ## Forward propagation
                P3hat = DenseNet_forwardprop(X3, P3weights, P3biases, Dense_list)

            ## Optimization
            P3loss = tf.reduce_mean(tf.square(P3-P3hat))
            # P3loss = tf.losses.huber_loss(labels=P3, predictions=P3hat, delta=huber_delta)
            # P3train = tf.train.AdamOptimizer(learning_rate=lr_rate).minimize(P3loss)
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(
                lr_rate, global_step, 100, lr_decay, staircase=False)
            P3train = tf.train.RMSPropOptimizer(
                learning_rate=learning_rate).minimize(
                    P3loss, global_step=global_step)

    with tf.Session(graph=g1) as sess:
        sess.run(tf.global_variables_initializer())

        # Training
        P2trainloss = []
        for n in range(int(Nlearning/n_batch)):
            input_X = np.reshape(sX[n*n_batch:(n+1)*n_batch, :], [n_batch, INPUT_SIZE])
            output_P2 = np.reshape(sP2[n*n_batch:(n+1)*n_batch, :], [n_batch, OUTPUT_SIZE])
            feed2 = {X2: input_X, P2: output_P2}
            sess.run(P2train, feed_dict=feed2)
            P2trainloss.append(sess.run(P2loss, feed_dict=feed2))

        # Save
        if DNN_mode == 0:
            FCDNN_save_weights(P2weights, P2biases, output_folder + "/FCDNN/P2", weight_name_save, num_layers)
        elif DNN_mode == 1:
            ResNet_save_weights(P2weights, P2biases, output_folder + "/ResNet/P2", weight_name_save, RNnum_block)
        elif DNN_mode == 2:
            DenseNet_save_weights(P2weights, P2biases, output_folder + "/DenseNet/P2", weight_name_save, Dense_list)
        
        P2testloss = []
        # Test
        for n in range(int(Ntest/n_batch)):
            input_X = np.reshape(
                sX[n_batch*int(Nlearning/n_batch) + n*n_batch:n_batch*int(Nlearning/n_batch) + (n+1)*n_batch, :],
                [n_batch, INPUT_SIZE])
            output_P2 = np.reshape(
                sP2[n_batch*int(Nlearning/n_batch) + n*n_batch:n_batch*int(Nlearning/n_batch) + (n+1)*n_batch, :],
                [n_batch, OUTPUT_SIZE])
            feed2 = {X2: input_X, P2: output_P2}
            P2testloss.append(sess.run(P2loss, feed_dict=feed2))
        P2test = np.reshape(sess.run(P2hat, feed_dict={X2: np.reshape(sX[0, :], [1, INPUT_SIZE])}), OUTPUT_SIZE)
        plt.figure(1)
        plt.subplot(2, 1, 1)
        plt.plot(P2test)
        plt.subplot(2, 1, 2)
        plt.plot(np.reshape(sP2[0, :], OUTPUT_SIZE))

    with tf.Session(graph=g2) as sess:
        sess.run(tf.global_variables_initializer())

        # Training
        P3trainloss = []
        for n in range(int(Nlearning/n_batch)):
            input_X = np.reshape(
                sX[n*n_batch:(n+1)*n_batch, :], [n_batch, INPUT_SIZE])
            output_P3 = np.reshape(
                sP3[n*n_batch:(n+1)*n_batch, :], [n_batch, OUTPUT_SIZE])
            feed3 = {X3: input_X, P3: output_P3}
            sess.run(P3train, feed_dict=feed3)
            P3trainloss.append(sess.run(P3loss, feed_dict=feed3))

        # Save
        if DNN_mode == 0:
            FCDNN_save_weights(P3weights, P3biases, output_folder + "/FCDNN/P3", weight_name_save, num_layers)
        elif DNN_mode == 1:
            ResNet_save_weights(P3weights, P3biases, output_folder + "/ResNet/P3", weight_name_save, RNnum_block)
        elif DNN_mode == 2:
            DenseNet_save_weights(P3weights, P3biases, output_folder + "/DenseNet/P3", weight_name_save, Dense_list)
        
        P3testloss = []
        # Test
        for n in range(int(Ntest/n_batch)):
            input_X = np.reshape(
                sX[n_batch*int(Nlearning/n_batch) + n*n_batch:n_batch*int(Nlearning/n_batch) + (n+1)*n_batch, :],
                [n_batch, INPUT_SIZE])
            output_P3 = np.reshape(
                sP3[n_batch*int(Nlearning/n_batch) + n*n_batch:n_batch*int(Nlearning/n_batch) + (n+1)*n_batch, :],
                [n_batch, OUTPUT_SIZE])
            feed3 = {X3: input_X, P3: output_P3}
            P3testloss.append(sess.run(P3loss, feed_dict=feed3))
        P3test = np.reshape(sess.run(P3hat, feed_dict={X3: np.reshape(sX[0, :], [1, INPUT_SIZE])}), OUTPUT_SIZE)
        plt.figure(2)
        plt.subplot(2, 1, 1)
        plt.plot(P3test)
        plt.subplot(2, 1, 2)
        plt.plot(np.reshape(sP3[0, :], OUTPUT_SIZE))

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
    parser.add_argument("--output_folder",type=str, default='D:/NBTP_Lab/Machine Learning/2D splitter/2D splitter_DNN/NN_parameter')
    parser.add_argument("--weight_name_save", type=str, default="")
    parser.add_argument("--n_batch", type=int, default=100)
    parser.add_argument("--lr_rate", type=float, default=1E-2)
    parser.add_argument("--lr_decay", type=float, default=0.9)
    parser.add_argument("--huber_delta", type=float, default=0.05)
    parser.add_argument("--num_layers", default=4)
    parser.add_argument("--RNnum_block", default=4)
    parser.add_argument("--n_hidden", default=120)
    parser.add_argument("--Dense_list", default=[6, 6, 6, 6])
    parser.add_argument("--DNN_mode", default=0) # 0: FCDNN, 1: ResNet, 2: DenseNet

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
            'huber_delta':dict['huber_delta'],
            'num_layers':int(dict['num_layers']),
            'RNnum_block':int(dict['RNnum_block']),
            'n_hidden':int(dict['n_hidden']),
            'Dense_list':dict['Dense_list'],
            'DNN_mode':int(dict['DNN_mode'])
            }

    main(**kwargs)
    # Ratio_Optimization(**kwargs)