import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from splitter_core import *

N_pixel = 20

tarwave = 500
bandwidth = 40

Nsample = 10000
PATH = 'D:/NBTP_Lab/Machine Learning/2D splitter/2D splitter_FCDNN'
TRAIN_PATH = PATH + '/trainset/01'

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


def getData():
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


def Ratio_Optimization(output_folder, weight_name_save, n_batch, lr_rate, num_layers, n_hidden):
    init_list_rand = tf.constant(np.random.randint(2, size=(1, N_pixel)), dtype=tf.float32)
    X = tf.get_variable(name='b', initializer=init_list_rand)
    Xint = binaryRound(X)
    Xint = tf.clip_by_value(Xint, clip_value_min=0, clip_value_max=1)
    Y = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE], name="output_y")
    weights, biases = load_weights(output_folder, weight_name_save, num_layers)
    Yhat = forwardprop(Xint, weights, biases, num_layers)

    Inval = tf.matmul(Y, tf.transpose(Yhat))
    Outval = tf.matmul((1-Y), tf.transpose(Yhat))
    # cost = Inval / Outval
    cost = Outval / Inval
    optimizer = tf.train.AdamOptimizer(learning_rate=1E-3).minimize(cost, var_list=[X])

    design_y = pd.read_csv(PATH + "/SpectFile(200,800)_500.csv", header=None)
    design_y = design_y.values

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for n in range(10000):
            sess.run(optimizer, feed_dict={Y: design_y})
            if (n % 100) == 0:
                temp_R = np.reshape(sess.run(Yhat), newshape=(1, wavelength.shape[1]))
                temp_cost = sess.run(cost, feed_dict={Y: design_y})[0][0]
                temp_reward = pixelDBR.reward(temp_R, tarwave, wavelength, bandwidth)
                print("{}th epoch, reward: {:.4f}, cost: {:.4f}".format(n, temp_reward[0], temp_cost))
        optimized_x = np.reshape(Xint.eval().astype(int), newshape=N_pixel)
        # optimized_R = np.reshape(sess.run(Yhat), newshape=(1, wavelength.shape[1]))
        optimized_R = np.reshape(pixelDBR.calR(optimized_x, dx, N_pixel, wavelength, nh, nl), newshape=(1, wavelength.shape[1]))
        optimized_reward = pixelDBR.reward(optimized_R, tarwave, wavelength, bandwidth)
    print("Optimized result: {:.4f}".format(optimized_reward[0]))
    print(optimized_x)

    wavelength_x = np.reshape(wavelength, wavelength.shape[1])
    optimized_R = np.reshape(optimized_R, wavelength.shape[1])
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(wavelength_x, optimized_R)

    pixel_x = np.arange(N_pixel)
    plt.subplot(2, 1, 2)
    plt.bar(pixel_x, optimized_x, width=1, color="black")
    plt.show()
    

def main(output_folder, weight_name_save, n_batch, lr_rate, num_layers, n_hidden):
    # Load training data
    sX, _, sP2, sP3, wavelength = getData()

    INPUT_SIZE = N_pixel
    OUTPUT_SIZE = wavelength.shape[1]

    ## Define NN ##
    X = tf.placeholder(tf.float32, [None, INPUT_SIZE], name="input_x")
    P2 = tf.placeholder(tf.float32, [None, OUTPUT_SIZE], name="output_port2")
    P3 = tf.placeholder(tf.float32, [None, OUTPUT_SIZE], name="output_port3")
    P2weights = []
    P2biases = []
    P3weights = []
    P3biases = []
    NN_P2 = Splitter
    NN_P3 = Splitter

    for i in range(0,num_layers):
        if i == 0:
            P2weights.append(NN_P2.init_weights((INPUT_SIZE, n_hidden)))
            P3weights.append(NN_P3.init_weights((INPUT_SIZE, n_hidden)))
        else:
            P2weights.append(NN_P2init_weights((n_hidden, n_hidden)))
            P3weights.append(NN_P3init_weights((n_hidden, n_hidden)))
        P2biases.append(NN_P2.init_bias(n_hidden))
        P3biases.append(NN_P3.init_bias(n_hidden))
    P2weights.append(NN_P2.init_weights((n_hidden, OUTPUT_SIZE)))
    P2biases.append(NN_P2.init_bias(OUTPUT_SIZE))
    P3weights.append(NN_P3.init_weights((n_hidden, OUTPUT_SIZE)))
    P3biases.append(NN_P3.init_bias(OUTPUT_SIZE))

    # Forward propagation
    P2hat = NN_P2.forwardprop(X, weights, biases, num_layers)
    P3hat = NN_P3.forwardprop(X, weights, biases, num_layers)
    P2loss = tf.reduce_sum(tf.square(P2-P2hat))
    P3loss = tf.reduce_sum(tf.square(P3-P3hat))
    P2train = tf.train.AdamOptimizer(learning_rate=lr_rate).minimize(P2loss)
    P3train = tf.train.AdamOptimizer(learning_rate=lr_rate).minimize(P3loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Training
        for n in range(int(Nsample/n_batch)):
            input_X = np.reshape(sX[n*n_batch:(n+1)*n_batch], [n_batch, INPUT_SIZE])
            output_P2 = np.reshape(P2[n*n_batch:(n+1)*n_batch], [n_batch, OUTPUT_SIZE])
            output_P3 = np.reshape(P3[n*n_batch:(n+1)*n_batch], [n_batch, OUTPUT_SIZE])
            feed2 = {X: input_X, Y: output_P2}
            feed3 = {X: input_X, Y: output_P3}
            sess.run(P2train, feed_dict=feed2)
            sess.run(P3train, feed_dict=feed3)

        # Save
        save_weights(weights, biases, output_folder, weight_name_save, num_layers)

        # Test



if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Physics Net Training")
    parser.add_argument("--output_folder",type=str, default='D:/NBTP_Lab/Machine Learning/2D splitter/2D splitter_FCDNN/NN_parameter')
    parser.add_argument("--weight_name_save", type=str, default="")
    parser.add_argument("--n_batch", type=int, default=100)
    parser.add_argument("--lr_rate", type=float, default=1E-3)
    parser.add_argument("--num_layers", default=3)
    parser.add_argument("--n_hidden", default=200)

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
            'num_layers':int(dict['num_layers']),
            'n_hidden':int(dict['n_hidden'])
            }

    # main(**kwargs)
    Ratio_Optimization(**kwargs)