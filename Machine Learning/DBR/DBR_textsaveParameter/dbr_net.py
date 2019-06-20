import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pixelDBR
import argparse
from dbr_core import *

N_pixel = 100
dx = 5
nh = 2.6811  # TiO2 at 400 nm (Siefke)
nl = 1.4701  # SiO2 at 400 nm (Malitson)

minwave = 200
maxwave = 800
wavestep = 5
wavelength = np.array([np.arange(minwave, maxwave, wavestep)])
tarwave = 500
bandwidth = 40

INPUT_SIZE = N_pixel
OUTPUT_SIZE = wavelength.shape[1]

Nfile = 10
Nsample = 10000
PATH = 'D:/NBTP_Lab/Machine Learning/DBR/DBR_gaussianloglikelihood_lossfunction'
TRAIN_PATH = PATH + '/trainset/01'

def getData():
    # Load Training Data
    print("========      Load Data     ========")
    Xarray = []
    Yarray = []
    for nf in range(Nfile):
        sname = TRAIN_PATH + '/state_' + str(nf) + '.csv'
        Xtemp = pd.read_csv(sname, header=None)
        Xtemp = Xtemp.values
        Xarray.append(Xtemp)

        Rname = TRAIN_PATH + '/R_' + str(nf) + '.csv'
        Ytemp = pd.read_csv(Rname, header=None)
        Ytemp = Ytemp.values
        Yarray.append(Ytemp)

    sX = np.concatenate(Xarray)
    sY = np.concatenate(Yarray)

    return sX, sY


def Ratio_Optimization(output_folder, weight_name_save, n_batch, lr_rate, num_layers, n_hidden):
    init_list_rand = tf.constant(np.random.randint(2, size=(1, N_pixel)), dtype=tf.float32)
    X = tf.get_variable(name='b', initializer=init_list_rand)
    weights, biases = load_weights(output_folder, weight_name_save, num_layers)
    

def main(output_folder, weight_name_save, n_batch, lr_rate, num_layers, n_hidden):
    # Load training data
    sX, sY = getData()

    ## Define NN ##
    X = tf.placeholder(tf.float32, [None, INPUT_SIZE], name="input_x")
    Y = tf.placeholder(tf.float32, [None, OUTPUT_SIZE], name="output_y")
    weights = []
    biases = []

    for i in range(0,num_layers):
        if i ==0:
            weights.append(init_weights((INPUT_SIZE, n_hidden)))
        else:
            weights.append(init_weights((n_hidden, n_hidden)))
        biases.append(init_bias(n_hidden))
    weights.append(init_weights((n_hidden, OUTPUT_SIZE)))
    biases.append(init_bias(OUTPUT_SIZE))

    # Forward propagation
    Yhat = forwardprop(X, weights, biases, num_layers)
    loss = tf.reduce_sum(tf.square(Y-Yhat))
    train = tf.train.AdamOptimizer(learning_rate=lr_rate).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Training
        for n in range(int(Nsample*Nfile/n_batch)):
            input_X = np.reshape(sX[n*n_batch:(n+1)*n_batch], [n_batch, INPUT_SIZE])
            output_Y = np.reshape(sY[n*n_batch:(n+1)*n_batch], [n_batch, OUTPUT_SIZE])
            feed = {X: input_X, Y: output_Y}
            sess.run(train, feed_dict=feed)

        # Save
        save_weights(weights, biases, output_folder, weight_name_save, num_layers)

        # Test
        Tstate = np.random.randint(2, size=N_pixel)
        TR = pixelDBR.calR(Tstate, dx, N_pixel, wavelength, nh, nl)
        tX = np.reshape(Tstate, [-1, INPUT_SIZE])
        tY = np.reshape(TR, [-1, OUTPUT_SIZE])
        NR = sess.run(Yhat, feed_dict={X: tX})
        Tloss = sess.run(loss, feed_dict={X: tX, Y: tY})
    # sess.close()

    print("LOSS: ", Tloss)
    x = np.reshape(wavelength, wavelength.shape[1])
    TR = np.reshape(TR, wavelength.shape[1])
    NR = np.reshape(NR, wavelength.shape[1])
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(x, TR)

    plt.subplot(2, 1, 2)
    plt.plot(x, NR)
    plt.show()



if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Physics Net Training")
    parser.add_argument("--output_folder",type=str, default='D:/NBTP_Lab/Machine Learning/DBR/DBR_textsaveParameter/NN_parameter')
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

    main(**kwargs)