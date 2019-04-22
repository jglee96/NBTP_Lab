import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import sliceDBR
import pandas as pd
from datetime import datetime
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.framework import arg_scope

# Real world environnment
Nslice = 7
epsi = 12.25
eps0 = 1.

minwave = 400
maxwave = 1200
wavestep = 5
wavelength = np.array([np.arange(minwave,maxwave,wavestep)])
tarwave = 800

# Constants defining our neural network
INPUT_SIZE = Nslice
OUTPUT_SIZE = len(wavelength[0])

Nfile = 8
Nsample = 10000
l_rate = 1E-3

# train set file name
FPATH = 'D:/NBTP_Lab/DBR/DBR_DNN_slice'
SAVE_PATH = FPATH +'/result/'
statefilename = '/trainset/state_trainset02'
Rfilename = '/trainset/R_trainset02'

# Batch Normalization function
def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                    scope=scope,
                    updates_collections=None,
                    decay=0.9,
                    center=True,
                    scale=True,
                    zero_debias_moving_mean=True) :
        return tf.cond(training,
                        lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                        lambda : batch_norm(inputs=x, is_training=training, reuse=True))

# Clear our computational graph
tf.reset_default_graph()

def main():
    # hiddenlayer's number and length
    num_layer = 4
    Phase = tf.placeholder(tf.bool)
    
    X = tf.placeholder(tf.float32, [None, INPUT_SIZE], name="input_x")
    # net = tf.contrib.layers.batch_norm(X, is_training=Phase)
    net = Batch_Normalization(X, training=Phase, scope='Input_X')
            
    # more hidden layer not one
    for i in range(num_layer):
        layer_name = 'FC'+str(i)
        with tf.name_scope(layer_name):
            #activation function is relu
            # net = tf.layers.dense(net, Hidden_Layer[i], activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), bias_initializer=tf.contrib.layers.variance_scaling_initializer())
            net = tf.contrib.layers.fully_connected(net, 250, activation_fn=tf.nn.relu, scope=layer_name)

    net = tf.layers.dense(net, OUTPUT_SIZE, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.contrib.layers.xavier_initializer())
    Rpred = net

    Y = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE], name="output_y")
    loss = tf.losses.mean_squared_error(Y, Rpred)
    loss_hist = tf.summary.scalar('loss', loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)
    train = optimizer.minimize(loss)

    log_name = FPATH+'/logs/'+datetime.now().strftime("%Y%m%d%H%M")
    net.writer = tf.summary.FileWriter(log_name)

    with tf.Session() as sess:        
        net.writer.add_graph(sess.graph)   
        sess.run(tf.global_variables_initializer()) # Initialize Tensorflow variables

        # Load Training Data
        Xarray = []
        Yarray = []
        for nf in range(Nfile):
            Xfname = FPATH+statefilename+'_'+str(nf)+'.txt'
            # Xtemp = np.loadtxt(Xfname)
            Xtemp = pd.read_csv(Xfname, names=['a'])
            Xtemp = Xtemp.values
            Xarray.append(Xtemp)

            Yfname = FPATH+Rfilename+'_'+str(nf)+'.txt'
            # Ytemp = np.loadtxt(Yfname)
            Ytemp = pd.read_csv(Yfname, names=['a'])
            Ytemp = Ytemp.values
            Yarray.append(Ytemp)
        
        Xcon = np.concatenate(Xarray)
        Ycon = np.concatenate(Yarray)

        sX = Xcon.reshape(-1,INPUT_SIZE)
        sY = Ycon.reshape(-1,OUTPUT_SIZE)

        batch_size = 32
        for n in range(int(Nsample*Nfile/batch_size)):
            feed = {X: np.reshape(sX[n*batch_size:(n+1)*batch_size],[batch_size,INPUT_SIZE]),
                    Y: np.reshape(sY[n*batch_size:(n+1)*batch_size],[batch_size,OUTPUT_SIZE]),
                    Phase: True}
            if (n+1)%100 == 0:
                merged_summary = tf.summary.merge([loss_hist])
                _, summary = sess.run([train, merged_summary], feed_dict=feed)
                net.writer.add_summary(summary, global_step=n)
                print(n+1,'th trained')
            else:
                sess.run(train, feed_dict=feed) 
        #test
        Tstate = np.random.randint(int(0.5*tarwave),size=Nslice)
        # Tstate = sX[0]
        TR = sliceDBR.calR(Tstate,Nslice,wavelength,epsi,eps0,True)
        # TR = sY[0]
        NR, Tloss = sess.run([Rpred,loss],feed_dict={X: np.reshape(Tstate,[-1,INPUT_SIZE]),
                                                    Y: np.reshape(TR,[-1,OUTPUT_SIZE]),
                                                    Phase: False})
        NR = np.reshape(NR,[OUTPUT_SIZE,-1])

        print('LOSS: ', Tloss)
        x = np.reshape(wavelength,wavelength.shape[1])
        plt.figure(2)
        plt.subplot(2,1,1)
        plt.plot(x,TR)
    
        plt.subplot(2,1,2)
        plt.plot(x,NR)

        fig2 = plt.gcf()
        plt.show()
        fig2_name = SAVE_PATH+datetime.now().strftime("%Y%m%d%H")+'_result model.png'
        fig2.savefig(fig2_name)

if __name__ == "__main__":
    main()