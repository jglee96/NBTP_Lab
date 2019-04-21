import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import sliceDBR
import winsound as ws
from datetime import datetime

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
l_rate = 1E-5

SAVE_PATH = './result/'

# train set file name
FPATH = 'D:/NBTP_Lab/DBR/DBR_DNN_slice'
statefilename = '/trainset/state_trainset02'
Rfilename = '/trainset/R_trainset02'

# Batch Noramlized Fully Connected Network
def dense_batch_relu(x, phase):
        h1 = tf.contrib.layers.fully_connected(x, 16*OUTPUT_SIZE, activation_fn=None)
        h2 = tf.contrib.layers.batch_norm(h1, center=True, scale=True, is_training=phase)
        
        return tf.nn.relu(h2)

# Clear our computational graph
tf.reset_default_graph()

def main():
    with tf.Session() as sess:
        # hiddenlayer's number and length
        num_layer = 4
        Phase = tf.placeholder(tf.bool)
        
        X = tf.placeholder(tf.float32, [None, INPUT_SIZE], name="input_x")
        net = tf.contrib.layers.batch_norm(X, center=True, scale=True, is_training=Phase)
                
        # more hidden layer not one
        for i in range(num_layer):
            #activation function is relu
            # net = tf.layers.dense(net, Hidden_Layer[i], activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), bias_initializer=tf.contrib.layers.variance_scaling_initializer())
            net = dense_batch_relu(net,Phase)

        Rpred = tf.layers.dense(net, OUTPUT_SIZE, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.contrib.layers.xavier_initializer())

        Y = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE], name="output_y")
        loss = tf.losses.mean_squared_error(Y, Rpred)
        loss_hist = tf.summary.scalar('loss', loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)
        train = optimizer.minimize(loss)

        log_name = './logs/'+datetime.now().strftime("%Y%m%d%H%M")
        net.writer = tf.summary.FileWriter(log_name)
        
        net.writer.add_graph(sess.graph)   
        sess.run(tf.global_variables_initializer()) # Initialize Tensorflow variables

        # Load Training Data
        Xarray = []
        Yarray = []
        for nf in range(Nfile):
            Xfname = FPATH+statefilename+'_'+str(nf)+'.txt'
            Xtemp = np.loadtxt(Xfname)
            Xarray.append(Xtemp)

            Yfname = FPATH+Rfilename+'_'+str(nf)+'.txt'
            Ytemp = np.loadtxt(Yfname)
            Yarray.append(Ytemp)
        
        Xcon = np.concatenate(Xarray)
        Ycon = np.concatenate(Yarray)

        sX = Xcon.reshape(-1,INPUT_SIZE)
        sY = Ycon.reshape(-1,OUTPUT_SIZE)

        batch_size = 64
        for n in range(int(Nsample*Nfile/batch_size)):
            feed = {X: np.reshape(sX[n*batch_size:(n+1)*batch_size],[batch_size,INPUT_SIZE]),
                    Y: np.reshape(sY[n*batch_size:(n+1)*batch_size],[batch_size,OUTPUT_SIZE]),
                    Phase: True}
            sess.run(train, feed_dict=feed)

            if (n+1)%100 == 0:
                merged_summary = tf.summary.merge([loss_hist])
                summary = sess.run(merged_summary, feed_dict=feed)
                net.writer.add_summary(summary, global_step=n)
                print(n+1,'th trained')

        #test
        Tstate = np.random.randint(int(0.5*tarwave),size=Nslice)
        TR = sliceDBR.calR(Tstate,Nslice,wavelength,epsi,eps0,True)
        NR, Tloss = sess.run([Rpred, loss], feed_dict={X: np.reshape(Tstate,[-1,INPUT_SIZE]),Y: np.reshape(TR,[-1,OUTPUT_SIZE]), Phase: False})
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