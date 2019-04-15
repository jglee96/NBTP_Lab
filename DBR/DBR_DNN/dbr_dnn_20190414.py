import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import DBR
import winsound as ws
from datetime import datetime

# Real world environnment
Ngrid = 100
dx = 10
epsi = 12.25
eps0 = 1.

minwave = 400
maxwave = 1200
wavestep = 5
wavelength = np.array([np.arange(minwave,maxwave,wavestep)])
tarwave = 800

# Constants defining our neural network
INPUT_SIZE = Ngrid
OUTPUT_SIZE = len(wavelength[0])

Nsample = 100000
l_rate = 1E-4

SAVE_PATH = './result/'

# train set file name
statefilename = 'state_trainset02.txt'
Rfilename = 'R_trainset02.txt'

# Clear our computational graph
tf.reset_default_graph()

def beepsound():
    freq = 2000
    dur = 1000
    ws.Beep(freq,dur)

def main():
    with tf.Session() as sess:
        # hiddenlayer's number and length
        Hidden_Layer = np.array([round(3*OUTPUT_SIZE),round(3*OUTPUT_SIZE),round(3*OUTPUT_SIZE),round(3*OUTPUT_SIZE)])
        num_layer = Hidden_Layer.shape[0]
        
        X = tf.placeholder(tf.float32, [None, INPUT_SIZE], name="input_x")
        net = X
                
        # more hidden layer not one
        for i in range(num_layer):
            #activation function is elu
            net = tf.layers.dense(net, Hidden_Layer[i], activation=tf.nn.elu, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), bias_initializer=tf.contrib.layers.variance_scaling_initializer())
                
        net = tf.layers.dense(net, OUTPUT_SIZE, activation=tf.nn.elu, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), bias_initializer=tf.contrib.layers.variance_scaling_initializer())
        Rpred = net

        Y = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE], name="output_y")
        loss = tf.losses.mean_squared_error(Y, Rpred)
        loss_hist = tf.summary.scalar('loss', loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)
        train = optimizer.minimize(loss)

        log_name = './logs/'+datetime.now().strftime("%Y%m%d%H%M")
        net.writer = tf.summary.FileWriter(log_name)
        
        net.writer.add_graph(sess.graph)   
        sess.run(tf.global_variables_initializer()) # Initialize Tensorflow variables

        sX = np.loadtxt(statefilename).reshape(-1,INPUT_SIZE)
        sY = np.loadtxt(Rfilename).reshape(-1,OUTPUT_SIZE)

        for n in range(Nsample):
            feed = {X: np.reshape(sX[n],[-1,INPUT_SIZE]), Y: np.reshape(sY[n],[-1,OUTPUT_SIZE])}
            sess.run([loss,train], feed_dict=feed)

            if (n+1)%100 == 0:
                merged_summary = tf.summary.merge([loss_hist])
                summary = sess.run(merged_summary, feed_dict=feed)
                net.writer.add_summary(summary, global_step=n)
                print(n+1,'th trained')

        #test
        Tstate = np.random.randint(2,size=INPUT_SIZE)
        TR = DBR.calR(Tstate,Ngrid,wavelength,dx,epsi,eps0)
        NR = sess.run(Rpred, feed_dict={X: np.reshape(Tstate,[-1,INPUT_SIZE])})
        NR = np.reshape(NR,[OUTPUT_SIZE,-1])

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
        print(beepsound())


if __name__ == "__main__":
    main()