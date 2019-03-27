# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 18:03:13 2019

@author: Lee Jong Geon
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import DBR

MODEL_PATH = './model/'
MODEL_NAME = 'dqn_2019032616.ckpt'

# Real world environnment
Ngrid = 150
dx = 10
epsi = 12.25
eps0 = 1.

minwave = 500
maxwave = 1100
wavestep = 25
wavelength = np.array([np.arange(minwave,maxwave,wavestep)])
tarwave = 800

# Constants defining our neural network
INPUT_SIZE = Ngrid
OUTPUT_SIZE = 2*Ngrid

tf.reset_default_graph()
    
with tf.Session() as sess:
    
#    input_x = sess.graph.get_tensor_by_name('input_x')
#    output_y = sess.graph.get_tensor_by_name('output_y')
#    W1 = sess.graph.get_tensor_by_name('W1')
#    W2 = sess.graph.get_tensor_by_name('W2')    
    X = tf.placeholder(tf.float32, [None, INPUT_SIZE], name="input_x")
    Y = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE], name="output_y")
    
    W1 = tf.get_variable("target/dense_1/kernel", shape=(Ngrid,2*Ngrid))
    b1 = tf.get_variable("target/dense_1/bias", shape=(2*Ngrid))
    L1 = tf.nn.elu(tf.matmul(X,W1)+b1)
    
    W2 = tf.get_variable("target/dense_2/kernel", shape=(2*Ngrid,2*Ngrid))
    b2 = tf.get_variable("target/dense_2/bias", shape=(2*Ngrid))
    L2 = tf.nn.elu(tf.matmul(L1,W2)+b2)
    
    W3 = tf.get_variable("target/dense_3/kernel", shape=(2*Ngrid,Ngrid))
    b3 = tf.get_variable("target/dense_3/bias", shape=(Ngrid))
    L3 = tf.nn.elu(tf.matmul(L2,W3)+b3)
    
    W4 = tf.get_variable("target/dense_4/kernel", shape=(Ngrid,2*Ngrid))
    b4 = tf.get_variable("target/dense_4/bias", shape=(2*Ngrid))
    L4 = tf.nn.elu(tf.matmul(L3,W4)+b4)
    
    Y = L4

    saver = tf.train.Saver()
    
    # Load meta graph and restore weights
    load_path = os.path.join(MODEL_PATH,MODEL_NAME)
#    saver = tf.train.import_meta_graph(load_path)
    saver.restore(sess,load_path)
    print("*****Saved model is succefuly loaded*****")
    
    max_state = []
    max_Qfac = 0
    maxR = []
    state = np.ones((1,Ngrid))
        
    for iteridx in range(Ngrid):
        R = DBR.calR(state,Ngrid,wavelength,dx,epsi,eps0)
        Qfac = DBR.calQfac(R,wavelength,tarwave)
            
        # Save maximum Q factor case
        if Qfac > max_Qfac:
            max_Qfac = Qfac
            max_state = state.copy()
            maxR = R.copy()
            
#            action = sess.run(output_y, feed_dict={input_x: state})
        action = sess.run(Y, feed_dict={X: state})
        aidx = action.argmax()            
        state = DBR.step(state,aidx,Ngrid)
        print("Iternation: {}({:.2f}%), MAX Q fac.: {:.2f}".format(iteridx,100*(iteridx+1)/Ngrid,max_Qfac))
    
    print("*****Selected Optimized model*****")
    print("Q factor = {:.2f}".format(max_Qfac))
    
    x = np.reshape(wavelength,wavelength.shape[1])
    plt.figure(2)
    plt.subplot(2,1,1)
    plt.plot(x,maxR)
    
    plt.subplot(2,1,2)
    plt.imshow(max_state,cmap='gray')    
    
    fig2 = plt.gcf()
    plt.show()
    fig2_name = datetime.now().strftime("%Y%m%d%H")+'_result model.png'
    fig2.savefig(fig2_name)
    
    
            
            
            
    

