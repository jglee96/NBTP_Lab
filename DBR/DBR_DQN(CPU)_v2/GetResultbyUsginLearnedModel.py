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
MODEL_NAME = 'dqn_2019041100.ckpt'
META_NAME = 'dqn_2019041100.ckpt.meta'

# Real world environnment
Ngrid = 100
dx = 10
epsi = 12.25
eps0 = 1.

minwave = 400
maxwave = 1200
wavestep = 10
wavelength = np.array([np.arange(minwave,maxwave,wavestep)])
tarwave = 800

# Constants defining our neural network
INPUT_SIZE = Ngrid
OUTPUT_SIZE = Ngrid+1

tf.reset_default_graph()
    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
#    input_x = sess.graph.get_tensor_by_name('input_x')
#    output_y = sess.graph.get_tensor_by_name('output_y')
#    W1 = sess.graph.get_tensor_by_name('W1')
#    W2 = sess.graph.get_tensor_by_name('W2')    
    X = tf.placeholder(tf.float32, [None, INPUT_SIZE], name="input_x")
    Y = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE], name="output_y")
    
    W0 = tf.get_variable("main/dense/kernel", shape=(INPUT_SIZE,OUTPUT_SIZE))
    b0 = tf.get_variable("main/dense/bias", shape=(OUTPUT_SIZE))
    L0 = tf.nn.elu(tf.matmul(X,W0)+b0)
    
    W1 = tf.get_variable("main/dense_1/kernel", shape=(OUTPUT_SIZE,OUTPUT_SIZE))
    b1 = tf.get_variable("main/dense_1/bias", shape=(OUTPUT_SIZE))
    L1 = tf.nn.elu(tf.matmul(L0,W1)+b1)
    
    W2 = tf.get_variable("main/dense_2/kernel", shape=(OUTPUT_SIZE,OUTPUT_SIZE))
    b2 = tf.get_variable("main/dense_2/bias", shape=(OUTPUT_SIZE))
    L2 = tf.nn.elu(tf.matmul(L1,W2)+b2)
    
    W3 = tf.get_variable("main/dense_3/kernel", shape=(OUTPUT_SIZE,OUTPUT_SIZE))
    b3 = tf.get_variable("main/dense_3/bias", shape=(OUTPUT_SIZE))
    L3 = tf.nn.elu(tf.matmul(L2,W3)+b3)
    
    W4 = tf.get_variable("main/dense_4/kernel", shape=(OUTPUT_SIZE,OUTPUT_SIZE))
    b4 = tf.get_variable("main/dense_4/bias", shape=(OUTPUT_SIZE))
    L4 = tf.nn.elu(tf.matmul(L3,W4)+b4)
    
    Y = L4
    
    # Load meta graph and restore weights
    load_path = os.path.join(MODEL_PATH,MODEL_NAME)
#    meta_path = os.path.join(MODEL_PATH,META_NAME)
    saver = tf.train.Saver()
#    saver = tf.train.import_meta_graph(meta_path,clear_devices=True)
    saver.restore(sess,load_path)
    print("*****Saved model is succefuly loaded*****")
    
    max_state = []
    max_Qfac = 0
    maxR = []
    state = np.ones(Ngrid)
#    state = np.random.randint(2, size=Ngrid)
        
    N = 10000    
    for iteridx in range(N):
        e = 1. / ((iteridx / 100) + 1)
        R = DBR.calR(state,Ngrid,wavelength,dx,epsi,eps0)
        Qfac = DBR.calQfac(R,wavelength,tarwave)
        rawreward = DBR.reward(Ngrid,wavelength,R,tarwave)
            
        # Save maximum Q factor case
        if Qfac > max_Qfac:
            max_Qfac = Qfac
            max_state = state.copy()
            maxR = R.copy()
            
#            action = sess.run(output_y, feed_dict={input_x: state})
        
        if np.random.rand(1) < e:
            aidx = np.random.randint(OUTPUT_SIZE)
        else:
            # Choose an action by greedily from the Q-network
            feed_state = np.reshape(state, [-1,INPUT_SIZE])
            action = sess.run(Y, feed_dict={X: feed_state})
            aidx = action.argmax()            
                
        state, _ = DBR.step(state,aidx,Ngrid)
        print("Iternation: {}({:.2f}%), MAX Q fac.: {:.2f}, aidx: {}, reward: {:.2f}".format(iteridx,100*(iteridx+1)/N,max_Qfac,aidx,rawreward))
    
    print("*****Selected Optimized model*****")
    print("Q factor = {:.2f}".format(max_Qfac))
    
    x = np.reshape(wavelength,wavelength.shape[1])
    plt.figure(2)
    plt.subplot(2,1,1)
    plt.plot(x,maxR)
    
    lx = np.arange(Ngrid)
    plt.subplot(2,1,2)
    plt.bar(lx,max_state,width=1,color='blue')         
    
    fig2 = plt.gcf()
    plt.show()
    fig2_name = datetime.now().strftime("%Y%m%d%H")+'_result model.png'
    fig2.savefig(fig2_name)
    
    
            
            
            
    

