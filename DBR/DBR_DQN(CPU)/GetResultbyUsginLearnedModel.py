# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 18:03:13 2019

@author: Lee Jong Geon
"""

import tensorflow as tf
import numpy as np
import matplotlib as plt
from datetime import datetime
import os
import DBR

MODEL_PATH = './model/'

# Real world environnment
Ngrid = 200
N1 = Ngrid
N2 = Ngrid
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
OUTPUT_SIZE = Ngrid
MAX_EPISODES = 100

with tf.Session() as sess:
    # Load meta graph and restore weights
    load_name = 'dqn_2019-03-22-01.meta'
    load_path = os.path.join(MODEL_PATH,load_name)
#    saver = tf.train.import_meta_graph(load_path)
    saver = tf.train.Saver()
    saver.restore(sess,load_path)
    
    input_x = sess.graph.get_tensor_by_name('input_x')
    output_y = sess.graph.get_tensor_by_name('output_y')
    W1 = sess.graph.get_tensor_by_name('W1')
    W2 = sess.graph.get_tensor_by_name('W2')
    
    max_state = []
    max_Qfac = 0
    maxR = []
    
    for episode in range(MAX_EPISODES):
        state = np.ones((1,Ngrid))
        
        for iteridx in range(Ngrid):
            R = DBR.calR(state,Ngrid,wavelength,dx,epsi,eps0)
            Qfac = DBR.calQfac(R,wavelength,tarwave)
            
            # Save maximum Q factor case
            if Qfac > max_Qfac:
                max_Qfac = Qfac
                max_state = state.copy()
                maxR = R.copy()
            
            action = sess.run(output_y, feed_dict={input_x: state})
            aidx = action.argmax()            
            state = DBR.step(state,aidx)
    
    print("*****Selected Optimized model*****")
    print("Q factor = ",Qfac)
    
    x = np.reshape(wavelength,wavelength.shape[1])
    plt.figure(2)
    plt.subplot(2,1,1)
    plt.plot(x,R)
    
    plt.subplot(2,1,2)
    plt.imshow(state,cmap='gray')    
    
    fig2 = plt.gcf()
    plt.show()
    fig2_name = datetime.now().strftime("%Y-%m-%d-%H")+'_result model.png'
    fig2.savefig(fig2_name)
    
    
            
            
            
    

