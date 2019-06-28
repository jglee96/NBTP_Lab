import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops


# As per Xaiver init, this should be 2/n(input), though many different initializations can be tried. 
def init_weights(shape, stddev=.1):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=stddev)
    return tf.Variable(weights)

def init_bias(shape, stddev=.1):
    """ Bias initialization """
    biases = tf.random_normal([shape], stddev=stddev)
    return tf.Variable(biases)

## FCDNN
def FCDNN_save_weights(weights, biases, output_folder, weight_name_save, num_layers):
    for i in range(0, num_layers+1):
        weight_i = weights[i].eval()
        np.savetxt(output_folder+weight_name_save+"/w_"+str(i)+".txt",weight_i, delimiter=',')
        bias_i = biases[i].eval()
        np.savetxt(output_folder+weight_name_save+"/b_"+str(i)+".txt",bias_i, delimiter=',')

def load_weights(output_folder, weight_load_name, num_layers):
    weights = []
    biases = []
    for i in range(0, num_layers+1):
        weight_i = np.loadtxt(output_folder+weight_load_name+"/w_"+str(i)+".txt", delimiter=',')
        w_i = tf.Variable(weight_i, dtype=tf.float32)
        weights.append(w_i)
        bias_i = np.loadtxt(output_folder+weight_load_name+"/b_"+str(i)+".txt", delimiter=',')
        b_i = tf.Variable(bias_i, dtype=tf.float32)
        biases.append(b_i)
    return weights, biases

def FCDNN_forwardprop(X, weights, biases, num_layers,):
    for i in range(0, num_layers):
        if i == 0:
            htemp = tf.nn.sigmoid(tf.add(tf.matmul(X, weights[i]), biases[i]))
        else:
            htemp = tf.nn.sigmoid(tf.add(tf.matmul(htemp, weights[i]), biases[i]))
    yval = tf.add(tf.matmul(htemp, weights[-1]), biases[-1])
    return yval

## ResNet
def ResNet_save_weights(weights, biases, output_folder, weight_name_save, RNnum_block):
    weight_i = weights[0].eval()
    np.savetxt(output_folder+weight_name_save+"/w_"+str(0)+".txt",weight_i, delimiter=',')
    bias_i = biases[0].eval()
    np.savetxt(output_folder+weight_name_save+"/b_"+str(0)+".txt",bias_i, delimiter=',')

    for i in range(0, RNnum_block):
        weight_i = weights[2*i+1].eval()
        np.savetxt(output_folder+weight_name_save+"/w_"+str(2*i+1)+".txt",weight_i, delimiter=',')
        bias_i = biases[2*i+1].eval()
        np.savetxt(output_folder+weight_name_save+"/b_"+str(2*i+1)+".txt",bias_i, delimiter=',')

        weight_i = weights[2*i+2].eval()
        np.savetxt(output_folder+weight_name_save+"/w_"+str(2*i+2)+".txt",weight_i, delimiter=',')
        bias_i = biases[2*i+2].eval()
        np.savetxt(output_folder+weight_name_save+"/b_"+str(2*i+2)+".txt",bias_i, delimiter=',')

    weight_i = weights[-1].eval()
    np.savetxt(output_folder+weight_name_save+"/w_"+str(2*RNnum_block+2)+".txt",weight_i, delimiter=',')
    bias_i = biases[-1].eval()
    np.savetxt(output_folder+weight_name_save+"/b_"+str(2*RNnum_block+2)+".txt",bias_i, delimiter=',')

def ResNet_forwardprop(X, weights, biases, RNnum_block):     
    htemp = tf.nn.sigmoid(tf.add(tf.matmul(X, weights[0]), biases[0]))
    for i in range(0, RNnum_block):
        htemp = tf.nn.sigmoid(tf.add(tf.matmul(htemp, weights[2*i+1]), biases[2*i+1]))
        htemp = tf.nn.sigmoid(tf.add(tf.add(tf.matmul(htemp, weights[2*i+2]), biases[2*i+2]), htemp))
    yval = tf.add(tf.matmul(htemp, weights[-1]), biases[-1])
    return yval

## DenseNet
def DenseNet_save_weights(weights, biases, output_folder, weight_name_save, Dense_list):
    weight_i = weights[0].eval()
    np.savetxt(output_folder+weight_name_save+"/w_"+str(0)+".txt",weight_i, delimiter=',')
    bias_i = biases[0].eval()
    np.savetxt(output_folder+weight_name_save+"/b_"+str(0)+".txt",bias_i, delimiter=',')

    pre = 0
    for n in Dense_list:
        for i in range(n):
            idx = pre + i + 1
            weight_i = weights[idx].eval()
            np.savetxt(output_folder+weight_name_save+"/w_"+str(idx)+".txt",weight_i, delimiter=',')
            bias_i = biases[idx].eval()
            np.savetxt(output_folder+weight_name_save+"/b_"+str(idx)+".txt",bias_i, delimiter=',')
        pre += n
    weight_i = weights[-1].eval()
    np.savetxt(output_folder+weight_name_save+"/w_"+str(pre+1)+".txt",weight_i, delimiter=',')
    bias_i = biases[-1].eval()
    np.savetxt(output_folder+weight_name_save+"/b_"+str(pre+1)+".txt",bias_i, delimiter=',')

def DenseNet_forwardprop(X, weights, biases, Dense_list):
    pre = 0
    htemp = tf.nn.sigmoid(tf.add(tf.matmul(X, weights[0]), biases[0]))
    for n in Dense_list:
        dense = []
        for i in range(n):
            dense.append(htemp)
            htemp = tf.nn.sigmoid(tf.add(tf.matmul(htemp, weights[pre+i+1]), biases[pre+i+1]))
            for net in dense:
                htemp = htemp + net
            htemp = tf.nn.sigmoid(htemp)
        pre += n
    yval = tf.add(tf.matmul(htemp, weights[-1]), biases[-1])
    return yval