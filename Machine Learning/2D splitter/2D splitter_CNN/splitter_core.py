import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops


# As per Xaiver init, this should be 2/n(input), though many different initializations can be tried. 
def init_weights(shape, stddev=.5):
    """ Weight initialization """
    weights = tf.random_normal([shape], stddev=stddev)
    return tf.Variable(weights)

def init_bias(shape, stddev=.5):
    """ Bias initialization """
    biases = tf.random_normal([shape], stddev=stddev)
    return tf.Variable(biases)

def clipped_relu(x):
    net = tf.nn.leaky_relu(x)
    net = tf.minimum(x, 1)
    return net

## FCDNN
def save_weights(weights, biases, output_folder, weight_name_save, num_layers):
    for i in range(0, num_layers):
        weight_i = weights[i].eval()
        np.savetxt(output_folder+weight_name_save+"/w_"+str(i)+".txt",weight_i, delimiter=',')
        # bias_i = biases[i].eval()
        # np.savetxt(output_folder+weight_name_save+"/b_"+str(i)+".txt",bias_i, delimiter=',')

def load_weights(output_folder, weight_load_name, num_layers):
    weights = []
    biases = []
    for i in range(0, num_layers):
        weight_i = np.loadtxt(output_folder+weight_load_name+"/w_"+str(i)+".txt", delimiter=',')
        w_i = tf.Variable(weight_i, dtype=tf.float32)
        weights.append(w_i)
        bias_i = np.loadtxt(output_folder+weight_load_name+"/b_"+str(i)+".txt", delimiter=',')
        b_i = tf.Variable(bias_i, dtype=tf.float32)
        biases.append(b_i)
    return weights, biases

def forwardprop(X, weights, biases):

    net = tf.nn.conv2d(
        X, tf.reshape(weights[0], [3, 3, 1, 32]),
        strides=[1, 1, 1, 1], padding='SAME') # (?, 20, 20, 32)
    net = tf.nn.relu6(net)
    net = tf.nn.max_pool(
        net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # (?, 10, 10, 32)

    net = tf.nn.conv2d(
        net, tf.reshape(weights[1], [3, 3, 32, 64]),
        strides=[1, 1, 1, 1], padding='SAME') # (?, 10, 10, 64)
    net = tf.nn.relu6(net)
    net = tf.nn.max_pool(
        net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # (?, 5, 5, 64)
    
    net = tf.nn.conv2d(
        net, tf.reshape(weights[2], [3, 3, 64, 128]),
        strides=[1, 1, 1, 1], padding='SAME') # (?, 5, 5, 128)
    net = tf.nn.relu6(net)
    net = tf.nn.max_pool(
        net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # (?, 3, 3, 128)

    net = tf.reshape(net, [-1, 3*3*128])
    net = tf.nn.sigmoid(tf.matmul(net, tf.reshape(weights[3], [3*3*128, 256])))

    net = tf.nn.sigmoid(tf.matmul(net, tf.reshape(weights[4], [256, 128])))

    yhat = tf.matmul(net, tf.reshape(weights[5], [128, 100]))

    return yhat