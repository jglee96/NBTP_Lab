import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import random


class DRQN():
    def __init__(self, input_size: int, h_size: int,
                 rnn_cell: tf.nn.rnn_cell.LSTMCell, learning_rate: float, scope: str):
        self.input_size = input_size
        self.net_name = scope
        self._build_network(rnn_cell, learning_rate, h_size)

    def _build_network(self, rnn_cell, learning_rate, h_size):
        with tf.variable_scope(self.net_name):
            self.X = tf.placeholder(shape=[None, self.input_size], dtype=tf.float32)
            net = self.X
            self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])
            self.trainLength = tf.placeholder(dtype=tf.int32)

            Hidden_Layer = [250, 250, 250, 250]
            num_layer = len(Hidden_Layer)
            
            for num_neurons in Hidden_Layer:
                net = tf.contrib.layers.fully_connected(
                    inputs=net, num_outputs=num_neurons, activation_fn=tf.nn.relu)
            
            self.netFlat = tf.reshape(slim.flatten(net), [self.batch_size, self.trainLength, h_size])
            self.state_in = rnn_cell.zero_state(
                batch_size=self.batch_size, dtype=tf.float32),
            self.rnn, self.rnn_state = tf.nn.dynamic_rnn(
                cell=rnn_cell, inputs=self.netFlat,
                initial_state=self.state_in[0],
                dtype=tf.float32, scope=self.net_name+'_rnn')

            self.rnn = tf.reshape(self.rnn, shape=[-1, h_size])

            self.streamA, self.streamV = tf.split(
                value=self.rnn, num_or_size_splits=2, axis=1)
            self.AW = tf.Variable(tf.random_normal([int(h_size/2), 4]))
            self.VW = tf.Variable(tf.random_normal([int(h_size/2), 1]))
            self.Advantage = tf.matmul(self.streamA, self.AW)
            self.Value = tf.matmul(self.streamV, self.VW)

            self.salience = tf.gradients(self.Advantage, self.X)
            self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(
                self.Advantage, reduction_indices=1, keep_dims=True))
            self.predict = tf.argmax(self.Qout, 1)

            self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(indices=self.actions, depth=4, dtype=tf.float32)

            self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), reduction_indices=1)

            self.td_error = tf.square(self.targetQ - self.Q)

            self.maskA = tf.zeros(shape=[self.batch_size, tf.cast(self.trainLength/2, tf.int32)])
            self.maskB = tf.ones(shape=[self.batch_size, tf.cast(self.trainLength/2, tf.int32)])
            self.mask = tf.concat(values=[self.maskA, self.maskB], axis=1)
            self.mask = tf.reshape(self.mask, [-1])
            self.loss = tf.reduce_mean(self.td_error * self.mask)
            self.loss_hist = tf.summary.scalar('loss', self.loss)
            self.merged_summary = tf.summary.merge([self.loss_hist])

            self.train = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.updateModel = self.train.minimize(self.loss)


class Experience_Buffer():
    def __init__(self, buffer_size=1E4):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1+len(self.buffer)) - self.buffer_size] = []
        self.buffer.append(experience)

    def sample(self, batch_size, trace_length):
        sampled_episodes = random.sample(self.buffer, batch_size)
        sampledTraces = []
        for episode in sampled_episodes:
            point = random.randint(0, len(episode) - trace_length)
            sampledTraces.append(episode[point:point + trace_length])
        sampledTraces = np.array(sampledTraces)

        return np.reshape(sampledTraces, [batch_size * trace_length, 5])