import tensorflow as tf
import numpy as np
from datetime import datetime
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.framework import arg_scope

FPATH = 'D:/NBTP_Lab/DBR/DBR_MultiNet'


class MNN:
        # Variable: batch_size: bs, num_layer: nl, num_neuron: nn
        def __init__(
                self, input_size: int, output_size: int,
                lbound: float, ubound: float, nl: float, nh: float, learning_rate: float, name: str):
                self.input_size = input_size
                self.output_size = output_size
                self.net_name = name
                log_name = FPATH+'/logs/'+datetime.now().strftime("%Y%m%d%H")+'/'+self.net_name+datetime.now().strftime("%Y%m%d%H%M")
                self.writer = tf.summary.FileWriter(log_name)
                self._build_network(lbound, ubound, nl, nh, learning_rate)

        def _build_network(self, lbound, ubound, nl, nh, learning_rate):
                init_list_rand = tf.constant(np.random.rand(1, self.input_size)*(ubound - lbound) + lbound, dtype=tf.float32)
                self.X = tf.Variable(initial_value=init_list_rand)
                self.Y = tf.placeholder(tf.float32, shape=[None, 1])
                
                self.X = tf.maximum(self.X, lbound)
                self.X = tf.minimum(self.X, ubound)

                n_list = np.empty(self.input_size)
                for i in range(self.input_size):
                        if (i % 2) == 0:
                                n_list[i] = nh
                        else:
                                n_list[i] = nl
                n_list = np.concatenate(([1], n_list, [1]))
                n_list = np.reshape(n_list, (1, self.input_size + 2))
                kz_list = 2 * np.pi * n_list

                n_list = tf.constant(n_list, dtype=tf.float32)
                kz_list = tf.constant(kz_list, dtype=tf.float32)

                d_list = tf.concat([tf.zeros([1, 1]), x, tf.zeros([1, 1])], 1)
                delta = tf.multiply(kz_list, d_list)  # (1, n_list)

                eye = [[1.0, 0.0], [0.0, 1.0]]
                Btot = tf.Variable(initial_value=tf.complex(eye, 0.0), trainable=False)
                b00, b01, b10, b11, Bt = [], [], [], [], []
                for i in range(self.input_size - 1):
                        gather_delta = tf.gather_nd(delta, [0, i])
                        gather_P = (tf.gather_nd(n_list, [0, i + 1]) / tf.gather_nd(n_list, [0, i]))
                        b00.append(tf.Variable(
                                initial_value=tf.complex((1 + gather_P) * tf.cos(-1 * gather_delta), (1 + gather_P) * tf.sin(-1 * gather_delta)),
                                trainable=False))
                        b01.append(tf.Variable(
                                initial_value=tf.complex((1 - gather_P) * tf.cos(+1 * gather_delta), (1 - gather_P) * tf.sin(+1 * gather_delta)),
                                trainable=False))
                        b10.append(tf.Variable(
                                initial_value=tf.complex((1 - gather_P) * tf.cos(-1 * gather_delta), (1 - gather_P) * tf.sin(-1 * gather_delta)),
                                trainable=False))
                        b11.append(tf.Variable(
                                initial_value=tf.complex((1 + gather_P) * tf.cos(+1 * gather_delta), (1 + gather_P) * tf.sin(+1 * gather_delta)),\
                                trainable=False))
                        Bt.append(tf.Variable(
                                initial_value=tf.complex(0.5, 0.0) * [[b00[i], b01[i]], [b10[i], b11[i]]],
                                trainable=False))
                        Btot = tf.matmul(Btot, Bt[i])
                self.R = tf.square(tf.gather_nd(Btot, [1, 0]) / tf.gather_nd(Btot, [0, 0]))

                # self.loss = tf.losses.mean_squared_error(self.Y, self.Rpred)
                # self.loss = tf.reduce_mean(tf.square(self.Y-self.Rpred))
                self.loss = tf.reduce_sum(tf.square(self.Y-self.Rpred))
                # self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y, logits=self.Rpred))
                self.loss_hist = tf.summary.scalar('loss', self.loss)

                optimizer = tf.contrib.optimizer_v2.AdamOptimizer(
                        learning_rate=learning_rate, beta2=0.7)
                self.train = optimizer.minimize(self.loss)

                # Batch Normalization function

        def update_train(self, sess, x_stack, y_stack, phase):
                feed = {
                        self.X: x_stack,
                        self.Y: y_stack,
                        self.Phase: phase
                }
                return sess.run(self.train, feed)

        def update_loss(self, sess, x_stack, y_stack, phase):
                feed = {
                        self.X: x_stack,
                        self.Y: y_stack,
                        self.Phase: phase
                }
                return sess.run(self.loss, feed)

        def update_tensorboard(self, sess, x_stack, y_stack, phase):
                self.merged_summary = tf.summary.merge([self.loss_hist])
                feed = {
                        self.X: x_stack,
                        self.Y: y_stack,
                        self.Phase: phase
                }
                return sess.run(self.merged_summary, feed)

        def Test_paly(self, sess, x_stack, phase):

                feed = {
                        self.X: x_stack,
                        self.Phase: phase
                }
                return sess.run(self.Rpred, feed)