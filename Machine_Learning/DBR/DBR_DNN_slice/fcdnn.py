import tensorflow as tf
from datetime import datetime
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.framework import arg_scope

FPATH = 'D:/NBTP_Lab/DBR/DBR_DNN_slice'


class FC_DNN:
        # Variable: batch_size: bs, num_layer: nl, num_neuron: nn
        def __init__(self, session: tf.Session,
                     input_size: int, output_size: int, batch_size: int,
                     num_layer: int, num_neuron: int, learning_rate: float,
                     name: str):
                self.session = session
                self.input_size = input_size
                self.output_size = output_size
                self.net_name = name
                # log_name = FPATH+'/logs/FollowScatterNet/{:.1f}_{:.1f}'.format(beta1, beta2)
                log_name = FPATH+'/logs/'+datetime.now().strftime("%Y%m%d%H%M")
                self.writer = tf.summary.FileWriter(log_name)
                self._build_network(
                        batch_size, num_layer, num_neuron,
                        learning_rate)

        def _build_network(self, batch_size,
                           num_layer, num_neuron, learning_rate):
                # hiddenlayer's number and length
                self.Phase = tf.placeholder(tf.bool)

                self.X = tf.placeholder(tf.float32, [None, self.input_size],
                                        name="input_x")
                net = self.X

                # more hidden layer not one
                for i in range(num_layer):
                        layer_name = 'FC' + str(i)
                        with tf.name_scope(layer_name):
                                # Fully Connected
                                net = tf.contrib.layers.fully_connected(
                                        net, num_neuron,
                                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                                        biases_initializer=tf.contrib.layers.xavier_initializer(),
                                        activation_fn=tf.nn.sigmoid,
                                        scope=layer_name)

                net = tf.contrib.layers.fully_connected(
                        net, self.output_size, activation_fn=None,
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        biases_initializer=tf.contrib.layers.xavier_initializer()
                        )
                self.Rpred = net

                self.Y = tf.placeholder(
                        tf.float32, shape=[None, self.output_size],
                        name="output_y")
                self.loss = tf.reduce_sum(tf.divide(tf.square(self.Y-self.Rpred), 2))
                self.loss_hist = tf.summary.scalar('loss', self.loss)

                optimizer = tf.contrib.optimizer_v2.AdamOptimizer(learning_rate=learning_rate)
                self.train = optimizer.minimize(self.loss)

        def update_train(self, x_stack, y_stack, phase):
                feed = {
                        self.X: x_stack,
                        self.Y: y_stack,
                        self.Phase: phase
                }
                return self.session.run(self.train, feed)

        def update_loss(self, x_stack, y_stack, phase):
                feed = {
                        self.X: x_stack,
                        self.Y: y_stack,
                        self.Phase: phase
                }
                return self.session.run(self.loss, feed)

        def update_tensorboard(self, x_stack, y_stack, phase):
                self.merged_summary = tf.summary.merge([self.loss_hist])
                feed = {
                        self.X: x_stack,
                        self.Y: y_stack,
                        self.Phase: phase
                }
                return self.session.run(self.merged_summary, feed)

        def Test_paly(self, x_stack, y_stack, phase):

                feed = {
                        self.X: x_stack,
                        self.Y: y_stack,
                        self.Phase: phase
                }
                return self.session.run(self.Rpred, feed)