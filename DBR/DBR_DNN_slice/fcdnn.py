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
                     name: str, beta1: float, beta2: float):
                self.session = session
                self.input_size = input_size
                self.output_size = output_size
                self.net_name = name
                # log_name = FPATH+'/logs/FollowScatterNet/{:.1f}_{:.1f}'.format(beta1, beta2)
                log_name = FPATH+'/logs/'+datetime.now().strftime("%Y%m%d%H%M")
                self.writer = tf.summary.FileWriter(log_name)
                self._build_network(
                        batch_size, num_layer, num_neuron,
                        learning_rate, beta1, beta2)

        def _build_network(self, batch_size,
                           num_layer, num_neuron, learning_rate, beta1, beta2):
                # hiddenlayer's number and length
                self.Phase = tf.placeholder(tf.bool)

                self.X = tf.placeholder(tf.float32, [None, self.input_size],
                                        name="input_x")
                # net = self.Batch_Normalization(
                #         x=self.X, training=self.Phase, scope='BN_input')
                net = self.X
                # net = tf.reshape(self.X, [-1, 1, self.input_size, 1])

                # # Conv. net
                # for i in range(8):
                #         conv_name = 'Resnet' + str(i)
                #         with tf.name_scope(conv_name):
                #                 # Conv
                #                 conv1 = tf.layers.conv2d(
                #                         inputs=net, filters=1,
                #                         kernel_size=[1,3], padding="SAME")
                #                 # Dropout
                #                 dropout1 = tf.layers.dropout(inputs=tf.nn.relu(conv1), rate=0.5)
                #                 # Conv
                #                 conv2 = tf.layers.conv2d(
                #                         inputs=dropout1, filters=1,
                #                         kernel_size=[1,3], padding="SAME")
                #                 # Dropout
                #                 dropout2 = tf.layers.dropout(inputs=tf.nn.relu(conv2), rate=0.5)
                #                 net = tf.nn.relu(dropout2 + net)
                
                # net = tf.reshape(net, [-1, self.input_size])

                # more hidden layer not one
                for i in range(num_layer):
                        layer_name = 'FC' + str(i)
                        with tf.name_scope(layer_name):
                                # Fully Connected
                                net = tf.contrib.layers.fully_connected(
                                        net, num_neuron,
                                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                                        biases_initializer=tf.contrib.layers.xavier_initializer(),
                                        activation_fn=tf.nn.relu,
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
                # self.loss = tf.losses.mean_squared_error(self.Y, self.Rpred)
                # self.loss = tf.reduce_mean(tf.square(self.Y-self.Rpred))
                self.loss = tf.reduce_sum(tf.square(self.Y-self.Rpred))
                # self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y, logits=self.Rpred))
                self.loss_hist = tf.summary.scalar('loss', self.loss)

                optimizer = tf.contrib.optimizer_v2.AdamOptimizer(learning_rate=learning_rate,
                                                   beta1=beta1, beta2=beta2)
                self.train = optimizer.minimize(self.loss)

                # Batch Normalization function
        def Batch_Normalization(self, x, training, scope):
                with arg_scope([
                        batch_norm],
                        scope=scope,
                        updates_collections=None,
                        decay=0.9,
                        center=True,
                        scale=True,
                        zero_debias_moving_mean=True):
                        return tf.cond(
                                training,
                                lambda: batch_norm(
                                        inputs=x, is_training=training,
                                        reuse=None),
                                lambda: batch_norm(
                                        inputs=x, is_training=training,
                                        reuse=True))

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