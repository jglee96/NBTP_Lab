import tensorflow as tf
from datetime import datetime
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.framework import arg_scope

FPATH = 'D:/NBTP_Lab/DBR/DBR_DNN_slice'

class FC_DNN:
        # Variable: batch_size: bs, num_layer: nl, num_neuron: nn
        def __init__(self, session: tf.Session, input_size: int, output_size: int, batch_size: int, num_layer: int, num_neuron: int, learning_rate: float, name: str):
                self.session = session
                self.input_size = input_size
                self.output_size = output_size
                self.net_name = name
                # log_name = FPATH+'/logs/TMM/TMM_{}_{}_{}_{:.5f}'.format(batch_size, num_layer, num_neuron, learning_rate)
                log_name = FPATH+'/logs/'+datetime.now().strftime("%Y%m%d%H%M")
                self.writer = tf.summary.FileWriter(log_name)
                self._build_network(batch_size,num_layer,num_neuron, learning_rate)   

        def _build_network(self, batch_size, num_layer, num_neuron, learning_rate):
                # hiddenlayer's number and length
                self.Phase = tf.placeholder(tf.bool)

                self.X = tf.placeholder(tf.float32, [None, self.input_size], name="input_x")
                net = self.Batch_Normalization(x=self.X, training=self.Phase, scope='BN_input')
                # net = self.X
                        
                # more hidden layer not one
                for i in range(num_layer):
                        layer_name = 'FC'+str(i)
                        with tf.name_scope(layer_name):
                                #activation function is relu
                                net = tf.contrib.layers.fully_connected(net, num_neuron, activation_fn=tf.nn.relu, scope=layer_name)

                net = tf.contrib.layers.fully_connected(net, self.output_size, activation_fn=tf.nn.relu)
                self.Rpred = net

                self.Y = tf.placeholder(tf.float32, shape=[None, self.output_size], name="output_y")
                # self.loss = tf.losses.mean_squared_error(self.Y, self.Rpred)
                # self.loss = tf.reduce_mean(tf.square(self.Y-self.Rpred))
                self.loss = tf.reduce_sum(tf.square(self.Y-self.Rpred))
                # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.Rpred))
                self.loss_hist = tf.summary.scalar('loss', self.loss)

                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                self.train = optimizer.minimize(self.loss)

                # Batch Normalization function
        def Batch_Normalization(self, x, training, scope):
                with arg_scope([batch_norm],
                        scope=scope,
                        updates_collections=None,
                        decay=0.9,
                        center=True,
                        scale=True,
                        zero_debias_moving_mean=True) :
                        return tf.cond(training,
                                lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                                lambda : batch_norm(inputs=x, is_training=training, reuse=True))
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