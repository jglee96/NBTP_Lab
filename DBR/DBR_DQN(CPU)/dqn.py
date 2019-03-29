"""DQN Class

DQN(NIPS-2013)
"Playing Atari with Deep Reinforcement Learning"
https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

DQN(Nature-2015)
"Human-level control through deep reinforcement learning"
http://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
"""
import numpy as np
import tensorflow as tf
from datetime import datetime


class DQN:

    def __init__(self, session: tf.Session, input_size: int, output_size: int, name: str="main") -> None:
        """DQN Agent can

        1) Build network
        2) Predict Q_value given state
        3) Train parameters

        Args:
            session (tf.Session): Tensorflow session
            input_size (int): Input dimension
            output_size (int): Number of discrete actions
            name (str, optional): TF Graph will be built under this name scope
        """
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name
        log_name = './logs/dqn_mainDQN'+datetime.now().strftime("%Y%m%d%H%M")
        self.writer = tf.summary.FileWriter(log_name)

        self._build_network()

    def _build_network(self, l_rate=1E-7) -> None:
        """DQN Network architecture (simple MLP)

        Args:
            l_rate (float, optional): Learning rate
        """
        
        # hiddenlayer's number and length
        Hidden_Layer = np.array([round(self.output_size),round(self.output_size),round(self.output_size),round(self.output_size)])
        num_layer = Hidden_Layer.shape[0]
        
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(tf.float32, [None, self.input_size], name="input_x")
            net = self._X
                
            # more hidden layer not one
            for i in range(num_layer):
                #activation function is elu
                net = tf.layers.dense(net, Hidden_Layer[i], activation=tf.nn.elu, 
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), bias_initializer=tf.contrib.layers.variance_scaling_initializer())
                
            net = tf.layers.dense(net, self.output_size,
                                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), bias_initializer=tf.contrib.layers.variance_scaling_initializer())
            self._Qpred = net

            self._Y = tf.placeholder(tf.float32, shape=[None, self.output_size], name="output_y")
            self._loss = tf.losses.mean_squared_error(self._Y, self._Qpred)
            self._loss_hist = tf.summary.scalar('loss',self._loss)

            optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)
            self._train = optimizer.minimize(self._loss)

    def predict(self, state: np.ndarray) -> np.ndarray:
        """Returns Q(s, a)

        Args:
            state (np.ndarray): State array, shape (n, input_dim)

        Returns:
            np.ndarray: Q value array, shape (n, output_dim)
        """
        x = np.reshape(state, [-1,self.input_size])
        return self.session.run(self._Qpred, feed_dict={self._X: x})

    def update(self, x_stack: np.ndarray, y_stack: np.ndarray) -> list:
        """Performs updates on given X and y and returns a result

        Args:
            x_stack (np.ndarray): State array, shape (n, input_dim)
            y_stack (np.ndarray): Target Q array, shape (n, output_dim)

        Returns:
            list: First element is loss, second element is a result from train step
        """
        
        feed = {
            self._X: x_stack,
            self._Y: y_stack
        }
        return self.session.run([self._loss, self._train], feed)
    
    def updatewTboard(self, x_stack: np.ndarray, y_stack: np.ndarray) -> list:
        
        self.merged_summary = tf.summary.merge([self._loss_hist])
        
        feed = {
            self._X: x_stack,
            self._Y: y_stack
        }
        return self.session.run([self.merged_summary, self._loss, self._train], feed)
        
        
        
