import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import DBN

# Real world environnment
Ngrid = 100
dx = 10
dbr = np.zeros(shape=[1,Ngrid],dtype=np.int8)

minwave = 400
maxwave = 1200
wavestep = 10
wavelength = np.arrange(minwave,maxwave,wavestep)
tarwave = 800

# Input and output size based on the Env
input_size = Ngrid
output_size = Ngrid
learning_rate = 0.1

# These lines establish the feed-forward part of the network used to
# choose actions
X = tf.placeholder(shape=[1,input_size],dtype=tf.int8) # state input
W = tf.Variable(tf.random_uniform([input_size,output_size],0,0.01)) # weight

Qpred = tf.matmul(X,W) # Out Q prediction
Y = tf.placeholder(shape=[1,output_size],dtype=tf.float16)

loss = tf.reduce_sum(tf.square(Y-Qpred))
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss) # Y label

# Set Q-learning related parameters
dis = .99
num_episodes = 2000

# Create lists to contain total rewards and steps per episode
rList = []

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        # Reser environment and get first new observation
        s = dbr

        # The Q-Network training
        while not done:
            # Choose an action by greedily (with a chance of random action) from the Q-network
            Qs = sess.run(Qpred,feed_dict={X: s})
            if np.random,rand(1) < e:
                a = np.random.randint(-1,Ngrid)
            else:
                a = np.argmax(Qs)
            
            # Get new state and reward
            R = DBR.calR(s,Ngrid,wavelength,dx)
            reward = DBR.reward(s,Ngrid,wavelength,R,tarwave)
            s1,done = DBR.step(s,Ngrid,a)

            if done:
                # Update Q, and no Qs+1, since it's a terminal state
                Qs[0,a] = reward
            else:
                # Obtain the Q_s1 values by feeding the new state through our network
                Qs1 = sess. run(Qpred,feed_dict={X: s1})
                # Update Q
                Qs[0,a] = reward + dis*np.max(Qs1)

            # Train our network using target (Y) and predicted Q (Qpred) values
            sess.run(train, feed_dict={X: s, Y:Qs})

            rAll += reward
            s = s1
        rList.append(rAll)

print("percent of successful episodes: " +str(sum(rList) / num_episodes) + "%")
plt.bar(range(len(rList)), rList, color="blue")
plt.show()