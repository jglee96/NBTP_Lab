import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import DBR

# Real world environnment
Ngrid = 100
dx = 10
dbr = np.zeros(shape=[1,Ngrid],dtype=np.int8)
epsi = 12.25
eps0 = 1.

minwave = 400
maxwave = 1200
wavestep = 10
wavelength = np.array([np.arange(minwave,maxwave,wavestep)])
tarwave = 800

# Input and output size based on the Env
input_size = Ngrid
output_size = Ngrid
learning_rate = 0.1

# These lines establish the feed-forward part of the network used to
# choose actions
X = tf.placeholder(shape=[1,input_size],dtype=tf.float32) # state input

N1 = Ngrid
W1 = tf.Variable(tf.random_uniform([input_size,N1],0,1)) # weight
b1 = tf.Variable(tf.random_uniform([N1],0,1)) # weight
L1 = tf.nn.relu(tf.matmul(X,W1) + b1)

N2 = Ngrid
W2 = tf.Variable(tf.random_uniform([N1,N2],0,1)) # weight
b2 = tf.Variable(tf.random_uniform([N2],0,1)) # weight
L2 = tf.nn.relu(tf.matmul(L1,W2) + b2)

W3 = tf.Variable(tf.random_uniform([N2,output_size],0,1)) # weight
b3 = tf.Variable(tf.random_uniform([output_size],0,1)) # weight
Qpred = tf.matmul(L2,W3) + b3 # Out Q prediction
Y = tf.placeholder(shape=[1,output_size],dtype=tf.float32)

loss = tf.reduce_sum(tf.square(Y-Qpred))
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss) # Y label

# Set Q-learning related parameters
dis = .5
num_episodes = 2000
num_iter = 100

# Create lists to contain total rewards and steps per episode
rList = []

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        # Reser environment and get first new observation
        s = dbr
        e = 1/(1+i)
        rAll = 0
        prereward = 0
        done = False

        # The Q-Network training
        for iteridx in range(num_iter):
            # Choose an action by greedily (with a chance of random action) from the Q-network
            Qs = sess.run(Qpred,feed_dict={X: s})
            if np.random.rand(1) < e:
                a = np.random.randint(Ngrid)
            else:
                a = np.argmax(Qs)
            
            # Get new state and reward
            R = DBR.calR(s,Ngrid,wavelength,dx,epsi,eps0)
            reward = DBR.reward(Ngrid,wavelength,R,tarwave)
            reward = reward - prereward # the Q factor does not belong to action.
            s1 = DBR.step(s,a)

            if iteridx == num_iter-1:
                # Update Q, and no Qs+1, since it's a terminal state
                Qs[0,a] = reward
            else:
                # Obtain the Q_s1 values by feeding the new state through our network
                Qs1 = sess.run(Qpred,feed_dict={X: s1})
                # Update Q
                Qs[0,a] = reward + dis*np.max(Qs1)

            # Train our network using target (Y) and predicted Q (Qpred) values
            sess.run(train, feed_dict={X: s, Y: Qs})

            rAll += reward
            prereward = reward
            s = s1

        rList.append(rAll)

print("percent of successful episodes: " +str(sum(rList) / num_episodes) + "%")
plt.bar(range(len(rList)), rList, color="blue")
plt.show()