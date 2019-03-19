import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import DBR

# Real world environnment
Ngrid = 100
N1 = Ngrid
N2 = Ngrid
dx = 5
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
learning_rate = 0.01

# These lines establish the feed-forward part of the network used to
# choose actions
with tf.device('/device:GPU:1'):
    X = tf.placeholder(shape=[1,input_size],dtype=tf.float32) # state input

    W1 = tf.Variable(tf.random_uniform([input_size,N1],minval=0,maxval=0.01)) # weight
#    b1 = tf.Variable(tf.random_uniform([N1],minval=0,maxval=0.01)) # weight
    L1 = tf.nn.relu(tf.matmul(X,W1))
    
    W2 = tf.Variable(tf.random_uniform([N1,N2],minval=0,maxval=0.01)) # weight
#    b2 = tf.Variable(tf.random_uniform([N2],minval=0,maxval=0.01)) # weight
    L2 = tf.nn.relu(tf.matmul(L1,W2))

    W3 = tf.Variable(tf.random_uniform([N2,output_size],minval=0,maxval=0.01)) # weight
#    b3 = tf.Variable(tf.random_uniform([output_size],minval=0,maxval=0.01)) # weight
    Qpred = tf.matmul(L2,W3) # Out Q prediction
    Y = tf.placeholder(shape=[1,output_size],dtype=tf.float32)

    loss = tf.reduce_sum(tf.square(Y-Qpred))
    train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss) # Y label

# Set Q-learning related parameters
dis = .99
num_episodes = 100
num_iter = 100

# Create lists to contain total rewards and steps per episode
rList = []

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess.run(init)
    for i in range(num_episodes):
        # Reser environment and get first new observation
        s = np.zeros((1,Ngrid))
        e = 1/(1+i)
        rAll = 0
        prereward = 0
        local_loss = []

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
            rawreward = DBR.reward(Ngrid,wavelength,R,tarwave)
            reward = rawreward - prereward # the Q factor does not belong to action.
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
            prereward = rawreward
            s = s1.copy()

        rList.append(rAll)

print("percent of successful episodes: " +str(sum(rList) / num_episodes) + "%")
plt.bar(range(len(rList)), rList, color="blue")
plt.show()

plt.imshow(s, cmap = "gray")
plt.show()