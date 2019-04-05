import numpy as np
from collections import deque
import tensorflow as tf
import os

#s = np.array([1,0,0,0,0,2,1,0,0,1])
#s1 = 3
#s2 = 5
#smean = np.mean(np.hstack((s[0:s1+1],s[s2:])))
#
#sidx = np.where(s == 2)[0][0]
#
#print("result = ", smean)

#buffer = deque(maxlen = 100)
#batch_size = 20
#
#for i in range(50):
#    buffer.append(i)
#
#if len(buffer) > batch_size:
#    minibatch = random.sample(buffer, batch_size)
#
#print("buffer length = ",len(buffer),"minibatch length = ",len(minibatch))

#
#a = ['W1', 'W2']
#
#for i in range(2):
#        print(a[i])


MODEL_PATH = './model/'
MODEL_NAME = 'dqn_2019040307.ckpt'

load_path = os.path.join(MODEL_PATH,MODEL_NAME)

variables = tf.contrib.framework.list_variables(load_path)

for i, v in enumerate(variables):
    print("{}. name : {} \n    shape : {}".format(i, v[0], v[1]))