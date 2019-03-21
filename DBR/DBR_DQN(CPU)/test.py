import numpy as np
from collections import deque
import random

#s = np.array([1,0,0,0,0,2,1,0,0,1])
#s1 = 3
#s2 = 5
#smean = np.mean(np.hstack((s[0:s1+1],s[s2:])))
#
#sidx = np.where(s == 2)[0][0]
#
#print("result = ", smean)

buffer = deque(maxlen = 100)
batch_size = 20

for i in range(50):
    buffer.append(i)

if len(buffer) > batch_size:
    minibatch = random.sample(buffer, batch_size)

print("buffer length = ",len(buffer),"minibatch length = ",len(minibatch))