import numpy as np

s = np.array([1,0,0,0,0,2,1,0,0,1])
s1 = 3
s2 = 5
smean = np.mean(np.hstack((s[0:s1+1],s[s2:])))

sidx = np.where(s == 2)[0][0]

print("result = ", smean)