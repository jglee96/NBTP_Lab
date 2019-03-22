import numpy as np
from datetime import datetime

s = np.array([[1,0,0,0,0,2,1,0,0,1]])

ss = np.reshape(s,s.shape[1])

file_name = 'result_'+datetime.now().strftime("%Y-%m-%d")+'.txt'

print(file_name)