import numpy as np

s = np.array(range(30, 71, 1))
s_mean = np.mean(s)
s_std = np.std(s)

print('mean: {:.2f}, std: {:.2f}'.format(s_mean, s_std))