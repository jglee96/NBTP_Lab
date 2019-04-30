import numpy as np
from collections import deque
import tensorflow as tf
import os

MODEL_PATH = 'D:/NBTP_Lab/DBR/ScatterNetfollow/model/'
MODEL_NAME = '2019042921.ckpt'

load_path = os.path.join(MODEL_PATH,MODEL_NAME)
variables = tf.contrib.framework.list_variables(load_path)

for i, v in enumerate(variables):
    print("{}. name : {} \n    shape : {}".format(i, v[0], v[1]))