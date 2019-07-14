import numpy as np
from collections import deque
import tensorflow as tf
import os

MODEL_PATH = 'D:/NBTP_Lab/DBR/ScatterNetfollow/model/'
MODEL_NAME = '2019043021.ckpt'

load_path = os.path.join(MODEL_PATH,MODEL_NAME)
variables = tf.contrib.framework.list_variables(load_path)
train_saver = tf.train.import_meta_graph(MODEL_PATH+MODEL_NAME+'.meta')
train_graph = tf.get_default_graph()

for i, v in enumerate(variables):
    print("{}. name : {} \n    shape : {}".format(i, v[0], v[1]))

# graph_op  = train_graph.get_operations()
# with open('output.txt', 'w') as f:
#     for i in graph_op:
#         f.write(str(i))