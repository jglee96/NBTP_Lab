import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0,x)
    
x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x)
y2 = relu(x)

plt.plot(x, y1)
plt.plot([0,0],[1.0,0.0], ':')
plt.ylim(-0.1, 1.1) #y축 범위 지정
plt.title('Sigmoid Function')
plt.show()

plt.plot(x, y2)
plt.plot([0,0],[1.0,0.0], ':')
plt.ylim(-0.1, 1.1) #y축 범위 지정
plt.title('ReLu Function')
plt.show()