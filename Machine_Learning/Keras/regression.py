import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = list(range(-20,21))

np.random.seed(2020)

mu = 0
sigma = 20
n = len(x_data)

noises = np.random.normal(mu, sigma, n)

W_answer = 3
b_answer = -3

y_temp = list(np.array(x_data)*W_answer + b_answer)
y_data = list(np.array(y_temp) + np.array(noises))

plt.figure(figsize=(6,6))
plt.scatter(x_data, y_data)
plt.xlabel('x_data')
plt.ylabel('y_data')
plt.show()

W = tf.Variable(-0.5)
b = tf.Variable(-0.5)

learning_rate = 0.001
cost_list=[]

# Train
for i in range(1000+1):
    
    # Gradient Descent
    with tf.GradientTape() as tape:
        
        hypothesis = W * x_data + b
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))
    
        # Update
        W_grad, b_grad = tape.gradient(cost, [W, b])
        W.assign_sub(learning_rate * W_grad)
        b.assign_sub(learning_rate * b_grad)
        cost_list.append(cost.numpy())
    
    # Output
    if i % 200 == 0:
        
        print("#%s \t W: %s \t b: %s \t Cost: %s" % (i, W.numpy(), b.numpy(), cost.numpy()))
        
        plt.figure(figsize=(6,6))
        plt.title('#%s Training Linear Regression Model' % i, size=15)
        plt.scatter(x_data, y_data, color='blue', label='Real Values')
        plt.plot(x_data, hypothesis, color='red', label='Hypothesis')
        plt.xlabel('x_data')
        plt.legend(loc='upper left')
        plt.show()
        
        print('\n')

plt.title('Cost Values', size=15)
plt.plot(cost_list, color='orange')
plt.xlabel('Number of learning')
plt.ylabel('Cost Values')
plt.show()