import numpy as np
import tensorflow as tf
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fcdnn
import pandas as pd
import os
from datetime import datetime

# train set file name
FPATH = 'D:/NBTP_Lab/DBR/ScatterNetfollow'
SAVE_PATH = FPATH + '/result/'

# saved model name
MODEL_PATH = 'D:/NBTP_Lab/DBR/ScatterNetfollow/model/'
MODEL_NAME = '2019042921.ckpt'


def Normalization(x):
        x_mean = x.mean(axis=0)
        x_std = x.std(axis=0)
        return (x-x_mean)/x_std

def getdesignData(filename):
        design_filename = filename + '.csv'
        design = pd.read_csv(design_filename, header=None)
        design = design.values

        return design

def getData(filename):
        shell_fileneame = filename + '_val.csv'
        spectrum_filename = filename + '.csv'
        shell = pd.read_csv(shell_fileneame, header=None)
        shell = shell.values
        shell = Normalization(shell)
        INPUT_SIZE = shell.shape[1]
        spectrum = pd.read_csv(spectrum_filename, header=None)
        spectrum = np.transpose(spectrum.values)
        OUTPUT_SIZE = spectrum.shape[1]

        return shell, spectrum, INPUT_SIZE, OUTPUT_SIZE

def forward_network(input_x):
        forwardnet_graph = tf.Graph()
        with forwardnet_graph.as_default() as g:
                X = tf.placeholder(tf.float32, [None, 5])
                Y = tf.placeholder(tf.float32, shape=[None, 201])

                W0 = tf.get_variable('FC0/weights', shape=(5, 250))
                b0 = tf.get_variable('FC0/biases', shape=250)
                L0 = tf.nn.relu(tf.matmul(X, W0) + b0)

                W1 = tf.get_variable('FC1/weights', shape=(250, 250))
                b1 = tf.get_variable('FC1/biases', shape=250)
                L1 = tf.nn.relu(tf.matmul(L0, W1) + b1)

                W2 = tf.get_variable('FC2/weights', shape=(250, 250))
                b2 = tf.get_variable('FC2/biases', shape=250)
                L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

                W3 = tf.get_variable('FC3/weights', shape=(250, 250))
                b3 = tf.get_variable('FC3/biases', shape=250)
                L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)

                W4 = tf.get_variable('fully_connected/weights', shape=(250, 201))
                b4 = tf.get_variable('fully_connected/biases', shape=201)
                L4 = tf.nn.relu(tf.matmul(L3,W4) + b4)
                Y = L4

                load_path = os.path.join(MODEL_PATH,MODEL_NAME)
                saver = tf.train.Saver()
        with tf.Session(graph=forwardnet_graph) as sess:
                saver.restore(sess, load_path)

                return sess.run(Y, feed_dict={X: input_x})

def design_specturm():
        # Load base data
        filename = 'D:/NBTP_Lab/DBR/ScatterNetfollow/data/5_layer_tio2'
        train_X, train_Y, INPUT_SIZE, OUTPUT_SIZE = getData(filename)
        wavelength = range(400, 801, 2)
        designnet_graph = tf.Graph()
        with designnet_graph.as_default() as g:
                init_list_rand = tf.constant(
                        np.random.rand(1, INPUT_SIZE)*40.0+30.0, dtype=tf.float32)
                x = tf.Variable(init_list_rand)
                y = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE])
                
                # get design data where we want
                design_name = 'D:/NBTP_Lab/DBR/ScatterNetfollow/data/test_gen_spect'
                design_y = getdesignData(design_name)

                myl = np.array(wavelength)
                newL = 1.0/(myl*myl*3).astype(np.float32)
                # Forward propagation
                x_mean, x_var = tf.nn.moments(x, axes=[1])
                x_std = tf.sqrt(x_var)
                x_norm = (x - x_mean) / x_std
                yhat = forward_network(x_norm
                yhat = tf.multply(yhat,newL)
                # Backward propagation
                # This will select all the values that we want.
                topval = tf.abs(tf.matmul(y,tf.transpose(tf.abs(yhat))))
                # This will get the values that we do
                botval = tf.abs(tf.matmul(tf.abs(y-1),tf.transpose(tf.abs(yhat))))
                cost = topval/botval
                global_step = tf.Variable(0, trainable=Flase)
                learning_rate = tf.train.exponential_decay(
                        lr_rate,global_step,1000,lr_decay, staircase=False)
                optimizer = tf.train.RMSPropOptimizer(
                        learning_rate=learning_rate).minimize(
                                cost, global_step=global_step, var_list=[x])

        numEpoch = 10
        with tf.Session(graph=designnet_graph) as sess:
                sess.run(tf.global_variables_initializer())
                for n in numEpoch:
                        sess.run(optimizer, feed_dict={y: design_y})
                        loss = sess.run(cost, feed_dict={y:design_y})
                        print('Step: {}, Loss: {}, X: {}'.format((n+1), loss, x.eval()))
        print("--Iterations completed--")



def main():
    # Load testing data
    filename = 'D:/NBTP_Lab/DBR/ScatterNetfollow/data/5_layer_tio2'
    train_X, train_Y, INPUT_SIZE, OUTPUT_SIZE = getData(filename)
    print("""Data Loading Sucess!!\
 INPUT_SIZE: {}, OUTPUT_SIZE: {}""".format(INPUT_SIZE, OUTPUT_SIZE))
    bs = 90  # remain 40 data to test
    nl = 4
    nn = 250
    lr = 1E-2

    wavelength = range(400, 801, 2)
    # Clear our computational graph
    tf.reset_default_graph()
    with tf.Session() as sess:
        mainDNN = fcdnn.FC_DNN(session=sess,
                               input_size=INPUT_SIZE,
                               output_size=OUTPUT_SIZE,
                               batch_size=bs, num_layer=nl,
                               num_neuron=nn, learning_rate=lr,
                               name='ScatterNet')
        mainDNN.writer.add_graph(sess.graph)
        # Initialize Tensorflow variables
        sess.run(tf.global_variables_initializer())

        for n in range(int(train_X.shape[0]/bs)):
            X = np.reshape(train_X[n*bs:(n+1)*bs], [bs, INPUT_SIZE])
            Y = np.reshape(train_Y[n*bs:(n+1)*bs], [bs, OUTPUT_SIZE])
            mainDNN.update_train(X, Y, True)
            if (n+1) % 100 == 0:
                summary = mainDNN.update_tensorboard(X, Y, True)
                mainDNN.writer.add_summary(summary, global_step=n)
                print(n+1, 'th trained')
        # Save model
        model_file = FPATH + '/model/' + datetime.now().strftime("%Y%m%d%H")+'.ckpt'
        saver = tf.train.Saver()
        saver.save(sess, model_file)
        print('*****Trained Model Save(',model_file,')*****')

        # Test
        print("Testing Model...")
        Tstate = train_X[39990, :]
        TR = train_Y[39990, :]
        X = np.reshape(Tstate, [-1, INPUT_SIZE])
        Y = np.reshape(TR, [-1, OUTPUT_SIZE])
        NR = mainDNN.Test_paly(X, Y, False)
        NR = np.reshape(NR, [OUTPUT_SIZE, -1])
        Tloss = mainDNN.update_loss(X, Y, False)
        X = np.reshape(train_X[bs*int(train_X.shape[0]/bs):-1], [-1, INPUT_SIZE])
        Y = np.reshape(train_Y[bs*int(train_X.shape[0]/bs):-1], [-1, OUTPUT_SIZE])
        tloss = mainDNN.update_loss(X, Y, False)
        print('Trained number: ', bs*int(train_X.shape[0]/bs), 'LOSS: ', tloss)

        print('Example LOSS: ', Tloss)
        x = np.reshape(wavelength, OUTPUT_SIZE)
        plt.figure(2)
        plt.subplot(2, 1, 1)
        plt.plot(x, TR)
        plt.subplot(2, 1, 2)
        plt.plot(x, NR)
        plt.show()

if __name__ == "__main__":
        train = False
        design = True
        if train:
                main()
        elif design:
                design_specturm()