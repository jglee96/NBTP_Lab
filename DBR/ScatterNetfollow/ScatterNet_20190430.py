import numpy as np
import tensorflow as tf
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fcdnn
import pandas as pd
import os
import time
from datetime import datetime

# train set file name
FPATH = 'D:/NBTP_Lab/DBR/ScatterNetfollow'
SAVE_PATH = FPATH + '/result/'

# saved model name
MODEL_PATH = 'D:/NBTP_Lab/DBR/ScatterNetfollow/model/'
MODEL_NAME = '2019050221.ckpt'

# fixed data
wavelength = range(400, 801, 2)

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
        # save mean and std of data
        x_mean = np.mean(shell, axis=0)
        x_std = np.std(shell, axis=0)
        specname = 'D:/NBTP_Lab/DBR/ScatterNetfollow/result/5_layer_spec.csv'
        np.savetxt(specname, (x_mean, x_std), delimiter=',')

        shell = (shell - x_mean) / x_std
        INPUT_SIZE = shell.shape[1]
        spectrum = pd.read_csv(spectrum_filename, header=None)
        spectrum = np.transpose(spectrum.values)
        OUTPUT_SIZE = spectrum.shape[1]

        return shell, spectrum, INPUT_SIZE, OUTPUT_SIZE, x_mean, x_std

def gen_spect_file():
        npwave = np.array(wavelength)
        spect = np.zeros((1, npwave.shape[0]), dtype=int)
        minrange = 600
        maxrange = 650
        
        minidx = np.where(npwave == minrange)[0][0]
        maxidx = np.where(npwave == maxrange)[0][0]

        spect[0, minidx:maxidx+1] = 1
        filename = 'D:/NBTP_Lab/DBR/ScatterNetfollow/data/gen_spect('+str(minrange)+'_'+str(maxrange)+').csv'
        np.savetxt(filename, spect, delimiter=',')
        print('======== Spectrum File Saved ========')

def design_specturm():
        # Load base data
        INPUT_SIZE = 5
        OUTPUT_SIZE = len(wavelength)        
        
        init_list_rand = tf.constant(
                np.random.rand(1, INPUT_SIZE)*40.0+30.0, dtype=tf.float32)
        x = tf.get_variable(name='b', initializer=init_list_rand)
        y = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE])

        specname = 'D:/NBTP_Lab/DBR/ScatterNetfollow/result/5_layer_spec.csv'
        data = pd.read_csv(specname, header=None)
        data = data.values
        x_mean = data[0]
        x_std = data[1]

        # Forward propagation
        x_norm = (x - x_mean) / x_std

        minLimit = (30 - x_mean) / x_std
        maxLimit = (70 - x_mean) / x_std
        x_norm = tf.maximum(x_norm, minLimit)
        x_norm = tf.minimum(x_norm, maxLimit)

        train_saver = tf.train.import_meta_graph(MODEL_PATH+MODEL_NAME+'.meta')
        train_graph = tf.get_default_graph()
        W0 = train_graph.get_tensor_by_name('FC0/weights:0')
        b0 = train_graph.get_tensor_by_name('FC0/biases:0')
        L0 = tf.nn.relu(tf.matmul(x_norm, W0) + b0)

        W1 = train_graph.get_tensor_by_name('FC1/weights:0')
        b1 = train_graph.get_tensor_by_name('FC1/biases:0')
        L1 = tf.nn.relu(tf.matmul(L0, W1) + b1)

        W2 = train_graph.get_tensor_by_name('FC2/weights:0')
        b2 = train_graph.get_tensor_by_name('FC2/biases:0')
        L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

        W3 = train_graph.get_tensor_by_name('FC3/weights:0')
        b3 = train_graph.get_tensor_by_name('FC3/biases:0')
        L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)

        W4 = train_graph.get_tensor_by_name('fully_connected/weights:0')
        b4 = train_graph.get_tensor_by_name('fully_connected/biases:0')
        L4 = tf.add(tf.matmul(L3,W4), b4)  # No activation function

        yhat = L4  # use pre-trained model
        # Backward propagation
        # This will select all the values that we want.
        topval = tf.abs(tf.matmul(y,tf.transpose(tf.abs(yhat))))
        # topval = tf.reduce_mean(tf.matmul(y,tf.transpose(tf.abs(yhat))))
        # This will get the values that we do not want
        botval = tf.abs(tf.matmul(tf.abs(y-1),tf.transpose(tf.abs(yhat))))
        # botval = tf.reduce_mean(tf.matmul(tf.abs(y-1),tf.transpose(tf.abs(yhat))))
        cost = botval/topval
        # global_step = tf.Variable(0, trainable=False)
        # learning_rate = tf.train.exponential_decay(
        #         1E-3,global_step,1000,0.7, staircase=False)
        # optimizer = tf.train.RMSPropOptimizer(
        #         learning_rate=learning_rate).minimize(
        #                 cost, global_step=global_step, var_list=[x])
        optimizer = tf.train.AdamOptimizer(learning_rate=1E-2).minimize(cost, var_list=[x])

        # get design data where we want
        design_name = 'D:/NBTP_Lab/DBR/ScatterNetfollow/data/gen_spect(500_550)'
        design_y = getdesignData(design_name)
        numEpoch = 50000
        start_time=time.time()
        print("========                         Iterations started                  ========")
        with tf.Session() as sess:
                load_path = os.path.join(MODEL_PATH,MODEL_NAME)
                train_saver.restore(sess, load_path)
                sess.run(tf.global_variables_initializer())
                for n in range(numEpoch):
                        sess.run(optimizer, feed_dict={y: design_y})
                        if (n+1) % 500 == 0:
                                loss = sess.run(cost, feed_dict={y: design_y})[0][0]
                                print('Step: {}, Loss: {:.5f}, X: {}'.format((n+1), loss, x.eval()))
                converge_x = x.eval()
        print("========Iterations completed in : {:.3f} ========".format((time.time()-start_time)))
        return converge_x, INPUT_SIZE, OUTPUT_SIZE, x_mean, x_std

def show_spectrum(input_x, INPUT_SIZE, OUTPUT_SIZE, x_mean, x_std):
        print("========Show Spectrum of inverse design's result========")
        print(input_x)
        x = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE])
        x_norm = (x - x_mean) / x_std

        train_saver = tf.train.import_meta_graph(MODEL_PATH+MODEL_NAME+'.meta')
        train_graph = tf.get_default_graph()
        W0 = train_graph.get_tensor_by_name('FC0/weights:0')
        b0 = train_graph.get_tensor_by_name('FC0/biases:0')
        L0 = tf.nn.relu(tf.matmul(x_norm, W0) + b0)

        W1 = train_graph.get_tensor_by_name('FC1/weights:0')
        b1 = train_graph.get_tensor_by_name('FC1/biases:0')
        L1 = tf.nn.relu(tf.matmul(L0, W1) + b1)

        W2 = train_graph.get_tensor_by_name('FC2/weights:0')
        b2 = train_graph.get_tensor_by_name('FC2/biases:0')
        L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

        W3 = train_graph.get_tensor_by_name('FC3/weights:0')
        b3 = train_graph.get_tensor_by_name('FC3/biases:0')
        L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)

        W4 = train_graph.get_tensor_by_name('fully_connected/weights:0')
        b4 = train_graph.get_tensor_by_name('fully_connected/biases:0')
        Y = tf.add(tf.matmul(L3,W4), b4)  # No activation function
        
        with tf.Session() as sess:
                load_path = os.path.join(MODEL_PATH,MODEL_NAME)
                train_saver.restore(sess, load_path)
                spectrum = sess.run(Y, feed_dict={x: input_x})
                l = np.reshape(wavelength, OUTPUT_SIZE)
                spectrum = np.reshape(spectrum, OUTPUT_SIZE)
                plt.figure(3)
                plt.plot(l, spectrum)
                plt.show()

def Test_spectrum():
        tf.reset_default_graph()
        # Load base data
        INPUT_SIZE = 5
        OUTPUT_SIZE = len(wavelength)
        specname = 'D:/NBTP_Lab/DBR/ScatterNetfollow/result/5_layer_spec.csv'
        data = pd.read_csv(specname, header=None)
        data = data.values
        x_mean = data[0]
        x_std = data[1]

        print("========Test model by showing random structure spectrum========")
        input_x = np.random.rand(1, INPUT_SIZE)*40.0+30.0
        print('raw input: ', input_x)
        input_x = (input_x - x_mean) / x_std
        print('normalized input: ', input_x)
        x = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE])

        train_saver = tf.train.import_meta_graph(MODEL_PATH+MODEL_NAME+'.meta')
        train_graph = tf.get_default_graph()
        W0 = train_graph.get_tensor_by_name('FC0/weights:0')
        b0 = train_graph.get_tensor_by_name('FC0/biases:0')
        L0 = tf.nn.relu(tf.add(tf.matmul(x, W0), b0))

        W1 = train_graph.get_tensor_by_name('FC1/weights:0')
        b1 = train_graph.get_tensor_by_name('FC1/biases:0')
        L1 = tf.nn.relu(tf.add(tf.matmul(L0, W1), b1))

        W2 = train_graph.get_tensor_by_name('FC2/weights:0')
        b2 = train_graph.get_tensor_by_name('FC2/biases:0')
        L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), b2))

        W3 = train_graph.get_tensor_by_name('FC3/weights:0')
        b3 = train_graph.get_tensor_by_name('FC3/biases:0')
        L3 = tf.nn.relu(tf.add(tf.matmul(L2, W3), b3))

        W4 = train_graph.get_tensor_by_name('fully_connected/weights:0')
        b4 = train_graph.get_tensor_by_name('fully_connected/biases:0')
        Y = tf.add(tf.matmul(L3,W4), b4)  # No activation function
        
        with tf.Session() as sess:
                load_path = os.path.join(MODEL_PATH,MODEL_NAME)
                train_saver.restore(sess, load_path)
                spectrum = sess.run(Y, feed_dict={x: input_x})
                l = np.reshape(wavelength, OUTPUT_SIZE)
                spectrum = np.reshape(spectrum, OUTPUT_SIZE)
                plt.figure(3)
                plt.plot(l, spectrum)
                plt.show()

def main():
    # Load testing data
    filename = 'D:/NBTP_Lab/DBR/ScatterNetfollow/data/5_layer_tio2'
    train_X, train_Y, INPUT_SIZE, OUTPUT_SIZE, x_mean, x_std = getData(filename)
    print("""Data Loading Sucess!!\
 INPUT_SIZE: {}, OUTPUT_SIZE: {}""".format(INPUT_SIZE, OUTPUT_SIZE))
    bs = 100  # remain 40 data to test
    nl = 4
    nn = 225
    lr = 1E-2

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
        tx = np.random.rand(1, INPUT_SIZE)*40.0+30.0
        print(tx)
        tx = (tx - x_mean) / x_std
        tr = mainDNN.Test_paly(tx, False)
        tr = np.reshape(tr, [OUTPUT_SIZE, -1])
        Tstate = train_X[39990, :]
        TR = train_Y[39990, :]
        X = np.reshape(Tstate, [-1, INPUT_SIZE])
        Y = np.reshape(TR, [-1, OUTPUT_SIZE])
        NR = mainDNN.Test_paly(X, False)
        NR = np.reshape(NR, [OUTPUT_SIZE, -1])
        Tloss = mainDNN.update_loss(X, Y, False)
        X = np.reshape(train_X[bs*int(train_X.shape[0]/bs):-1], [-1, INPUT_SIZE])
        Y = np.reshape(train_Y[bs*int(train_X.shape[0]/bs):-1], [-1, OUTPUT_SIZE])
        tloss = mainDNN.update_loss(X, Y, False)
        print('Trained number: ', bs*int(train_X.shape[0]/bs), 'LOSS: ', tloss)

        print('Example LOSS: ', Tloss)
        x = np.reshape(wavelength, OUTPUT_SIZE)
        plt.figure(3)
        plt.plot(x, tr)
        plt.figure(2)
        plt.subplot(2, 1, 1)
        plt.plot(x, TR)
        plt.subplot(2, 1, 2)
        plt.plot(x, NR)
        plt.show()

if __name__ == "__main__":
        train = False
        test = False
        genspect = False
        if train:  # trin model
                main()
        elif test:
                Test_spectrum()
        elif genspect:
                gen_spect_file()
        else:  # use trained model
                input_x, INPUT_SIZE, OUTPUT_SIZE, x_mean, x_std = design_specturm()
                input_x = np.reshape(input_x, [-1, INPUT_SIZE])
                show_spectrum(input_x, INPUT_SIZE, OUTPUT_SIZE, x_mean, x_std)