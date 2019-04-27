import numpy as np
import tensorflow as tf
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fcdnn
import pandas as pd
from datetime import datetime

# train set file name
FPATH = 'D:/NBTP_Lab/DBR/ScatterNetfollow'
SAVE_PATH = FPATH +'/result/'

def getData(filename):
    shell_fileneame = filename + '_val.csv'
    spectrum_filename = filename + '.csv'
    shell = pd.read_csv(shell_fileneame, header=None)
    shell = shell.values
    INPUT_SIZE = shell.shape[1]
    spectrum = pd.read_csv(spectrum_filename, header=None)
    spectrum = spectrum.values
    OUTPUT_SIZE = spectrum.shape[1]

    return shell, spectrum, INPUT_SIZE, OUTPUT_SIZE


def main():
    # Load testing data
    train_X, train_Y, INPUT_SIZE, OUTPUT_SIZE = getData('D:/NBTP_Lab/DBR/ScatterNetfollow/data/5_layer_tio2')
    # batch_size = [64]
    # num_layer = [2, 4, 6, 8]
    # num_neuron = [100, 150, 200, 250]
    # learning_rate = [1E-2, 1E-3, 1E-4]
    # 32, 10, 250, 1E-3 is best hyperparameter
    batch_size = [100]
    num_layer = [4]
    num_neuron = [225]
    learning_rate = [1E-3]

    wavelength = range(400,801,2)

    # nfigure = 0
    for bs in batch_size:
        for nl in num_layer:
            for nn in num_neuron:
                for lr in learning_rate:
                    # Clear our computational graph
                    tf.reset_default_graph()
                    with tf.Session() as sess:
                        print(bs,nl,nn,lr)
                        mainDNN = fcdnn.FC_DNN(session=sess, input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, batch_size=bs, num_layer=nl, num_neuron=nn, learning_rate=lr, name='Inputbn_{}_{}_{}_{:.5f}'.format(bs, nl, nn, lr))
                        mainDNN.writer.add_graph(sess.graph)   
                        sess.run(tf.global_variables_initializer()) # Initialize Tensorflow variables

                        for n in range(int(train_X.shape[0]/bs)):
                            X = np.reshape(train_X[n*bs:(n+1)*bs],[bs,INPUT_SIZE])
                            Y = np.reshape(train_Y[n*bs:(n+1)*bs],[bs,OUTPUT_SIZE])
                            mainDNN.update_train(X, Y, True)
                            if (n+1)%100 == 0:
                                summary = mainDNN.update_tensorboard(X, Y, True)
                                mainDNN.writer.add_summary(summary, global_step=n)
                                print(n+1,'th trained')
                        #Test
                        print("Testing Model...")
                        Tstate = train_X[-1,:]
                        TR = train_Y[-1,:]
                        X = np.reshape(Tstate,[-1,INPUT_SIZE])
                        Y = np.reshape(TR,[-1,OUTPUT_SIZE])
                        NR = mainDNN.Test_paly(X, Y, False)
                        NR = np.reshape(NR,[OUTPUT_SIZE,-1])
                        Tloss = mainDNN.update_loss(X, Y, False)
                        # loss_List = []
                        # for n in range(Nsample*Ntest):
                        #     tX = np.reshape(stX[n], [-1,INPUT_SIZE])
                        #     tY = np.reshape(stY[n], [-1,OUTPUT_SIZE])
                        #     tloss = mainDNN.update_loss(tX, tY, False)
                        #     loss_List.append(tloss)
                        #     if (n+1)%1000 == 0:
                        #         print(n+1,'th tested')

                        print('LOSS: ', Tloss)
                        x = np.reshape(wavelength,OUTPUT_SIZE)
                        plt.figure(2)
                        plt.subplot(2,1,1)
                        plt.plot(x,TR)
                    
                        plt.subplot(2,1,2)
                        plt.plot(x,NR)
                        plt.show()

                        # plt.figure(nfigure)
                        # nfigure = nfigure + 1
                        # plt.plot(range(len(loss_List)), loss_List)
                        # fig2 = plt.gcf()
                        # fig2_name = SAVE_PATH+'/TMM/TMM_{}_{}_{}_{:.5f}'.format(bs, nl, nn, lr)+'_loss.png'
                        # fig2_name = SAVE_PATH+'/'+datetime.now().strftime("%Y%m%d%H%M")
                        # fig2.savefig(fig2_name)

                        # f = open(FPATH+'/Loss Test(20190425157).txt', 'a')
                        # loss_average = sum(loss_List)/len(loss_List)
                        # loss_max = max(loss_List)
                        # write_line = 'Batch Size: {}, Num Layer: {}, Num Neuron: {}, Learning Rate: {:.5f} -- Average: {:.3f}, Max: {:.5f}\n'.format(bs, nl, nn, lr, loss_average, loss_max)
                        # f.write(write_line)
                        # f.close()
if __name__ == "__main__":
    main()