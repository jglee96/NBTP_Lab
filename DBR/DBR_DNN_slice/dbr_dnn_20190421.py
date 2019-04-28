import numpy as np
import tensorflow as tf
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sliceDBR
import fcdnn
import pandas as pd
from datetime import datetime

# Real world environnment
Nslice = 7
nh = 2.6811  # TiO2 at 400 nm (Siefke)
nl = 1.4701  # SiO2 at 400 nm (Malitson)

minwave = 200
maxwave = 800
wavestep = 5
wavelength = np.array([np.arange(minwave, maxwave, wavestep)])
tarwave = 400

# Constants defining our neural network
INPUT_SIZE = Nslice
OUTPUT_SIZE = len(wavelength[0])

Nfile = 9
Ntest = 1
Nsample = 10000

# train set file name
FPATH = 'D:/NBTP_Lab/DBR/DBR_DNN_slice'
SAVE_PATH = FPATH + '/result/'
statefilename = '/trainset/Trainset04/state_trainset04'
Rfilename = '/trainset/Trainset04/R_trainset04'


def Normalization(x):
        x_mean = x.mean(axis=0)
        x_std = x.std(axis=0)
        return (x-x_mean)/x_std


def getData():
    # Load Training Data
    Xarray = []
    Yarray = []
    for nf in range(Nfile):
        Xfname = FPATH+statefilename+'_'+str(nf)+'.txt'
        Xtemp = pd.read_csv(Xfname, header=None)
        Xtemp = Xtemp.values
        Xarray.append(Xtemp)

        Yfname = FPATH+Rfilename+'_'+str(nf)+'.txt'
        Ytemp = pd.read_csv(Yfname, header=None)
        Ytemp = Ytemp.values
        Yarray.append(Ytemp)

    Xcon = np.concatenate(Xarray)
    Ycon = np.concatenate(Yarray)

    sX = Xcon.reshape(-1, INPUT_SIZE)
    sX = Normalization(sX)
    sY = Ycon.reshape(-1, OUTPUT_SIZE)

    # Load testing data
    Xtarray = []
    Ytarray = []
    for nt in range(Ntest):
        Xtfname = FPATH+statefilename+'_'+str(nt+Nfile)
        Xttemp = pd.read_csv(Xtfname+'.txt', header=None)
        Xttemp = Xttemp.values
        Xtarray.append(Xttemp)

        Ytfname = FPATH+Rfilename+'_'+str(nt+Nfile)
        Yttemp = pd.read_csv(Ytfname+'.txt', header=None)
        Yttemp = Yttemp.values
        Ytarray.append(Yttemp)

    Xtcon = np.concatenate(Xtarray)
    Ytcon = np.concatenate(Ytarray)

    stX = Xtcon.reshape(-1, INPUT_SIZE)
    stX = Normalization(stX)
    stY = Ytcon.reshape(-1, OUTPUT_SIZE)

    return sX, sY, stX, stY


def main():
    # Load Data
    sX, sY, stX, stY = getData()
    # batch_size = [100]
    # num_layer = [3, 4, 5, 6, 7]
    # num_neuron = [200, 250, 300]
    # learning_rate = [1E-4, 2E-4, 4E-4, 6E-4, 8E-4, 1E-3]
    batch_size = [100]
    num_layer = [7]
    num_neuron = [250]
    learning_rate = [8E-4]

    nfigure = 0
    for bs in batch_size:
        for nl in num_layer:
            for nn in num_neuron:
                for lr in learning_rate:
                    # Clear our computational graph
                    tf.reset_default_graph()
                    with tf.Session() as sess:
                        print(bs, nl, nn, lr)
                        mainDNN = fcdnn.FC_DNN(session=sess,
                                               input_size=INPUT_SIZE,
                                               output_size=OUTPUT_SIZE,
                                               batch_size=bs, num_layer=nl,
                                               num_neuron=nn, learning_rate=lr,
                                               name='DBRNet')
                        mainDNN.writer.add_graph(sess.graph)
                        # Initialize Tensorflow variables
                        sess.run(tf.global_variables_initializer())

                        for n in range(int(Nsample*Nfile/bs)):
                            X = np.reshape(
                                sX[n*bs:(n+1)*bs], [bs, INPUT_SIZE])
                            Y = np.reshape(
                                sY[n*bs:(n+1)*bs], [bs, OUTPUT_SIZE])
                            mainDNN.update_train(X, Y, True)
                            if (n+1) % 100 == 0:
                                summary = mainDNN.update_tensorboard(
                                    X, Y, True)
                                mainDNN.writer.add_summary(
                                    summary, global_step=n)
                                print(n+1, 'th trained')
                        # Test
                        print("Testing Model...")
                        lbound = (tarwave/(4*nh))*0.1
                        hbound = (tarwave/(4*nl))*3
                        print('lbound: {}, hbound: {}'.format(lbound, hbound))
                        Tstate = np.random.randint(low=int(lbound),
                                                   high=int(hbound),
                                                   size=Nslice, dtype=int)
                        # Tstate = np.ones(Nslice)
                        # Tstate = np.array([int(tarwave/(4*nh)),
                        #                     int(tarwave/(4*nl)),
                        #                     int(tarwave/(4*nh)),
                        #                     int(tarwave/(4*nl)),
                        #                     int(tarwave/(4*nh)),
                        #                     int(tarwave/(4*nl)),
                        #                     int(tarwave/(4*nh))])
                        TR = sliceDBR.calR(
                            Tstate, Nslice, wavelength, nh, nl, True)
                        X = np.reshape(Tstate, [-1, INPUT_SIZE])
                        Y = np.reshape(TR, [-1, OUTPUT_SIZE])
                        NR = mainDNN.Test_paly(X, Y, False)
                        NR = np.reshape(NR, [OUTPUT_SIZE, -1])
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
                        x = np.reshape(wavelength, wavelength.shape[1])
                        plt.figure(2)
                        plt.subplot(2, 1, 1)
                        plt.plot(x, TR)

                        plt.subplot(2, 1, 2)
                        plt.plot(x, NR)
                        plt.show()

                        # plt.figure(nfigure)
                        # nfigure = nfigure + 1
                        # plt.plot(range(len(loss_List)), loss_List)
                        # fig2 = plt.gcf()
                        # fig2_name = SAVE_PATH+'/FollowScatterNet/{}_{}_{}_{:.5f}'.format(bs, nl, nn, lr)+'_loss.png'
                        # fig2_name = SAVE_PATH+'/'+datetime.now().strftime("%Y%m%d%H%M")
                        # fig2.savefig(fig2_name)

                        # f = open(FPATH+'/Loss Test(followscatternet).txt', 'a')
                        # loss_average = sum(loss_List)/len(loss_List)
                        # loss_max = max(loss_List)
                        # write_line = 'Batch Size: {}, Num Layer: {}, Num Neuron: {}, Learning Rate: {:.5f} -- Average: {:.3f}, Max: {:.5f}\n'.format(bs, nl, nn, lr, loss_average, loss_max)
                        # f.write(write_line)
                        # f.close()
if __name__ == "__main__":
    main()