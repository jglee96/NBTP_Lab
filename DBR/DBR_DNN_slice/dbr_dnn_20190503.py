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

Nfile = 10
Ntest = 1
Nsample = 10000

# train set file name
FPATH = 'D:/NBTP_Lab/DBR/DBR_DNN_slice'
SAVE_PATH = FPATH + '/result/'
statefilename = '/trainset/Trainset07/state_trainset07'
Rfilename = '/trainset/Trainset07/R_trainset07'

# Base data
lbound = (tarwave/(4*nh)) - 20
ubound = (tarwave/(4*nl)) + 20
print("======== Design Information ========")
print('tarwave: {}, nh: {:.3f}, nl: {:.3f}'.format(tarwave, nh, nl))
print('lbound: {:.3f}, ubound: {:.3f}'.format(lbound, ubound))

def getData():
    # Load Training Data
    print("========      Load Data     ========")
    Xarray = []
    Yarray = []
    for nf in range(Nfile):
        Xfname = FPATH+statefilename+'_'+str(nf)+'.csv'
        Xtemp = pd.read_csv(Xfname, header=None)
        Xtemp = Xtemp.values
        Xarray.append(Xtemp)

        Yfname = FPATH+Rfilename+'_'+str(nf)+'.csv'
        Ytemp = pd.read_csv(Yfname, header=None)
        Ytemp = Ytemp.values
        Yarray.append(Ytemp)

    sX = np.concatenate(Xarray)
    sY = np.concatenate(Yarray)

    x_mean = np.mean(sX, axis=0)
    x_std = np.std(sX, axis=0)
    sX = (sX - x_mean) / x_std

    # Load testing data
    Xtarray = []
    Ytarray = []
    for nt in range(Ntest):
        Xtfname = FPATH+statefilename+'_'+str(nt+Nfile)
        Xttemp = pd.read_csv(Xtfname+'.csv', header=None)
        Xttemp = Xttemp.values
        Xtarray.append(Xttemp)

        Ytfname = FPATH+Rfilename+'_'+str(nt+Nfile)
        Yttemp = pd.read_csv(Ytfname+'.csv', header=None)
        Yttemp = Yttemp.values
        Ytarray.append(Yttemp)

    stX = np.concatenate(Xtarray)
    stY = np.concatenate(Ytarray)

    stX = (stX - x_mean) / x_std

    return sX, sY, stX, stY, x_mean, x_std


def main():
    # Load Data
    sX, sY, stX, stY, x_mean, x_std = getData()
    print("========  Load Data Sucess  ========")
    bs = 64
    nl = 2
    nn = 4000
    lr = 1E-4
    # beta1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # beta2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999]
    beta1 = [0.9]
    beta2 = [0.7]

    print("========   Training Start   ========")
    nfigure = 0
    for b1 in beta1:
        for b2 in beta2:
            # Clear our computational graph
            tf.reset_default_graph()
            with tf.Session() as sess:
                print(b1, b2)
                mainDNN = fcdnn.FC_DNN(
                    session=sess, input_size=INPUT_SIZE,
                    output_size=OUTPUT_SIZE, batch_size=bs,
                    num_layer=nl, num_neuron=nn, learning_rate=lr,
                    name='DBRNet', beta1=b1, beta2=b2)
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
                        print('{}th trained'.format(n+1))
                # Test
                print("Testing Model...")               
                # Tstate = np.ones(Nslice)
                # Tstate = np.array([int(tarwave/(4*nh)),
                #                     int(tarwave/(4*nl)),
                #                     int(tarwave/(4*nh)),
                #                     int(tarwave/(4*nl)),
                #                     int(tarwave/(4*nh)),
                #                     int(tarwave/(4*nl)),
                #                     int(tarwave/(4*nh))])
                Tstate = stX[1]
                # TR = sliceDBR.calR(
                #     Tstate, Nslice, wavelength, nh, nl, True)
                Tstate = (Tstate - x_mean) / x_std
                TR = stY[1]
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

                # f = open(FPATH+'/Loss Test(Sigmoid Entropy SSE).txt', 'a')
                # # loss_average = sum(loss_List)/len(loss_List)
                # # loss_max = max(loss_List)
                # write_line = 'Beta1: {:.1f}, Beta2: {:.1f} -- Loss: {:.5f}\n'.format(b1, b2, Tloss)
                # f.write(write_line)
                # f.close()
if __name__ == "__main__":
    main()