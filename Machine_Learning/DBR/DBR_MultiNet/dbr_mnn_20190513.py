import numpy as np
import tensorflow as tf
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import layerDBR
import mnn
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
OUTPUT_SIZE = wavelength.shape[1]

Nfile = 10
Ntest = 1
Nsample = 10000

# train set file name
FPATH = 'D:/NBTP_Lab/DBR/DBR_MultiNet'
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
    bs = 100
    nl = 4
    nn = 250
    lr = 1E-3

    print("========   Training Start   ========")
    nfigure = 0
    # Clear our computational graph
    MNN_list = []
    graph_list = []
    for w in range(OUTPUT_SIZE):
        graph_list.append(tf.Graph())
        with graph_list[w].as_default() as g:
            graph_name = 'Graph' + str(w)
            MNN_list.append(
                mnn.MNN(
                    input_size=INPUT_SIZE, batch_size=bs, num_layer=nl,
                    num_neuron=nn, learning_rate=lr, name='DBRNet' + str(w)))
            MNN_list[w].writer.add_graph(g)
        tf.reset_default_graph()

    # Prepare Test
    Tstate = stX[1]
    # TR = sliceDBR.calR(
    #     Tstate, Nslice, wavelength, nh, nl, True)
    Tstate = (Tstate - x_mean) / x_std
    TR = stY[1]
    tX = np.reshape(Tstate, [-1, INPUT_SIZE])
    NR = []
    Tloss = []

    for w in range(OUTPUT_SIZE):
        with tf.Session(graph=graph_list[w]) as sess:
            # Initialize Tensorflow variables
            sess.run(tf.global_variables_initializer())

            for n in range(int(Nsample*Nfile/bs)):
                X = np.reshape(
                    sX[n*bs:(n+1)*bs], [bs, INPUT_SIZE])
                Y = np.reshape(
                    sY[n*bs:(n+1)*bs, w], [bs, 1])
                MNN_list[w].update_train(sess, X, Y, True)
                if (n+1) % 500 == 0:
                    summary = MNN_list[w].update_tensorboard(
                        sess, X, Y, True)
                    MNN_list[w].writer.add_summary(
                        summary, global_step=n)
                    print('{}th trained'.format(n+1))

            print("Testing ", w, "th Model...")
            NR.append(MNN_list[w].Test_paly(sess, tX, False))
            sTR = np.reshape(TR[w], [-1, 1])
            Tloss.append(MNN_list[w].update_loss(sess, tX, sTR, False))

    TR = np.reshape(TR, [OUTPUT_SIZE, -1])
    NR = np.reshape(NR, [OUTPUT_SIZE, -1])
    # loss_List = []
    # for n in range(Nsample*Ntest):
    #     tX = np.reshape(stX[n], [-1,INPUT_SIZE])
    #     tY = np.reshape(stY[n], [-1,OUTPUT_SIZE])
    #     tloss = mainDNN.update_loss(tX, tY, False)
    #     loss_List.append(tloss)
    #     if (n+1)%1000 == 0:
    #         print(n+1,'th tested')
    avg_loss = sum(Tloss) / float(len(Tloss))
    print('LOSS: ', avg_loss)
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