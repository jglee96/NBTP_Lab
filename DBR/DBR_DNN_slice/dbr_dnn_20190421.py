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
nh = 2.6811 # TiO2 at 400 nm (Siefke)
nl = 1.4701 # SiO2 at 400 nm (Malitson)

minwave = 200
maxwave = 800
wavestep = 5
wavelength = np.array([np.arange(minwave,maxwave,wavestep)])
tarwave = 400

# Constants defining our neural network
INPUT_SIZE = Nslice
OUTPUT_SIZE = len(wavelength[0])

Nfile = 49
Nsample = 10000

# train set file name
FPATH = 'D:/NBTP_Lab/DBR/DBR_DNN_slice'
SAVE_PATH = FPATH +'/result/'
statefilename = '/trainset/Trainset03/state_trainset03'
Rfilename = '/trainset/Trainset03/R_trainset03'

def main():
    # Load Training Data
    Xarray = []
    Yarray = []
    for nf in range(Nfile):
        Xfname = FPATH+statefilename+'_'+str(nf)+'.txt'
        Xtemp = pd.read_csv(Xfname, names=['a'])
        Xtemp = Xtemp.values
        Xarray.append(Xtemp)

        Yfname = FPATH+Rfilename+'_'+str(nf)+'.txt'
        Ytemp = pd.read_csv(Yfname, names=['a'])
        Ytemp = Ytemp.values
        Yarray.append(Ytemp)
    
    Xcon = np.concatenate(Xarray)
    Ycon = np.concatenate(Yarray)

    sX = Xcon.reshape(-1,INPUT_SIZE)
    sY = Ycon.reshape(-1,OUTPUT_SIZE)

    # Load testing data
    Xtarray = []
    Ytarray = []
    Ntest = 1
    for nt in range(Ntest):
        Xtfname = FPATH+statefilename+'_'+str(nt+Nfile)
        Xttemp = pd.read_csv(Xtfname+'.txt', names=['a'])
        Xttemp = Xttemp.values
        Xtarray.append(Xttemp)
        
        Ytfname = FPATH+Rfilename+'_'+str(nt+Nfile)
        Yttemp = pd.read_csv(Ytfname+'.txt', names=['a'])
        Yttemp = Yttemp.values        
        Ytarray.append(Yttemp)
    
    Xtcon = np.concatenate(Xtarray)
    Ytcon = np.concatenate(Ytarray)
    
    stX = Xtcon.reshape(-1,INPUT_SIZE)
    stY = Ytcon.reshape(-1,OUTPUT_SIZE)
    
    # batch_size = [64]
    # num_layer = [2, 4, 6, 8]
    # num_neuron = [100, 150, 200, 250]
    # learning_rate = [1E-2, 1E-3, 1E-4]
    # 32, 10, 250, 1E-3 is best hyperparameter
    batch_size = [32]
    num_layer = [8]
    num_neuron = [1000]
    learning_rate = [1E-4]

    nfigure = 0
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

                        for n in range(int(Nsample*Nfile/bs)):
                            X = np.reshape(sX[n*bs:(n+1)*bs],[bs,INPUT_SIZE])
                            Y = np.reshape(sY[n*bs:(n+1)*bs],[bs,OUTPUT_SIZE])
                            mainDNN.update_train(X, Y, True)
                            if (n+1)%100 == 0:
                                summary = mainDNN.update_tensorboard(X, Y, True)
                                mainDNN.writer.add_summary(summary, global_step=n)
                                print(n+1,'th trained')
                        #Test
                        print("Testing Model...")
                        Tstate = np.random.randint(int(0.5*tarwave),size=Nslice)
                        # Tstate = np.ones(Nslice)
                        # Tstate = np.array([int(tarwave/(4*np.sqrt(epsi))),
                        #                     int(tarwave/4),
                        #                     int(tarwave/(4*np.sqrt(epsi))),
                        #                     int(tarwave/4),
                        #                     int(tarwave/(4*np.sqrt(epsi))),
                        #                     int(tarwave/4),
                        #                     int(tarwave/(4*np.sqrt(epsi)))])
                        TR = sliceDBR.calR(Tstate,Nslice,wavelength,nh,nl,True)
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
                        x = np.reshape(wavelength,wavelength.shape[1])
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
                        # fig2.savefig(fig2_name)

                        # f = open(FPATH+'/Loss Test(20190425157).txt', 'a')
                        # loss_average = sum(loss_List)/len(loss_List)
                        # loss_max = max(loss_List)
                        # write_line = 'Batch Size: {}, Num Layer: {}, Num Neuron: {}, Learning Rate: {:.5f} -- Average: {:.3f}, Max: {:.5f}\n'.format(bs, nl, nn, lr, loss_average, loss_max)
                        # f.write(write_line)
                        # f.close()
if __name__ == "__main__":
    main()