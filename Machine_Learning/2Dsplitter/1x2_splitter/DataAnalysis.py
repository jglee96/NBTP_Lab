import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TRAIN_PATH = 'D:/NBTP_Lab/Machine_Learning/2Dsplitter/1x2_splitter/trainset/04'
Nfile = 30


def getData(mode):
    # Load Training Data
    print("========      Load Data     ========")

    if mode == 'pack':
        sname = TRAIN_PATH + '/index.csv'
        Xtemp = pd.read_csv(sname, header=None, delimiter=",")
        sX = Xtemp.values

        port1_name = TRAIN_PATH + '/PORT1result_total.csv'
        port1 = pd.read_csv(port1_name, header=None, delimiter=",")
        P1 = port1.values

        port2_name = TRAIN_PATH + '/PORT2result_total.csv'
        port2 = pd.read_csv(port2_name, header=None, delimiter=",")
        P2 = port2.values
    elif mode == 'unpack':
        for n in range(Nfile):
            sname = TRAIN_PATH + '/' + str(n) + '_index.txt'
            Xintarray = []
            Xtemp = pd.read_csv(sname, header=None, delimiter=",")
            Xstrarray = Xtemp.values
            for j in range(Xstrarray.shape[0]):
                temp = Xstrarray[j][0]
                temp = list(map(int, temp))
                Xintarray.append(temp)
            tempX = np.asarray(Xintarray)

            port1_name = TRAIN_PATH + '/' + str(n) + '_PORT1result.csv'
            port1 = pd.read_csv(port1_name, delimiter=",")
            tempP1 = port1.values

            port2_name = TRAIN_PATH + '/' + str(n) + '_PORT2result.csv'
            port2 = pd.read_csv(port2_name, delimiter=",")
            tempP2 = port2.values

            if n == 0:
                sX = tempX
                P1 = tempP1
                P2 = tempP2
            else:
                sX = np.concatenate((sX, tempX), axis=0)
                P1 = np.concatenate((P1, tempP1), axis=0)
                P2 = np.concatenate((P2, tempP1), axis=0)

    Nsample = P1.shape[0]

    return P1, P2, Nsample


def Tstatic(pav, Nsample):
    tstack = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(Nsample):
        if pav[i] >= 0 and pav[i] < 0.1:
            tstack[0] += 1
        elif pav[i] >= 0.1 and pav[i] < 0.2:
            tstack[1] += 1
        elif pav[i] >= 0.2 and pav[i] < 0.3:
            tstack[2] += 1
        elif pav[i] >= 0.3 and pav[i] < 0.4:
            tstack[3] += 1
        elif pav[i] >= 0.4 and pav[i] < 0.5:
            tstack[4] += 1
        elif pav[i] >= 0.5 and pav[i] < 0.6:
            tstack[5] += 1
        elif pav[i] >= 0.6 and pav[i] < 0.7:
            tstack[6] += 1
        elif pav[i] >= 0.7 and pav[i] < 0.8:
            tstack[7] += 1
        elif pav[i] >= 0.8 and pav[i] < 0.9:
            tstack[8] += 1
        elif pav[i] >= 0.9 and pav[i] < 1.0:
            tstack[9] += 1

    return tstack


def main():
    _, P2, Nsample = getData(mode='unpack')

    P2av = np.average(P2, axis=1)

    Pt = Tstatic(2*P2av, Nsample)

    x = np.arange(10)
    Tvalues = ['0,1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
    plt.bar(x, Pt)
    plt.xticks(x, Tvalues)
    plt.show()


if __name__=="__main__":
    main()