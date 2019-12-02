import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TRAIN_PATH = 'D:/NBTP_Lab/Machine_Learning/2Dsplitter/1x2_splitter/trainset/03'
Nfile = 101



def getData(mode):
    # Load Training Data
    print("========      Load Data     ========")

    if mode == 'pack':
        sname = TRAIN_PATH + '/index.csv'
        Xtemp = pd.read_csv(sname, header=None, delimiter=",")
        sX = Xtemp.values

        port1_name = TRAIN_PATH + '/PORT1result.csv'
        port1 = pd.read_csv(port1_name, delimiter=",")
        P1 = port1.values

        port2_name = TRAIN_PATH + '/PORT2result.csv'
        port2 = pd.read_csv(port2_name, delimiter=",")
        P2 = port2.values
    elif mode == 'unpack':
        for n in range(Nfile):
            try:
                sname = TRAIN_PATH + '/' + str(n) + '_index.txt'
                Xintarray = []
                Xtemp = pd.read_csv(sname, header=None, delimiter=",")
                Xstrarray = Xtemp.values
                for j in range(Xstrarray.shape[0]):
                    temp = Xstrarray[j][0]
                    temp = list(map(int, temp))
                    Xintarray.append(temp)
                tempX = np.asarray(Xintarray)
            except:
                sname = TRAIN_PATH + '/' + str(n) + '_index.csv'
                Xtemp = pd.read_csv(sname, header=None, delimiter=",")
                tempX = Xtemp.values

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
                P2 = np.concatenate((P2, tempP2), axis=0)

    Nsample = P1.shape[0]

    return P1, P2, Nsample


def Tstatic(pav, Nsample):
    tstack = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(Nsample):
        if pav[i] >= 0 and pav[i] < 0.05:
            tstack[0] += 1
        elif pav[i] >= 0.05 and pav[i] < 0.1:
            tstack[1] += 1
        elif pav[i] >= 0.1 and pav[i] < 0.15:
            tstack[2] += 1
        elif pav[i] >= 0.15 and pav[i] < 0.2:
            tstack[3] += 1
        elif pav[i] >= 0.2 and pav[i] < 0.25:
            tstack[4] += 1
        elif pav[i] >= 0.25 and pav[i] <= 0.3:
            tstack[5] += 1
        elif pav[i] >= 0.3 and pav[i] <= 0.35:
            tstack[6] += 1
        elif pav[i] >= 0.35 and pav[i] <= 0.4:
            tstack[7] += 1
        elif pav[i] >= 0.4 and pav[i] <= 0.45:
            tstack[8] += 1
        elif pav[i] >= 0.45 and pav[i] <= 0.5:
            tstack[9] += 1

    return tstack


def TdBstatic(pav, Nsample):
    tstack = [0, 0, 0, 0, 0]
    for i in range(Nsample):
        if pav[i] >= 3.0 and pav[i] < 4.0:
            tstack[0] += 1
        elif pav[i] >= 4.0 and pav[i] < 5.0:
            tstack[1] += 1
        elif pav[i] >= 5.0 and pav[i] < 6.0:
            tstack[2] += 1
        elif pav[i] >= 6.0 and pav[i] < 7.0:
            tstack[3] += 1
        else:
            tstack[4] += 1

    return tstack


def main():
    _, T, Nsample = getData(mode='unpack')

    Tmin = np.min(T, axis=1)
    Tmean = np.mean(T, axis=1)

    TdB = -10 * np.log10(T + 1e-7)
    TdBmin = np.min(TdB, axis=1)
    TdBmean = np.mean(TdB, axis=1)

    Tmin_bar = Tstatic(Tmin, Nsample)
    Tmean_bar = Tstatic(Tmean, Nsample)
    TdBmin_bar = TdBstatic(TdBmin, Nsample)
    TdBmean_bar = TdBstatic(TdBmean, Nsample)

    print("data number: ", Nsample)
    plt.figure(1)
    x = np.arange(10)
    Tvalues = ['0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45', '0.5']
    plt.bar(x, Tmin_bar)
    plt.xticks(x, Tvalues)

    with open(TRAIN_PATH + '/Data_distribution(min).csv', "a") as sf:
        np.savetxt(sf, np.reshape(np.arange(0.05, 0.55, 0.05), (1, len(Tvalues))), fmt='%.2f', delimiter=',')
    with open(TRAIN_PATH + '/Data_distribution(min).csv', "a") as sf:
        np.savetxt(sf, np.reshape(Tmin_bar, (1, len(Tmin_bar))), fmt='%d', delimiter=',')

    plt.figure(2)
    x = np.arange(10)
    Tvalues = ['0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45', '0.5']
    plt.bar(x, Tmean_bar)
    plt.xticks(x, Tvalues)

    plt.figure(3)
    x = np.arange(5)
    Tvalues = ['4.0', '5.0', '6.0', '7.0', 'Inf']
    plt.bar(x, TdBmin_bar)
    plt.xticks(x, Tvalues)

    plt.figure(4)
    x = np.arange(5)
    Tvalues = ['4.0', '5.0', '6.0', '7.0', 'Inf']
    plt.bar(x, TdBmean_bar)
    plt.xticks(x, Tvalues)

    plt.show()


if __name__ == "__main__":
    main()