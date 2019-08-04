import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TRAIN_PATH = 'D:/NBTP_Lab/Machine_Learning/2Dsplitter/2Dsplitter_DNN/trainset/12'


def getData():  # 03
    # Load Training Data
    print("========      Load Data     ========")

    port1_name = TRAIN_PATH + '/PORT1result_total.csv'
    # port1 = pd.read_csv(port1_name, header=None, delimiter=",")
    port1 = pd.read_csv(port1_name, delimiter=",")
    P1 = port1.values

    port2_name = TRAIN_PATH + '/PORT2result_total.csv'
    # port2 = pd.read_csv(port2_name, header=None, delimiter=",")
    port2 = pd.read_csv(port2_name, delimiter=",")
    P2 = port2.values

    port3_name = TRAIN_PATH + '/PORT3result_total.csv'
    # port3 = pd.read_csv(port3_name, header=None, delimiter=",")
    port3 = pd.read_csv(port3_name, delimiter=",")
    P3 = port3.values

    Nsample = P1.shape[0]
    x = np.arange(P1.shape[0])

    return P1, P2, P3, Nsample


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
    _, P2, P3, Nsample = getData()

    P2av = np.average(P2, axis=1)
    P3av = np.average(P3, axis=1)

    Pt = Tstatic(P2av+P3av, Nsample)

    x = np.arange(10)
    Tvalues = ['0,1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
    plt.bar(x, Pt)
    plt.xticks(x, Tvalues)
    plt.show()


if __name__=="__main__":
    main()