import pandas as pd
import numpy as np

PATH = 'D:/NBTP_Lab/Machine_Learning/2Dsplitter/1x2_splitter/trainset'
Nfile = 20


def getData():
    # Load Training Data
    print("========      Load Data     ========")
    for i in range(Nfile):
        fPATH = PATH + '/' + str(i)
        Xintarray = []
        sname = fPATH + '_index.txt'
        Xtemp = pd.read_csv(sname, header=None, dtype=object)
        Xstrarray = Xtemp.values
        if 1:
            for j in range(Xstrarray.shape[0]):
                temp = Xstrarray[j][0]
                temp = list(map(int, temp))
                Xintarray.append(temp)
            sX = np.asarray(Xintarray)
        else:
            for j in range(Xstrarray.shape[0]):
                if (j % 2) == 1:
                    temp = Xstrarray[j][0]
                    temp = list(map(int, temp))
                    Xintarray.append(temp)
            sX = np.asarray(Xintarray)

        port1_name = fPATH + '_PORT1result.csv'
        port1 = pd.read_csv(port1_name, delimiter=",")
        P1 = port1.values

        port2_name = fPATH + '_PORT2result.csv'
        port2 = pd.read_csv(port2_name, delimiter=",")
        P2 = port2.values

        port3_name = fPATH + '_PORT3result.csv'
        port3 = pd.read_csv(port3_name, delimiter=",")
        P3 = port3.values

        with open(PATH + '/index.csv', 'a') as fname:
            np.savetxt(fname, sX, fmt='%d', delimiter=',')
        with open(PATH + '/PORT1result_total.csv', 'a') as fname:
            np.savetxt(fname, P1, fmt='%.8f', delimiter=',')
        with open(PATH + '/PORT2result_total.csv', 'a') as fname:
            np.savetxt(fname, P2, fmt='%.8f', delimiter=',')
        with open(PATH + '/PORT3result_total.csv', 'a') as fname:
            np.savetxt(fname, P3, fmt='%.8f', delimiter=',')


def getIndex():
    # Load Training Data
    print("========      Load Index     ========")

    Xintarray = []
    sname = PATH + '/index.txt'
    Xtemp = pd.read_csv(sname, header=None, dtype=object)
    Xstrarray = Xtemp.values
    for j in range(Xstrarray.shape[0]):
        # if (j % 2) == 1:  # only use when fdtd.runsetup() double play
        temp = Xstrarray[j][0]
        temp = list(map(int, temp))
        Xintarray.append(np.array(temp))
    sX = np.asarray(Xintarray)

    with open(PATH + '/index.csv', 'a') as fname:
        np.savetxt(fname, sX, fmt='%d', delimiter=',')


def getT():
    for i in range(Nfile):
        fPATH = PATH + '/' + str(i)

        port1_name = fPATH + '_PORT1result.csv'
        port1 = pd.read_csv(port1_name, delimiter=",")
        P1 = port1.values

        port2_name = fPATH + '_PORT2result.csv'
        port2 = pd.read_csv(port2_name, delimiter=",")
        P2 = port2.values

        port3_name = fPATH + '_PORT3result.csv'
        port3 = pd.read_csv(port3_name, delimiter=",")
        P3 = port3.values

        with open(PATH + '/PORT1result_total.csv', 'a') as fname:
            np.savetxt(fname, P1, fmt='%.8f', delimiter=',')
        with open(PATH + '/PORT2result_total.csv', 'a') as fname:
            np.savetxt(fname, P2, fmt='%.8f', delimiter=',')
        with open(PATH + '/PORT3result_total.csv', 'a') as fname:
            np.savetxt(fname, P3, fmt='%.8f', delimiter=',')


def getData_combine():
    # Load Training Data
    print("========      Load Data     ========")
    for i in range(2):
        idx = i + 2
        fPATH = PATH + '/' + str(idx)

        X_name = fPATH + '_index.csv'
        X = pd.read_csv(X_name, header=None, delimiter=",")
        sX = X.values
        # Xintarray = []
        # sname = fPATH + '_index.txt'
        # Xtemp = pd.read_csv(sname, header=None, dtype=object)
        # Xstrarray = Xtemp.values
        # for j in range(Xstrarray.shape[0]):
        #     temp = Xstrarray[j][0]
        #     temp = list(map(int, temp))
        #     Xintarray.append(np.array(temp))
        # sX = np.asarray(Xintarray)

        port1_name = fPATH + '_PORT1result.csv'
        port1 = pd.read_csv(port1_name, header=None, delimiter=",")
        P1 = port1.values

        port2_name = fPATH + '_PORT2result.csv'
        port2 = pd.read_csv(port2_name, header=None, delimiter=",")
        P2 = port2.values

        port3_name = fPATH + '_PORT3result.csv'
        port3 = pd.read_csv(port3_name, header=None, delimiter=",")
        P3 = port3.values

        with open(PATH + '/index.csv', 'a') as fname:
            np.savetxt(fname, sX, fmt='%d', delimiter=',')
        with open(PATH + '/PORT1result_total.csv', 'a') as fname:
            np.savetxt(fname, P1, fmt='%.8f', delimiter=',')
        with open(PATH + '/PORT2result_total.csv', 'a') as fname:
            np.savetxt(fname, P2, fmt='%.8f', delimiter=',')
        with open(PATH + '/PORT3result_total.csv', 'a') as fname:
            np.savetxt(fname, P3, fmt='%.8f', delimiter=',')

if __name__=="__main__":
    getData()
    # getIndex()
    # getT()
    # getData_combine()