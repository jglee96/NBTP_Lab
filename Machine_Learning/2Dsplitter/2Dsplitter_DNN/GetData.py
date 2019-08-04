import pandas as pd
import numpy as np

PATH = 'D:/NBTP_Lab/Machine_Learning/2Dsplitter/2Dsplitter_DNN/trainset/13'
Nfile1 = 0
# Nfile2 = 40-Nfile1
Nfile2 = 10

def getData1():
    # Load Training Data
    print("========      Load Data1     ========")
    for i in range(Nfile1):
        fPATH = PATH + '/' + str(i)
        Xintarray = []
        sname = fPATH + '/index.txt'
        Xtemp = pd.read_csv(sname, header=None, dtype=object)
        Xstrarray = Xtemp.values
        for j in range(Xstrarray.shape[0]):
            temp = Xstrarray[j][0]
            temp = temp.replace(str(j), "", 1)
            temp = list(map(int, temp))
            Xintarray.append(np.array(temp))
        sX = np.asarray(Xintarray)

        port1_name = fPATH + '/PORT1result.csv'
        port1 = pd.read_csv(port1_name, header=None, delimiter=",")
        P1 = port1.values
        wavelength = P1[0,:]
        P1 = np.delete(P1, 0, 0)

        port2_name = fPATH + '/PORT2result.csv'
        port2 = pd.read_csv(port2_name, delimiter=",")
        P2 = port2.values
        P2 = np.delete(P2, np.s_[1:wavelength.shape[0]], 1)
        P2 = np.reshape(np.transpose(P2), (-1, wavelength.shape[0]))

        port3_name = fPATH + '/PORT3result.csv'
        port3 = pd.read_csv(port3_name, delimiter=",")
        P3 = port3.values
        P3 = np.delete(P3, np.s_[1:wavelength.shape[0]], 1)
        P3 = np.reshape(np.transpose(P3), (-1, wavelength.shape[0]))

        with open(PATH + '/index.csv', 'a') as fname:
            np.savetxt(fname, sX, fmt='%d', delimiter=',')
        with open(PATH + '/PORT1result_total.csv', 'a') as fname:
            np.savetxt(fname, P1, fmt='%.8f', delimiter=',')
        with open(PATH + '/PORT2result_total.csv', 'a') as fname:
            np.savetxt(fname, P2, fmt='%.8f', delimiter=',')
        with open(PATH + '/PORT3result_total.csv', 'a') as fname:
            np.savetxt(fname, P3, fmt='%.8f', delimiter=',')

def getData2():
    # Load Training Data
    print("========      Load Data2     ========")
    for i  in range(Nfile2):
        idx = i + Nfile1
        fPATH = PATH + '/' + str(idx)
        Xintarray = []
        sname = fPATH + '/index.txt'
        Xtemp = pd.read_csv(sname, header=None, dtype=object)
        Xstrarray = Xtemp.values
        for j in range(Xstrarray.shape[0]):
            temp = Xstrarray[j][0]
            temp = temp.replace(str(j), "", 1)
            temp = list(map(int, temp))
            Xintarray.append(np.array(temp))
        sX = np.asarray(Xintarray)

        port1_name = fPATH + '/PORT1result.csv'
        port1 = pd.read_csv(port1_name, delimiter=",")
        P1 = port1.values

        port2_name = fPATH + '/PORT2result.csv'
        port2 = pd.read_csv(port2_name, delimiter=",")
        P2 = port2.values

        port3_name = fPATH + '/PORT3result.csv'
        port3 = pd.read_csv(port3_name, delimiter=",")
        P3 = port3.values

        # Triple
        '''
        port4_name = fPATH + '/PORT4result.csv'
        port4 = pd.read_csv(port4_name, delimiter=",")
        P4 = port4.values
        '''

        with open(PATH + '/index.csv', 'a') as fname:
            np.savetxt(fname, sX, fmt='%d', delimiter=',')
        with open(PATH + '/PORT1result_total.csv', 'a') as fname:
            np.savetxt(fname, P1, fmt='%.8f', delimiter=',')
        with open(PATH + '/PORT2result_total.csv', 'a') as fname:
            np.savetxt(fname, P2, fmt='%.8f', delimiter=',')
        with open(PATH + '/PORT3result_total.csv', 'a') as fname:
            np.savetxt(fname, P3, fmt='%.8f', delimiter=',')
        # Triple
        # with open(PATH + '/PORT4result_total.csv', 'a') as fname:
        #     np.savetxt(fname, P4, fmt='%.8f', delimiter=',')

def getData3():
    # Load Training Data
    print("========      Load Data3     ========")
    for i  in range(Nfile2):
        idx = i + Nfile1
        fPATH = PATH + '/' + str(idx)

        port1_name = fPATH + '/PORT1result.csv'
        port1 = pd.read_csv(port1_name, delimiter=",")
        P1 = port1.values

        port2_name = fPATH + '/PORT2result.csv'
        port2 = pd.read_csv(port2_name, delimiter=",")
        P2 = port2.values

        port3_name = fPATH + '/PORT3result.csv'
        port3 = pd.read_csv(port3_name, delimiter=",")
        P3 = port3.values

        with open(PATH + '/07_PORT1result_total.csv', 'a') as fname:
            np.savetxt(fname, P1, fmt='%.8f', delimiter=',')
        with open(PATH + '/07_PORT2result_total.csv', 'a') as fname:
            np.savetxt(fname, P2, fmt='%.8f', delimiter=',')
        with open(PATH + '/07_PORT3result_total.csv', 'a') as fname:
            np.savetxt(fname, P3, fmt='%.8f', delimiter=',')


def getData4():
    # Load Training Data
    print("========      Load Data3     ========")
    for i  in range(Nfile2):
        idx = i + Nfile1
        fPATH = PATH + '/' + str(idx)

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


def getIndex():
    # Load Training Data
    print("========      Load Index     ========")

    Xintarray = []
    sname = PATH + '/index.txt'
    Xtemp = pd.read_csv(sname, header=None, dtype=object)
    Xstrarray = Xtemp.values
    for j in range(Xstrarray.shape[0]):
        temp = Xstrarray[j][0]
        temp = list(map(int, temp))
        Xintarray.append(np.array(temp))
    sX = np.asarray(Xintarray)

    with open(PATH + '/index.csv', 'a') as fname:
        np.savetxt(fname, sX, fmt='%d', delimiter=',')


def getData_combine():
    # Load Training Data
    print("========      Load Data     ========")
    for i  in range(5):
        idx = i
        fPATH = PATH + '/' + str(idx)

        # X_name = fPATH + '_index.csv'
        # X = pd.read_csv(X_name, delimiter=",")
        # sX = X.values
        Xintarray = []
        sname = fPATH + '_index.txt'
        Xtemp = pd.read_csv(sname, header=None, dtype=object)
        Xstrarray = Xtemp.values
        for j in range(Xstrarray.shape[0]):
            temp = Xstrarray[j][0]
            temp = list(map(int, temp))
            Xintarray.append(np.array(temp))
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

        with open(PATH + '/index_total.csv', 'a') as fname:
            np.savetxt(fname, sX, fmt='%d', delimiter=',')
        with open(PATH + '/PORT1result_total.csv', 'a') as fname:
            np.savetxt(fname, P1, fmt='%.8f', delimiter=',')
        with open(PATH + '/PORT2result_total.csv', 'a') as fname:
            np.savetxt(fname, P2, fmt='%.8f', delimiter=',')
        with open(PATH + '/PORT3result_total.csv', 'a') as fname:
            np.savetxt(fname, P3, fmt='%.8f', delimiter=',')

if __name__=="__main__":
    # getData1()
    # getData2()
    # getData3()
    # getData4()
    # getIndex()
    getData_combine()