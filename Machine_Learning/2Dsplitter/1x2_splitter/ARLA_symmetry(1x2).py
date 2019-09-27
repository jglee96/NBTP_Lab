import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N_pixel = 20

# Base data
print("======== Design Information ========")
print('nh: Si, nl: Air')

PATH = 'D:/NBTP_Lab/Machine_Learning/2Dsplitter/1x2_splitter'
TRAIN_PATH = PATH + '/trainset/03'
Nfile = 50


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

    return sX, P1, P2


def broad_reward(R):
    min_R = []
    max_R = []
    for i in range(R.shape[0]):
        min_R.append(np.min(R[i, :]))
        max_R.append(np.max(R[i, :]))
    min_R = np.asarray(min_R)
    max_R = np.asarray(max_R)
    return min_R, max_R


def main():
    X, P1, P2 = getData(mode='unpack')
    print("Load Data Success!!")

    R_min, R_max = broad_reward(P1)
    T1_min, T1_max = broad_reward(P2)

    FOM = T1_min + R_min
    FOM_temp = np.reshape(FOM, newshape=(-1, 1))
    rX = X * FOM_temp
    rX = np.sum(rX, axis=0)

    minX = np.min(rX)
    rX = rX - minX
    avgX = np.mean(rX)

    result_state = (rX > avgX).astype(int)
    print("Result Calculation Success!!")

    result_state = np.reshape(result_state, newshape=(N_pixel, int(N_pixel/2)))  # the simulation index order = (i, j)
    print_result = result_state.tolist()
    print('[', end='')
    for i,  index1 in enumerate(print_result):
        for j, index2 in enumerate(index1):
            if j == (int(N_pixel/2) - 1) and i != (N_pixel-1):
                print(str(index2) + ';')
            elif j == (int(N_pixel/2) - 1) and i == (N_pixel-1):
                print(str(index2) + '];')
            else:
                print(str(index2) + ',', end='')

    plt.imshow(result_state, cmap='gray')
    plt.show()
    print("Print Result Success!!")


if __name__ == "__main__":
    main()
