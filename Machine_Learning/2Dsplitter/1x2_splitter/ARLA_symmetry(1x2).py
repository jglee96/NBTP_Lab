import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N_pixel = 20

# Base data
print("======== Design Information ========")
print('nh: Si, nl: Air')

PATH = 'D:/NBTP_Lab/Machine_Learning/2Dsplitter/1x2_splitter'
TRAIN_PATH = PATH + '/trainset/07' # Thesis ARL traindata: 07
Nfile = 101


def getData(mode, header):
    # Load Training Data
    print("========      Load Data     ========")

    if mode == 'pack':
        sname = TRAIN_PATH + '/index.csv'
        Xtemp = pd.read_csv(sname, header=None, delimiter=",")
        sX = Xtemp.values

        port1_name = TRAIN_PATH + '/PORT1result.csv'
        port1 = pd.read_csv(port1_name, header=header, delimiter=",")
        P1 = port1.values

        port2_name = TRAIN_PATH + '/PORT2result.csv'
        port2 = pd.read_csv(port2_name, header=header, delimiter=",")
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
            # file error check
            if n == 0:
                pre_len_check = Xstrarray.shape[0]
            len_check = Xstrarray.shape[0]
            if pre_len_check != len_check:
                print(n, len_check)
            pre_len_check = len_check

            port1_name = TRAIN_PATH + '/' + str(n) + '_PORT1result.csv'
            port1 = pd.read_csv(port1_name, header=header, delimiter=",")
            tempP1 = port1.values

            port2_name = TRAIN_PATH + '/' + str(n) + '_PORT2result.csv'
            port2 = pd.read_csv(port2_name, header=header, delimiter=",")
            tempP2 = port2.values

            if n == 0:
                sX = tempX
                P1 = tempP1
                P2 = tempP2
            else:
                sX = np.concatenate((sX, tempX), axis=0)
                P1 = np.concatenate((P1, tempP1), axis=0)
                P2 = np.concatenate((P2, tempP2), axis=0)

    return sX, P1, P2


def broad_reward(R):
    min_R = []
    max_R = []
    mean_R = []
    for i in range(R.shape[0]):
        min_R.append(np.min(R[i, :]))
        max_R.append(np.max(R[i, :]))
        mean_R.append(np.mean(R[i, :]))
    min_R = np.asarray(min_R)
    max_R = np.asarray(max_R)
    mean_R = np.asarray(mean_R)
    return min_R, max_R, mean_R


def main():
    X, P1, P2 = getData(mode='pack', header=None)
    print("Load Data Success!!")

    R_min, R_max, R_mean = broad_reward(P1)
    T1_min, T1_max, T1_mean = broad_reward(P2)

    # FOM = T1_min
    # FOM = T1_min + R_min
    # FOM = T1_mean + R_mean
    # FOM = T1_mean

    Tmin = np.min(T1_min)
    Tmax = np.max(T1_min)

    center_fact = (Tmin + Tmax)/2
    # FOM = T1_min - center_fact
    FOM = T1_min - center_fact

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
