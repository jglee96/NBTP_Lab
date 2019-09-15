import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N_pixel = 20

# Base data
print("======== Design Information ========")
print('nh: Si, nl: Air')

PATH = 'D:/NBTP_Lab/Machine_Learning/2Dsplitter/1x3_splitter'
TRAIN_PATH = PATH + '/trainset/02'


def getData():
    # Load Training Data
    print("========      Load Data     ========")

    sname = TRAIN_PATH + '/index.csv'
    Xtemp = pd.read_csv(sname, header=None, delimiter=",")
    sX = Xtemp.values

    port1_name = TRAIN_PATH + '/PORT1result_total.csv'
    port1 = pd.read_csv(port1_name, header=None, delimiter=",")
    P1 = port1.values

    port2_name = TRAIN_PATH + '/PORT2result_total.csv'
    port2 = pd.read_csv(port2_name, header=None, delimiter=",")
    P2 = port2.values

    port3_name = TRAIN_PATH + '/PORT3result_total.csv'
    port3 = pd.read_csv(port3_name, header=None, delimiter=",")
    P3 = port3.values

    port4_name = TRAIN_PATH + '/PORT4result_total.csv'
    port4 = pd.read_csv(port4_name, header=None, delimiter=",")
    P4 = port4.values

    return sX, P1, P2, P3, P4


def broad_reward(R):
    min_R = []
    max_R = []
    for i in range(R.shape[0]):
        min_R.append(np.min(R[i, :]))
        max_R.append(np.min(R[i, :]))
    min_R = np.asarray(min_R)
    max_R = np.asarray(max_R)
    return min_R, max_R


def main():
    X, P1, P2, P3, P4 = getData()
    print("Load Data Success!!")

    R_min, R_max = broad_reward(P1)
    T1_min, T1_max = broad_reward(P2)
    T2_min, T2_max = broad_reward(P3)
    T3_min, T3_max = broad_reward(P4)

    # FOM = T1_min + T2_min + T3_min + R_min
    FOM = T1_min + T2_min + T3_min + R_min - 0.2*abs(T1_min-T2_min)
    # FOM = T1_min + T2_min + T3_min + R_min - abs(T2_min-T2_max)
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
