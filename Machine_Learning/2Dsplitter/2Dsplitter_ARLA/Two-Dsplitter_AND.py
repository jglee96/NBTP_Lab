import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Two_Dcalculator as TwoCal

N_pixel = 20

tarwave = 300e-6
bandwidth = 10e-6

# Base data
print("======== Design Information ========")
print('tarwave: {}um, nh: Si, nl: Air'.format(tarwave))

PATH = 'D:/NBTP_Lab/Machine Learning/2D splitter/2D splitter_ARLA'
TRAIN_PATH = PATH + '/trainset/05'

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
    P2 = -P2

    port3_name = TRAIN_PATH + '/PORT3result_total.csv'
    port3 = pd.read_csv(port3_name, header=None, delimiter=",")
    P3 = port3.values

    wavelength = P1[0,:]
    P1 = np.delete(P1, 0, 0)

    return sX, P1, P2, P3, wavelength


def main():
    X, P1, P2, P3, wavelength = getData()
    print("Load Data Success!!")
    rP1 = TwoCal.broad_reward(P1)
    rP2 = TwoCal.broad_reward(P2)
    rP3 = TwoCal.broad_reward(P3)

    P2min = np.min(rP2)
    P2max = np.max(rP2)
    P3min = np.min(rP3)
    P3max = np.max(rP3)

    FOM1 = rP2
    FOM2 = rP3

    print("FOM Calculation Success!!")
    # T1
    FOM1_temp = np.reshape(FOM1, newshape=(-1, 1))
    rX1 = X * FOM1_temp
    rX1 = np.sum(rX1, axis=0)

    minX1 = np.min(rX1)
    rX1 = rX1 - minX1
    avgX1 = np.mean(rX1)

    result_state1 = (rX1 > avgX1)

    # T2
    FOM2_temp = np.reshape(FOM2, newshape=(-1, 1))
    rX2 = X * FOM2_temp
    rX2 = np.sum(rX2, axis=0)

    minX2 = np.min(rX2)
    rX2 = rX2 - minX2
    avgX2 = np.mean(rX2)

    result_state2 = (rX2 > avgX2)
    result_state2 = np.logical_not(result_state2)

    result_state = (np.logical_and(result_state1, result_state2)).astype(int)
    print("Result Calculation Success!!")

    result_state = np.reshape(result_state, newshape=(N_pixel, N_pixel)) # the simulation index order = (i, j)
    print_result = result_state.tolist()
    print('[', end='')
    for i,  index1 in enumerate(print_result):
        for j, index2 in enumerate(index1):
            if j == (N_pixel -1) and i != (N_pixel-1):
                print(str(index2) + ';')
            elif j == (N_pixel -1) and i == (N_pixel-1):
                print(str(index2) + '];')
            else:
                print(str(index2) + ',', end='')

    plt.imshow(result_state, cmap='gray')
    plt.show()
    print("Print Result Success!!")


if __name__ == "__main__":
    main()
