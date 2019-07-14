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
TRAIN_PATH = PATH + '/trainset/04'

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

    return sX, P1, P2, P3


def main():
    X, P1, P2, P3 = getData()
    print("Load Data Success!!")

    rP1 = TwoCal.broad_reward(P1)
    rP2 = TwoCal.broad_reward(P2)
    rP3 = TwoCal.broad_reward(P3)

    P2min = np.min(rP2)
    P2max = np.max(rP2)
    P3min = np.min(rP3)
    P3max = np.max(rP3)
    rP1 = 1-rP1

    r = 1
    center_fact = ((P2min + P2max)/2 + (P3min + P3max)/2)/2
    FOM = r*((rP2-center_fact) + (rP3-center_fact)) - (1-r)*abs(rP2 - rP3)# linear
    # FOM = np.sqrt(((rP2 - center_fact)**2 + (rP3 - center_fact)**2))
    # FOM = rP3
    print("FOM Calculation Success!!", center_fact)
    FOM_temp = np.reshape(FOM, newshape=(-1, 1))
    rX = X * FOM_temp
    rX = np.sum(rX, axis=0)

    minX = np.min(rX)
    rX = rX - minX
    avgX = np.mean(rX)

    result_state = (rX > avgX).astype(int)
    print("Result Calculation Success!!")

    # result_state = np.transpose(np.reshape(result_state, newshape=(N_pixel, N_pixel))) # the simulation index order = (j, i)
    result_state = np.reshape(result_state, newshape=(N_pixel, int(N_pixel/2))) # the simulation index order = (i, j)
    print_result = result_state.tolist()
    print('[', end='')
    for i,  index1 in enumerate(print_result):
        for j, index2 in enumerate(index1):
            if j == (int(N_pixel/2) -1) and i != (N_pixel-1):
                print(str(index2) + ';')
            elif j == (int(N_pixel/2) -1) and i == (N_pixel-1):
                print(str(index2) + '];')
            else:
                print(str(index2) + ',', end='')

    plt.imshow(result_state, cmap='gray')
    plt.show()
    print("Print Result Success!!")


if __name__ == "__main__":
    main()
