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

Nsample = 14996
PATH = 'D:/NBTP_Lab/Machine Learning/2D splitter'
TRAIN_PATH = PATH + '/trainset/02'

def getData():
    # Load Training Data
    print("========      Load Data     ========")

    Xintarray = []

    sname = TRAIN_PATH + '/index.csv'
    Xtemp = pd.read_csv(sname, header=None, dtype=object)
    Xstrarray = Xtemp.values
    for i in range(Nsample):
        temp = Xstrarray[i][0]
        # temp = temp.replace(str(i+1), "", 1)
        temp = list(map(int, temp))
        Xintarray.append(np.array(temp))

    port1_name = TRAIN_PATH + '/PORT1result.csv'
    port1 = pd.read_csv(port1_name, header=None, delimiter=",")
    P1 = port1.values

    port2_name = TRAIN_PATH + '/PORT2result.csv'
    port2 = pd.read_csv(port2_name, header=None, delimiter=",")
    P2 = port2.values

    port3_name = TRAIN_PATH + '/PORT3result.csv'
    port3 = pd.read_csv(port3_name, header=None, delimiter=",")
    P3 = port3.values

    sX = np.asarray(Xintarray)

    wavelength = P1[0,:]
    P1 = np.delete(P1, 0, 0)
    P2 = np.delete(P2, 0, 0)
    P3 = np.delete(P3, 0, 0)

    return sX, P1, P2, P3, wavelength


def main():
    X, _, P2, P3, wavelength = getData()
    print("Load Data Success!!")

    # rP1 = TwoCal.reward(P1, tarwave, wavelength, bandwidth)
    rP2 = TwoCal.reward(P2, tarwave, wavelength, bandwidth)
    rP3 = TwoCal.reward(P3, tarwave, wavelength, bandwidth)

    r = 0.8
    FOM = r*(rP2 + rP3) - (1-r)*abs(rP2 - rP3)
    # FOM = 1/(rP2/rP3 + rP3/rP2)
    # FOM = rP2
    print("FOM Calculation Success!!")
    FOM_temp = np.reshape(FOM, newshape=(-1, 1))
    rX = X * FOM_temp
    rX = np.sum(rX, axis=0)

    minX = np.min(rX)
    rX = rX - minX
    avgX = np.mean(rX)

    result_state = (rX > avgX).astype(int)
    print("Result Calculation Success!!")

    result_state = np.transpose(np.reshape(result_state, newshape=(N_pixel, N_pixel))) # the simulation index order = (j, i)
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
