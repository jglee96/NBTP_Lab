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

PATH = 'D:/NBTP_Lab/Machine_Learning/2Dsplitter/2Dsplitter_ARLA'
TRAIN_PATH = PATH + '/trainset/12'

def getData():
    # Load Training Data
    print("========      Load Data     ========")

    Xintarray = []

    '''
    sname = TRAIN_PATH + '/index.csv'
    Xtemp = pd.read_csv(sname, header=None, dtype=object)
    Xstrarray = Xtemp.values
    for i in range(Xstrarray.shape[0]):
        temp = Xstrarray[i][0]
        # temp = temp.replace(str(i+1), "", 1)
        temp = list(map(int, temp))
        Xintarray.append(np.array(temp))
    sX = np.asarray(Xintarray)
    '''

    sname = TRAIN_PATH + '/index.csv'
    Xtemp = pd.read_csv(sname, header=None, delimiter=",")
    sX = Xtemp.values

    port1_name = TRAIN_PATH + '/PORT1result_total.csv'
    port1 = pd.read_csv(port1_name, header=None, delimiter=",")
    P1 = port1.values

    port2_name = TRAIN_PATH + '/PORT2result_total.csv'
    port2 = pd.read_csv(port2_name, header=None, delimiter=",")
    P2 = port2.values
    # P2 = -P2 # vertical splitter

    port3_name = TRAIN_PATH + '/PORT3result_total.csv'
    port3 = pd.read_csv(port3_name, header=None, delimiter=",")
    P3 = port3.values

    wavelength = P1[0,:]
    # P1 = np.delete(P1, 0, 0)

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

    r = 0.7
    pr = 0.5
    center_fact = ((P2min + P2max)/2 + (P3min + P3max)/2)/2
    # center_fact1 = (P2min + P2max)/2
    # center_fact2 = (P3min + P3max)/2
    rP2 = (rP2 - P2min)/(P2max - P2min)
    rP3 = (rP3 - P3min)/(P3max - P3min)
    FOM = r*(pr*(rP2-center_fact) + (1-pr)*(rP3-center_fact)) - (1-r)*abs(rP2 - rP3) + rP1# linear
    # FOM = r*(pr*(rP2-center_fact1) + (1-pr)*(rP3-center_fact2)) - (1-r)*abs(rP2 - rP3) - (1-rP1)# linear
    # FOM = (rP2-0.5) + (rP3-0.5) - 0.5*abs(rP2-rP3)
    print("FOM Calculation Success!!")
    FOM_temp = np.reshape(FOM, newshape=(-1, 1))
    rX = X * FOM_temp
    rX = np.sum(rX, axis=0)

    minX = np.min(rX)
    rX = rX - minX
    avgX = np.mean(rX)

    result_state = (rX > avgX).astype(int)
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
