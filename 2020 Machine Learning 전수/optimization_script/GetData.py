import numpy as np
import matplotlib.pyplot as plt
import imp
lumapi = imp.load_source("lumapi", "C:\\Program Files\\Lumerical\\2020a\\api\\python\\lumapi.py")

# simulation Info.
PATH = "D:\\FDTD_file\\LJG\\DBR\\"


def main():
    fdtd = lumapi.FDTD()

    try:
        fdtd.load(PATH + "DBR_single.fsp")
    except:
        fdtd = lumapi.FDTD()
        fdtd.load(PATH + "DBR_single.fsp")


    fdtd.switchtolayout
    fdtd.run()

    fdtd.save()

    # T results
    T = fdtd.getresult("monitor_T", "T")
    R = fdtd.getresult("monitor_R", "T")
    freq = R["f"]

    with open(PATH + "result_T.csv", "a") as sf:
        np.savetxt(sf, np.reshape(freq, (1, -1)), fmt='%.8f', delimiter=',')
        np.savetxt(sf, np.reshape(T["T"], (1, -1)), fmt='%.8f', delimiter=',')
    with open(PATH + "result_R.csv", "a") as sf:
        np.savetxt(sf, np.reshape(freq, (1, -1)), fmt='%.8f', delimiter=',')
        np.savetxt(sf, np.reshape(R["T"], (1, -1)), fmt='%.8f', delimiter=',')
    
    fdtd.close()

    plt.plot(freq*(10**-12), T["T"])
    plt.plot(freq*(10**-12), -R["T"])
    plt.xlabel("Frequency [THz]")
    plt.ylabel("Transmission")
    plt.show()


if __name__ == "__main__":
    main()