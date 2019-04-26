import numpy as np
import tmm

def calR(s, Nslice, wavelength, nh, nl, high: bool):
    # list of layer thickness in nm
    d_list = [np.inf] + s.tolist() + [np.inf]
    # list of refractive indices
    nt = np.empty(s.shape)
    for i in range(Nslice):
        if high: # high index material start
            if (i%2) == 1:
                nt[i] = nl
            else:
                nt[i] = nh
        else: # low index material start
            if (i%2) == 1:
                nt[i] = nh
            else:
                nt[i] = nl
    n_list = [1] + nt.tolist() + [1]
    # initialize lists of y-values
    Rnorm = []
    for w in wavelength[0]:
        Rnorm.append(tmm.coh_tmm('s', n_list, d_list, 0, w)['R'])

    return Rnorm