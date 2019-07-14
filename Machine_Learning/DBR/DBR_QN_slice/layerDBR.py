import numpy as np
import tmm


def calR(s, Nslice, wavelength, nh, nl, high: bool):
    # list of layer thickness in nm
    d_list = [np.inf] + s.tolist() + [np.inf]
    # list of refractive indices
    nt = np.empty(s.shape)
    for i in range(Nslice):
        if high:  # high index material start
            if (i % 2) == 1:
                nt[i] = nl
            else:
                nt[i] = nh
        else:  # low index material start
            if (i % 2) == 1:
                nt[i] = nh
            else:
                nt[i] = nl
    n_list = [1] + nt.tolist() + [1]
    # initialize lists of y-values
    Rnorm = []
    for w in wavelength[0]:
        Rnorm.append(tmm.coh_tmm('s', n_list, d_list, 0, w)['R'])

    return Rnorm


def reward(R, tarwave, half_bandwidth, wavelength):
    lbound = tarwave - half_bandwidth
    ubound = tarwave + half_bandwidth
    lb_idx = np.where(wavelength == lbound)[0][0]
    ub_idx = np.where(wavelength == ubound)[0][0]

    R_in = np.mean(R[lb_idx:ub_idx+1])
    R_out = np.mean(np.hstack((R[0:lb_idx+1], R[ub_idx:])))

    return R_in / R_out


def step(s, action, INPUT_SIZE):
    if action == 2 * INPUT_SIZE:
        return s
    else:
        idx = int(action / 2)
        if action % 2 == 0:
            s[idx] = s[idx] - 5
        else:
            s[idx] = s[idx] + 5

        return s
