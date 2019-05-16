import numpy as np
import tmm


def calR(s, N_layer, wavelength, nh, nl, high: bool):
    # list of layer thickness in nm
    d_list = [np.inf] + s.tolist() + [np.inf]
    # list of refractive indices
    nt = np.empty(s.shape)
    for i in range(N_layer):
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


def step(R, tarwave, half_bandwidth, wavelength, s, action, INPUT_SIZE, lbound, ubound):
    done = False

    lband = tarwave - half_bandwidth
    uband = tarwave + half_bandwidth
    lb_idx = np.where(wavelength == lband)[0][0]
    ub_idx = np.where(wavelength == uband)[0][0]

    R_in = np.mean(R[lb_idx:ub_idx+1])
    R_out = np.mean(np.hstack((R[0:lb_idx+1], R[ub_idx:])))

    reward = R_in / R_out

    if action == 2 * INPUT_SIZE:
        next_state = s
        done = True
    else:
        idx = int(action / 2)
        if action % 2 == 0:
            s[idx] = s[idx] - 5
            if s[idx] < lbound:
                s[idx] = s[idx] + 5
                reward = -5
        else:
            s[idx] = s[idx] + 5
            if s[idx] > ubound:
                s[idx] = s[idx] - 5
                reward = -5
        next_state = s
    
    return next_state, reward, done
