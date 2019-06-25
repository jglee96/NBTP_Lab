import numpy as np


def band_reward(R, tarwave, wavelength, bandwidth):
    lband = tarwave - (bandwidth / 2)
    uband = tarwave + (bandwidth / 2)
    lb_idx = np.where(abs(wavelength - lband) <= 1E-6)[0][0]
    ub_idx = np.where(abs(wavelength - uband) <= 1E-6)[0][0]

    R_in = np.mean(R[:, ub_idx:lb_idx+1], axis=1)
    R_out = np.mean(np.hstack((R[:, 0:ub_idx+1], R[:, lb_idx:])), axis=1)

    # return  R_in * (1 - R_out)
    return R_in / R_out

def target_reward(R, tarwave, wavelength):
    target_idx = np.where(abs(wavelength - tarwave) <= 1E-6)[0][0]

    R_target = R[:, target_idx]

    # return  R_in * (1 - R_out)
    return R_target