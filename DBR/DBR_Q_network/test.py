import numpy as np

s = np.array([1,0,0,0,0,1,1,0,0,1])
epsi = 12.25
eps0 = 1.

ei_element = epsi*(s == 1).astype(int)
e0_element = eps0*(s == 0).astype(int)
epst = ei_element + e0_element