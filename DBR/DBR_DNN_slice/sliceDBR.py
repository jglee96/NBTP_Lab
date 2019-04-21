import numpy as np


def calR(s, Nslice, wavelength, epsi, eps0, high: bool):
    epst = np.empty(s.shape)
    len_wavelength = len(wavelength[0])
    for i in range(Nslice):
        if high: # high index material start
            if (i%2) == 1:
                epst[i] = eps0
            else:
                epst[i] = epsi
        else: # low index material start
            if (i%2) == 1:
                epst[i] = epsi
            else:
                epst[i] = eps0

    Pn = np.hstack((epst,[1]))
    Pn1 = np.hstack(([1],epst))
    P = Pn/Pn1
    kx = np.sqrt(Pn)*2*np.pi/np.transpose(wavelength)
    h = np.hstack((np.array([i for i in s]),[0]))
    
    P = np.vstack([P for x in range(len_wavelength)]) # extend for wavelength dependent calcaulation
    h = np.vstack([h for x in range(len_wavelength)]) # extend for wavelength dependent calcaulation

    B11 = (1+P)*np.exp(-1j*kx*h)
    B12 = (1-P)*np.exp(1j*kx*h)
    B21 = (1-P)*np.exp(-1j*kx*h) 
    B22 = (1+P)*np.exp(1j*kx*h)

    R = np.empty(len_wavelength)
    for w in range(len_wavelength):
        Btot = np.eye(2)
        for i in range(Nslice+1):
            Bt = (0.5)*np.array([[B11[w,i],B12[w,i]],[B21[w,i],B22[w,i]]])
            Btot = np.matmul(Btot,Bt)
        R[w] = np.abs(Btot[1,0]/Btot[0,0])**2

    return R