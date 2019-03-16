import numpy as np


def calR(s,Ngrid,wavelength,dx,epsi,eps0):
    ei_element = epsi*(s == 1).astype(int)
    e0_element = eps0*(s == 0).astype(int)
    epst = ei_element + e0_element

    Pn = np.hstack((epst,[[1]]))
    Pn1 = np.hstack(([[1]],epst))
    P = Pn/Pn1
    kx = np.sqrt(Pn)*2*np.pi/np.transpose(wavelength)
    h = np.hstack(([np.array([dx for i in range(Ngrid)])],[[0]]))
    
    P = np.vstack([P for x in range(len(wavelength))]) # extend for wavelength dependent calcaulation
    h = np.vstack([h for x in range(len(wavelength))]) # extend for wavelength dependent calcaulation

    B11 = (1+P)*np.exp(-1j*kx*h)
    B12 = (1-P)*np.exp(1j*kx*h)
    B21 = (1-P)*np.exp(-1j*kx*h) 
    B22 = (1+P)*np.exp(1j*kx*h)

    R = np.empty(len(wavelength[0]))
    for w in range(len(wavelength[0])):
        Btot = np.eye(2)
        for i in range(Ngrid+1):
            Bt = (0.5)*np.array([[B11[w,i],B12[w,i]],[B21[w,i],B22[w,i]]])
            Btot = np.matmul(Btot,Bt)
        R[w] = np.abs(Btot[1,0]/Btot[0,0])**2

    return R

def failreward():
    Qfac = -1
    MSL = 1
    return (Qfac,MSL)

def reward(s,Ngrid,wavelength,R,tarwave):
    taridx = np.where(wavelength == tarwave)[0][0]
    tarint = R[taridx]
    try:
        tarhi = list(i for i in range(taridx,wavelength[-1]) if R[i] < 0.5*tarint)[0]
        tarlo = list(i for i in range(taridx,0) if R[i] < 0.5*tarint)[0]

        if tarlo == tarhi:
            Qfac,MSL = failreward()

        else:
            Qfac = tarint*(1/wavelength(taridx))/(1/wavelength(tarlo)-1/wavelength(tarhi))
            MSL = np.mean(np.hstack((R[0:tarlo+1],R[tarlhi:])))            

    except:
        Qfac,MSL = failreward()
    
    reward = Qfac/MSL
    return reward


def step(s,Ngrid,a,dupcnt):
    # action 0: stay
    # action 1: go to opposite (0 -> 1, 1 -> 0) -> XOR
    
    done = False
    s1 = np.copy(s)

    if s1[0,a] == 1:
        s1[0,a] = 0
    else:
        s1[0,a] = 1
    
    if np.array_equal(s,s1):
        dupcnt = dupcnt+1
        if dupcnt == 10:
            done = True

    else:
        dupcnt = 0

    return (s1,done,dupcnt)