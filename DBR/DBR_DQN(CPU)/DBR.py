import numpy as np


def calR(s,Ngrid,wavelength,dx,epsi,eps0):
    ei_element = epsi*(s == 1).astype(int)
    e0_element = eps0*(s == 0).astype(int)
    epst = ei_element + e0_element

    Pn = np.hstack((epst,[1]))
    Pn1 = np.hstack(([1],epst))
    P = Pn/Pn1
    kx = np.sqrt(Pn)*2*np.pi/np.transpose(wavelength)
    h = np.hstack((np.array([dx for i in range(Ngrid)]),[0]))
    
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
    Qfac = -2500
    MSL = 1
    return (Qfac,MSL)

def calFWHM(R,wavelength,tarwave,ratio):
    taridx = np.where(wavelength == tarwave)[1][0]
    tarint = R[taridx]
    
    tarhi = list(i for i in range(taridx,wavelength.shape[1],1) if R[i] < ratio*tarint)
    tarlo = list(i for i in range(taridx,0,-1) if R[i] < ratio*tarint)

    return (tarhi,tarlo)

def calQfac(R,wavelength,tarwave):
    taridx = np.where(wavelength == tarwave)[1][0]

    tarhi,tarlo = calFWHM(R,wavelength,tarwave,0.5)
    if (tarhi) and (tarlo):
        tarhi = tarhi[0]
        tarlo = tarlo[0]
        Qfac = (1/wavelength[0,taridx])/(1/wavelength[0,tarlo]-1/wavelength[0,tarhi])
    else:
        Qfac,MSL = failreward()
    
    return Qfac

def reward(Ngrid,wavelength,R,tarwave):
    taridx = np.where(wavelength == tarwave)[1][0]
    tarint = R[taridx]
    done = True

    ratio = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    for i in ratio:
        tarhi,tarlo = calFWHM(R,wavelength,tarwave,i)
        if (tarhi) and (tarlo):
            tarhi = tarhi[0]
            tarlo = tarlo[0]
            Qfac = (1/i)*tarint*(1/wavelength[0,taridx])/(1/wavelength[0,tarlo]-1/wavelength[0,tarhi])
            MSL = np.mean(np.hstack((R[0:tarlo+1],R[tarhi:])))
            done = False
            break
    
    if done:
        Qfac,MSL = failreward()
        
    reward = (Qfac/MSL)/3000
    
    return reward


def step(s,a,Ngrid):
    
    s1 = np.copy(s)
    
    if a < Ngrid:
        s1[a] = 0
    else:
        s1[a-Ngrid] = 1

    return s1