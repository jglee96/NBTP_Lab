import numpy as np

class DBR:
    def calR(s,Ngrid,wavelength,dx):
        ei_element = epsi*tf.cast((s == 1),tf.int8)
        e0_element = eps0*tf.cast((s == 0),tf.int8)
        epst = ei_element + e0_element

        Pn = np.hstack(epst,[1])
        Pn1 = np.hstack([1],epst)
        P = Pn/Pn1
        kx = np.sqrt(Pn)*2*np.pi/np.transpose(wavelength)
        h = np.hstack(np.array([dx for i in range(Ngrid)]),[0])
        
        P = np.vstack([P for x in range(len(wavelength))]) # extension for wavelength dependent calcaulation
        h = np.vstack([h for x in range(len(wavelrngth))]) # extension for wavelength dependent calcaulation

        B11 = (1+P)*np.exp(-j*kx*h)
        B12 = (1-P)*np.exp(j*kx*h)
        B21 = (1-P)*np.exp(-j*kx*h) 
        B22 = (1+P)*np.exp(j*kx*h)

    def failreward():
        Qfac = -1
        MSL = 1
        return (Qfac,MSL)

    def reward(s,Ngrid,wavelength,R,tarwave):
        taridx = wavelength[wavelength == tarwave]
        tarint = R[taridx]
        try:
            tarhi = list(i for i in range(taridx,wavelength[-1]) if R[i] < 0.5*tarint)[0]
            tarlo = list(i for i in range(taridx,0) if R[i] < 0.5*tarint)[0]

            if tarlo == tarhi:
                Qfac,MSL = failreward()

            else:
                Qfac = tarint*(1/wavelength(taridx))/(1/wavelength(tarlo)-1/wavelength(tarhi))
                MSL = np.mean(R[0:tarlo+1, tarlhi:-1])            

        except:
            Qfac,MSL = failreward()
        
        reward = Qfac/MSL
        return reward


    def step(s,Ngrid,a):

        return (s1,done)