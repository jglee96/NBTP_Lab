class DBR:
    def calR(s,Ngrid,wavelength):
        ei_element = epsi*tf.cast((s == 1),tf.int8)
        e0_element = eps0*tf.cast((s == 0),tf.int8)
        epst = ei_element + e0_element

        Pn = np.hstack(epst,[1])
        Pn1 = np.hstack([1],epst)
        P = Pn/Pn1
        kx = np.zeros(shape=[len(wavelength),Ngrid+1],dtype=float)

        for l in range(len(wavelrngth)):
            kx = 

        return R



    def reward(s,Ngrid,R):

        return reward


    def step(s,Ngrid,a):

        return s1,done