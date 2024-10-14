import numpy as np

def get_basis_tdep():
    """Get basis for TDEP structures"""
    
    with open('./infile.ucposcar', 'r') as f:
        f.readline()
        scaler = float(f.readline().rstrip().split()[0])
    basis = np.genfromtxt('./infile.ucposcar', skip_header=2, max_rows = 3)
    return basis*scaler

def read_forceconsts_second():
    """Reads second order forceconstants from the file"""
    
    basis_uc = get_basis_tdep()
    #print(basis_uc)
    with open('outfile.forceconstant', 'r') as f:
        Na = int(f.readline().rstrip().split()[0])
        f.readline()

        fct_Phi = {(i+1, j+1): [] for i in range(Na) for j in range(Na)}
        fct_R = {(i+1, j+1): [] for i in range(Na) for j in range(Na)}
        for i in range(Na):
            Nnn = int(f.readline().rstrip().split()[0])
            ind_1 = i + 1
            for j in range(Nnn):
                ind_2 = int(f.readline().rstrip().split()[0])

                # reading R
                R = np.array(f.readline().rstrip().split(), dtype = np.float64)
                R = R[0]*basis_uc[0] + R[1]*basis_uc[1] + R[2]*basis_uc[2]
                # reading forceconstant
                fct = np.array([f.readline().rstrip().split() for i in range(3)], dtype = np.float64)

                fct_R[(ind_1, ind_2)].append(R)
                fct_Phi[(ind_1, ind_2)].append(fct)
                #print(fct)

    fct_R = {key: np.array(fct_R[key]) for key in fct_R}
    fct_Phi = {key: np.array(fct_Phi[key]) for key in fct_Phi}
        
    return fct_R, fct_Phi, Na

def read_forceconsts_third(xtl):
    """Read thirdorder forceconsts from tdep file"""
    
    basis_uc = xtl.uc
    with open('outfile.forceconstant_thirdorder', 'r') as f:
        Na = xtl.Na
        f.readline()
        f.readline()

        fct_Phi3ph = []
        fct_R3ph = []
        fct_ind3ph = []
        for i in range(Na):
            Nnn = int(f.readline().rstrip().split()[0])
            for j in range(Nnn):
                ind_1 = int(f.readline().rstrip().split()[0])-1
                ind_2 = int(f.readline().rstrip().split()[0])-1
                ind_3 = int(f.readline().rstrip().split()[0])-1

                fct_ind3ph.append((ind_1, ind_2, ind_3))
                # reading R
                fct_R3ph.append(np.zeros((3, 3)))
                fct_R3ph_tmp = np.zeros((3, 3))
                for i in range(3):
                    R = np.array(f.readline().rstrip().split(), dtype = np.float64) 
                    fct_R3ph_tmp[i] = R[0]*basis_uc[0] + R[1]*basis_uc[1] + R[2]*basis_uc[2] + xtl.r[[ind_1, ind_2, ind_3][i]]
                
                # Making relative differences
                fct_R3ph[-1][0] = 0.0
                fct_R3ph[-1][1] = fct_R3ph_tmp[1] - fct_R3ph_tmp[0]
                fct_R3ph[-1][2] = fct_R3ph_tmp[2] - fct_R3ph_tmp[0]
                
                
                # reading forceconstant
                fct = np.array([f.readline().rstrip().split() for i in range(9)], dtype = np.float64).flatten()/(xtl.mass[ind_1+1]*xtl.mass[ind_2+1]*xtl.mass[ind_3+1])**(1/2.)

                fct_Phi3ph.append(fct)
            
    fct_Phi3ph = np.array(fct_Phi3ph)
    fct_R3ph = np.array(fct_R3ph)
    fct_ind3ph = np.array(fct_ind3ph, dtype = np.int32)
    fct_ind3ph = fct_ind3ph[:, 0] * Na*Na + fct_ind3ph[:, 1] * Na + fct_ind3ph[:, 2]
    
    return fct_ind3ph, fct_R3ph, fct_Phi3ph, Na