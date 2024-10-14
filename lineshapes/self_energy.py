import numpy as np

from .utils import gauss, plank_distr
from .consts import h

def assign_to_large_array(cut_arr, w2, w3, plf1, plf2, E, tresh):
    """Assingning small part of array to large one"""
    
    buff = np.zeros(E.shape[0])
    
    for i, en in enumerate([- w2 - w3, w2+w3]):
    
        istart = E[E<=(en-tresh)].shape[0]
        iend = min(E[E<=(en+tresh)].shape[0]+1, istart + cut_arr.shape[0])
        #print(istart, iend)
        if istart != iend:
            if istart > 0 and iend < E.shape[0]:
                buff[istart:iend] += cut_arr*(-1)**(i+1)*plf1
            elif istart == 0:
                buff[istart:iend] += cut_arr[-(iend - istart):]*(-1)**(i+1)*plf1
            elif iend == E.shape[0]:
                buff[istart:iend] += cut_arr[:(iend - istart)]*(-1)**(i+1)*plf1
    
    for i, en in enumerate([w2 - w3, w3 - w2]):
    
        istart = E[E<=(en-tresh)].shape[0]
        iend = min(E[E<=(en+tresh)].shape[0]+1, istart + cut_arr.shape[0])
        #print(istart, iend)
        if istart != iend:
            if istart > 0 and iend < E.shape[0]:
                buff[istart:iend] += cut_arr*(-1)**(i)*plf2
            elif istart == 0:
                buff[istart:iend] += cut_arr[-(iend - istart):]*(-1)**(i)*plf2
            elif iend == E.shape[0]:
                buff[istart:iend] += cut_arr[:(iend - istart)]*(-1)**(i)*plf2

    return buff

def self_energy_img(eig_val_2_grid, eig_val_3_grid, Phi_q1_qrid, E, T, sigma = 1/4.):
    """Computes imaginary part of self energy (Gamma) in THz
    """
    
    Nmodes = eig_val_2_grid.shape[1]
    Nq = eig_val_2_grid.shape[0]
    gamma = np.zeros((Nmodes, Nq, Nmodes, E.shape[0]),dtype = np.float64)
    
    Phi_q1_qrid_abs = np.abs(Phi_q1_qrid)**2
    
    dE = E[1] - E[0]
    tresh = sigma*4
    
    gauss_arr = gauss(sigma, dE, tresh)
    
    for i in range(Nq):
        for j in range(Nmodes):
            for k in range(Nmodes):
                for l in range(Nmodes):
                    w2 = eig_val_2_grid[i, k]
                    w3 = eig_val_3_grid[i, l]
                    
                    plf1 = np.nan_to_num(1.+plank_distr(T, w2) + plank_distr(T, w3))
                    plf2 = np.nan_to_num(-plank_distr(T, w2) + plank_distr(T, w3))
                    gamma[j, i, k] += assign_to_large_array(gauss_arr, w2, w3, plf1, plf2, E, tresh)*Phi_q1_qrid_abs[i, j*Nmodes**2 + k*Nmodes + l]*h/2/16
    
    return gamma

def get_avg_gamma(gamma,eig_val_2_grid, Omega, E, sigma = 1/4.):
    """Gets evarage near value of omega and outputs array of the same shape"""
    
    Omega_ind = E[E <= Omega].shape[0]
    
    delta_ind = int(sigma/(E[1] - E[0]))
    gamma_map = np.zeros((gamma.shape[0], gamma.shape[1], E.shape[0]))
    
    tresh = sigma*4
    dE = E[1] - E[0]
    gauss_arr = gauss(sigma, dE, tresh)
    
    for i in range(gamma.shape[0]):
        for j in range(gamma.shape[1]):
            for k in range(gamma.shape[2]):
                eig_2_ind = E[E <= eig_val_2_grid[j, k]].shape[0]
                
                """
                if np.abs(Omega_ind - eig_2_ind) > eig_2_ind - 2*delta_ind and np.abs(Omega_ind - eig_2_ind) < eig_2_ind + 2*delta_ind:
                    pref = np.abs(np.sum(gamma[i,j,k, Omega_ind - delta_ind:Omega_ind + delta_ind]))
                else:
                    pref = 0.
                
                buff = np.zeros(E.shape[0])
                
                istart = E[E<=(eig_val_2_grid[j, k]-tresh)].shape[0]
                iend = min(E[E<=(eig_val_2_grid[j, k]+tresh)].shape[0]+1, istart + gauss_arr.shape[0])
                if istart != iend:
                    if istart > 0 and iend < E.shape[0]:
                        buff[istart:iend] += gauss_arr*pref
                    elif istart == 0:
                        buff[istart:iend] += gauss_arr[-(iend - istart):]*pref
                    elif iend == E.shape[0]:
                        buff[istart:iend] += gauss_arr[:(iend - istart)]*pref
                
                gamma_map[i, j, :] += buff
                """
                
                #factor = np.abs(np.sum(gamma[i,j,k, Omega_ind - delta_ind:Omega_ind + delta_ind]))
                    
                #gamma_map[i, j, eig_2_ind] += factor
                #if 2*eig_2_ind > Omega_ind - 2*delta_ind and 2*eig_2_ind < Omega_ind + 2*delta_ind:
                factor = np.abs(np.sum(gamma[i,j,k, Omega_ind - delta_ind:Omega_ind + delta_ind]))
                    
                gamma_map[i, j, eig_2_ind] += factor
                #gamma_map[i, j, eig_2_ind + 1] = factor*np.exp(-dE**2/2/sigma**2); gamma_map[i, j, eig_2_ind + 2] = factor*np.exp(-(2*dE)**2/2/sigma**2)
                #if eig_2_ind - 2 > 0:
                #    gamma_map[i, j, eig_2_ind - 1] = factor*np.exp(-dE**2/2/sigma**2); gamma_map[i, j, eig_2_ind - 2] = factor*np.exp(-(2*dE)**2/2/sigma**2)

                #gamma_map[i, ind, Omega_ind - delta_ind:Omega_ind + delta_ind] += gamma[i,j,k, Omega_ind - delta_ind:Omega_ind + delta_ind]
                
    return gamma_map

def evaluate_spectral_function(gamma_map, eig_val_1, E):
    """Calculates partial contribution to the spectral function based on gamma"""
    output = np.zeros(gamma_map.shape)
    for i in range(gamma_map.shape[0]):
        f0 = 2*eig_val_1[i]
        f1 = (E**2 - eig_val_1[i]**2)**2
        output[i] = gamma_map[i]*f0/f1[None, :]
        
        #f1 = (E[None, :]**2 - eig_val_1[None, i]**2 - 2*eig_val_1[i]*(self_en_re[i].T))**2 + 4*eig_val_1[None, i]**2*(self_en_im[i].T)**2
        #output[i] = gamma_map[i]*f0/f1
        #output[i] = gamma_map[i]*f0/(f1[None, :] + (f0**2*(gamma_map[i].sum(axis = (0, 1))**2[None, :])))
        
        
    return output