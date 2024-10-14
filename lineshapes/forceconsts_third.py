import numpy as np
from functools import reduce

from .consts import *
from .forceconsts_second import *

def get_3ph_eig(q1, q2, q3, Na, fct_R, fct_Phi, mass, polar = False, **kwargs):
    """Precompute eigen values and eigenvectors for 3ph process"""
    unit_conv_const = vel_light_THz**2/aem_to_eV
    
    eig_val_1, eig_vect_1, _ = get_eig_val(q1, Na, fct_Phi, fct_R, mass, polar = polar, **kwargs)
    eig_val_2, eig_vect_2, _ = get_eig_val(q2, Na, fct_Phi, fct_R, mass, polar = polar, **kwargs)
    eig_val_3, eig_vect_3, _ = get_eig_val(q3, Na, fct_Phi, fct_R, mass, polar = polar, **kwargs)
    
    # precompute eigen vector matrix
    eig_vect_1 = eig_vect_1.reshape((Na*3, Na, 3))/(eig_val_1[:, None, None])**(1/2.)*unit_conv_const**(1/2.)
    eig_vect_2 = eig_vect_2.reshape((Na*3, Na, 3))/(eig_val_2[:, None, None])**(1/2.)*unit_conv_const**(1/2.)
    eig_vect_3 = eig_vect_3.reshape((Na*3, Na, 3))/(eig_val_3[:, None, None])**(1/2.)*unit_conv_const**(1/2.)
    
    return reduce(np.kron, [eig_vect_1, eig_vect_2, eig_vect_3])

def get_3ph_dyn_mat(q1, q2, Na, fct_R, fct_Phi, fct_ind3ph, fct_Phi3ph, fct_R3ph, mass, polar = False, **kwargs):
    """Getting 3ph dynamical matrix for all the modes
    Output is Phi3ph(q1,q2,q3,s1,s2,s3) in 1/s^(3/2)/eV^(1/2)"""
    
    #precompute of eigenvectors and eigenvalues
    q3 = -q1-q2 # from momentum conservation laws
    
    # Precompute eigenvalues
    fEGV = get_3ph_eig(q1, q2, q3, Na, fct_R, fct_Phi, mass, polar = polar, **kwargs)
                  
    dyn_mat_3ph = np.zeros((3*Na*3*Na*3*Na), dtype = np.complex128)
    # Almost the write way to do this
    
    phase_factor = fct_R3ph[:, 1].dot(q2) + fct_R3ph[:, 2].dot(q3)
    phase_factor = np.cos(-phase_factor) + 1.0j*np.sin(-phase_factor)
    

    # Main loop
    for i in range(fct_ind3ph.shape[0]):
        dyn_mat_3ph += (fEGV[:, fct_ind3ph[i]]@fct_Phi3ph[i])*phase_factor[i]
    return dyn_mat_3ph


# GRID tools

def get_3ph_eig_grid(q1, qgrid, Na, fct_R, fct_Phi, mass, polar = False, **kwargs):
    """Precompute eigen values and eigenvectors for 3ph process"""
    unit_conv_const = vel_light_THz**2/aem_to_eV
    
    eig_val_1, eig_vect_1, _ = get_eig_val(q1, Na, fct_Phi, fct_R, mass, polar = polar, **kwargs)
    eig_vect_1 = eig_vect_1.reshape((Na*3, Na, 3))/(eig_val_1[:, None, None])**(1/2.)*unit_conv_const**(1/2.)
    
    q3 = -qgrid - q1
    
    # Work much faster but overall perfomance might be slower (suppose memory problems)
    eig_val_2_grid, eig_vect_2_grid, _ = get_eig_val_grid(qgrid, Na, fct_Phi, fct_R, mass, polar = polar, **kwargs)
    eig_val_3_grid, eig_vect_3_grid, _ = get_eig_val_grid(q3, Na, fct_Phi, fct_R, mass, polar = polar, **kwargs)
        
    # precompute eigen vector matrix
    eig_vect_2_grid = eig_vect_2_grid/(eig_val_2_grid[:, :, None])**(1/2.)*unit_conv_const**(1/2.)
    eig_vect_3_grid = eig_vect_3_grid/(eig_val_3_grid[:, :, None])**(1/2.)*unit_conv_const**(1/2.)
    
    eig_vect_2_grid = eig_vect_2_grid.reshape((qgrid.shape[0], Na*3, Na, 3))
    eig_vect_3_grid = eig_vect_3_grid.reshape((qgrid.shape[0], Na*3, Na, 3))
    
    return eig_vect_1, eig_vect_2_grid, eig_vect_3_grid, eig_val_1, eig_val_2_grid, eig_val_3_grid

def get_3ph_dyn_mat_grid(q1, qgrid, Na, fct_R, fct_Phi, fct_ind3ph, fct_Phi3ph, fct_R3ph, mass, polar = False, **kwargs):
    """Getting 3ph dynamical matrix for all the modes on grid of points
    Output is Phi3ph(q1,q2,q3,s1,s2,s3) in 1/s^(3/2)/eV^(1/2)"""
    
    
    #precompute of eigenvectors and eigenvalues
    q3 = -q1-qgrid # from momentum conservation laws
    
    # Precompute eigenvalues
    # Almost the write way to do this
    eig_vect_1, eig_vect_2_grid, eig_vect_3_grid, eig_val_1, eig_val_2_grid, eig_val_3_grid = get_3ph_eig_grid(q1, qgrid, Na, fct_R, fct_Phi, mass, polar = polar, **kwargs)

    # Start to compute three phonon lattice elements in reciprocal space
    dyn_mat_3ph = np.zeros((qgrid.shape[0], 3*Na*3*Na*3*Na), dtype = np.complex128)
    
    ##Precompute the phase factors
    phase_factor = fct_R3ph[:, 1].dot(qgrid.T) + fct_R3ph[:, 2].dot(q3.T)
    phase_factor = np.cos(-phase_factor) + 1.0j*np.sin(-phase_factor)
    
    for i, el in enumerate(eig_vect_2_grid):
        fEGV = reduce(np.kron, [eig_vect_1, eig_vect_2_grid[i], eig_vect_3_grid[i]])
        for j in range(fct_ind3ph.shape[0]):
            dyn_mat_3ph[i] += (fEGV[:, fct_ind3ph[j]]@fct_Phi3ph[j])*phase_factor[j, i]
            
    dyn_mat_3ph = np.nan_to_num(dyn_mat_3ph)
    eig_val_1 = np.nan_to_num(eig_val_1)
    eig_val_2_grid = np.nan_to_num(eig_val_2_grid).real
    eig_val_3_grid = np.nan_to_num(eig_val_3_grid).real
    return eig_val_1, eig_val_2_grid, eig_val_3_grid, dyn_mat_3ph