import numpy as np
from scipy import linalg as LA
from .consts import *
from .utils import *

def get_eig_val(q, Na, fct_Phi, fct_R, M_dict, polar = False, **kwargs):
    """Determines frequencies in units of THz from forceconsants
    mass in input should be given in aem"""
    
    unit_conv_const = 1./aem_to_eV*vel_light_THz**2
    
    dyn_mat = np.zeros((3*Na, 3*Na), dtype = np.complex128)
    
    phase_factor = {key: fct_R[key].dot(q) for key in fct_R}
    phase_factor = {key: np.cos(phase_factor[key]) + 1.0j*np.sin(phase_factor[key]) for key in phase_factor}
    
    for key in fct_Phi:
    
        dyn_mat[3*(key[0] - 1):3*key[0], 3*(key[1] - 1):3*key[1]] =  np.sum(phase_factor[key][:, None, None]*fct_Phi[key], axis = 0)/(M_dict[key[0]]*M_dict[key[1]])**(1/2.)*unit_conv_const
    
    if polar:
        dyn_mat_lo = kwargs['ew_xtl'].long_range_electrostatics(q/2/np.pi*bohr_to_A)
        
        for key in fct_Phi:
            dyn_mat[3*(key[0] - 1):3*key[0], 3*(key[1] - 1):3*key[1]] += dyn_mat_lo[3*(key[0] - 1):3*key[0], 3*(key[1] - 1):3*key[1]]/(M_dict[key[0]]*M_dict[key[1]])**(1/2.)*unit_conv_const
    
    eig_val, eig_vct = LA.eigh(dyn_mat, lower=False)
    #eig_val, eig_vct = LA.eig(dyn_mat)
    #print(eig_val)
    eig_val = np.nan_to_num(eig_val**(1/2.)); eig_val[eig_val < 1e-5] = 0.0
    eig_vct = np.round(eig_vct, 8).conj().T
    
    return eig_val, eig_vct, dyn_mat


# GRID tools

def get_eig_val_grid_legacy(qgrid, Na, fct_Phi, fct_R, M_dict, polar = False, **kwargs):
    """Determines frequencies from forceconsants in units of THz
    mass in input should be given in aem"""
    
    unit_conv_const = 1./aem_to_eV*vel_light_THz**2
    
    dyn_mat = np.zeros((qgrid.shape[0], 3*Na, 3*Na), dtype = np.complex128)
    
    phase_factor = {key: fct_R[key].dot(qgrid.T) for key in fct_R}
    phase_factor = {key: np.cos(phase_factor[key]) + 1.0j*np.sin(phase_factor[key]) for key in phase_factor}
    for key in fct_Phi:
        dyn_mat[:, 3*(key[0] - 1):3*key[0], 3*(key[1] - 1):3*key[1]] =  np.sum(phase_factor[key][:, :, None, None]*fct_Phi[key][:, None], axis = 0)/(M_dict[key[0]]*M_dict[key[1]])**(1/2.)*unit_conv_const
    
    eig_val, eig_vct = list(zip(*map(lambda x: LA.eigh(x, lower=False), dyn_mat)))
    #eig_val, eig_vct = list(zip(*map(lambda x: LA.eig(x), dyn_mat)))
    eig_val = np.nan_to_num(np.array(eig_val)**(1/2.)); eig_val[eig_val < 1e-5] = 0.0
    eig_vct = np.transpose(np.round(np.array(eig_vct,dtype = np.complex128), 8).conj(), axes = (0, 2, 1))
    
    return eig_val, eig_vct, dyn_mat


# GRID tools

def get_eig_val_grid(qgrid, Na, fct_Phi, fct_R, M_dict, polar = False, **kwargs):
    """Determines frequencies from forceconsants in units of THz
    mass in input should be given in aem"""
    
    unit_conv_const = 1./aem_to_eV*vel_light_THz**2
    
    dyn_mat = np.zeros((qgrid.shape[0], 3*Na, 3*Na), dtype = np.complex128)
    
    phase_factor = {key: fct_R[key].dot(qgrid.T) for key in fct_R}
    phase_factor = {key: np.cos(phase_factor[key]) + 1.0j*np.sin(phase_factor[key]) for key in phase_factor}
    for key in fct_Phi:
        dyn_mat[:, 3*(key[0] - 1):3*key[0], 3*(key[1] - 1):3*key[1]] =  np.sum(phase_factor[key][:, :, None, None]*fct_Phi[key][:, None], axis = 0)/(M_dict[key[0]]*M_dict[key[1]])**(1/2.)*unit_conv_const
    
    if polar:
        dyn_mat_lo = np.zeros((qgrid.shape[0], 3*Na, 3*Na), dtype = np.complex128)
        for i, q in enumerate(qgrid):
            dyn_mat_lo[i] = kwargs['ew_xtl'].long_range_electrostatics(q/2/np.pi*bohr_to_A)
        
        for key in fct_Phi:
            dyn_mat[:, 3*(key[0] - 1):3*key[0], 3*(key[1] - 1):3*key[1]] += dyn_mat_lo[:, 3*(key[0] - 1):3*key[0], 3*(key[1] - 1):3*key[1]]/(M_dict[key[0]]*M_dict[key[1]])**(1/2.)*unit_conv_const
    
    
    eig_val, eig_vct = list(zip(*map(lambda x: LA.eigh(x, lower=False), dyn_mat)))
    #eig_val, eig_vct = list(zip(*map(lambda x: LA.eig(x), dyn_mat)))
    
    eig_val = np.nan_to_num(np.array(eig_val)**(1/2.)); eig_val[eig_val < 1e-5] = 0.0
    eig_vct = np.nan_to_num(np.transpose(np.round(np.array(eig_vct,dtype = np.complex128), 8).conj(), axes = (0, 2, 1)))
    
    return eig_val, eig_vct, dyn_mat