import numpy as np
import scipy
from scipy.special import erfc
from scipy import optimize
from .crystal import crystal
import copy
from .consts import *
from .utils import points_on_a_sphere, inscribed_sphere_in_box, bounding_sphere_in_box, increment_dimensions

class ew_sum():
    
    def __init__(self, LAMBDA = None, xtl = None, trash = 1e-3):
        """Class for defining ewald sum, Gamma forceconsts"""
        
        self.xtl = copy.copy(xtl)
        
        self.xtl.uc /= bohr_to_A; self.xtl.r /= bohr_to_A; self.xtl.tau = {key: self.xtl.tau[key]/bohr_to_A for key in self.xtl.tau}; self.xtl.V /= bohr_to_A**3
        self.xtl.rc = np.linalg.inv(self.xtl.uc)
        #self.xtl.rc *= bohr_to_A
        
        self.eps, self.eborn = self._read_eps_eborn()
        self.inveps = np.linalg.inv(self.eps)
        
        
        # parameter of long range electrostatic forceconstant decay rate
        if LAMBDA:
            self.LAMBDA = LAMBDA
        else:
            self.find_LAMBDA()
        
        
        self.GAMMA = self._get_GAMMA() # grid of reciprocal space lattice vectors
        self.trash = 1e-6 # trashold for q
        
        # Gen born_onsight_correction
        self.born_onsite = np.zeros((self.xtl.Na, 3, 3))
        self.born_onsite = self._get_born_onsite()
        

    def find_LAMBDA(self, npts = 40, tol = 1e-5):
        """Determines LAMBDA from 1nn distance"""
        
        doublenndist = self.xtl.find_1nn_dist()*2.0
        invdet = 1.0/(np.linalg.det(self.eps))**(1/2.)
        
        pts = points_on_a_sphere(npts, seed = 0)
        
        root = optimize.brentq(self.uc_cutoff_func, 1e-8, 10, args = (doublenndist, invdet, pts, tol))
    
        return root
        
    def uc_cutoff_func(self, x, doublenndist, invdet, pts, tol):
        """Cutoff func for real cell parameters"""
        
        y = 0.0
        for pt in pts:
            v0 = self.inveps@pt*doublenndist
            f0 = (v0@pt*doublenndist)**(1/2.)
            y = y + (x**3)*np.sum(np.abs(self.ewald_H_thingy(v0*x, f0*x)))*invdet
            
        return y/len(pts) - tol
    
    def _get_REAL_cutoff(self, npts = 40, tol = 1e-20):
        """Determines LAMBDA from 1nn distance"""
        
        invdet = 1.0/(np.linalg.det(self.eps))**(1/2.)
        pts = points_on_a_sphere(npts, seed = 0)
        
        root = optimize.brentq(self.ss_cutoff_func, 1e-4, 1e4, args = (invdet, pts, tol))
        #print(root)
        return root + bounding_sphere_in_box(self.xtl.uc)
    
    
    
    def ss_cutoff_func(self, x, invdet, pts, tol):
        """Cutoff func for real cell parameters"""
        
        y = 0.0
        for pt in pts:
            v0 = self.inveps@pt*x
            f0 = (v0@pt*x)**(1/2.)
            y = y + (self.LAMBDA**3)*np.sum(np.abs(self.ewald_H_thingy(v0*self.LAMBDA, f0*self.LAMBDA)))*invdet
            
        return y - tol
    
    def _get_REAL(self, npts = 40, tol = 1e-20):
        """Get GAMMA list"""

        REALcut = self._get_REAL_cutoff(npts = 40, tol = 1e-20)
        
        bd = np.ones(3)
        m0 = np.array([(2*bd[i] + 1)*self.xtl.uc[i] for i in range(3)])
        f0 = inscribed_sphere_in_box(m0)

        while f0 < REALcut:
            bd = increment_dimensions(bd, self.xtl.uc)
            m0 = np.array([(2*bd[i] + 1)*self.xtl.uc[i] for i in range(3)])
            f0 = inscribed_sphere_in_box(m0)
                
        grid = np.c_[np.meshgrid(np.arange(-bd[0], bd[0] + 1), np.arange(-bd[1], bd[1] + 1), np.arange(-bd[2], bd[2] + 1))].reshape(3, -1)
        grid = (self.xtl.uc @ grid).T
        grid = grid[np.linalg.norm(grid, axis = 1) < np.linalg.norm(REALcut)]
        return grid
    
        
    def _read_eps_eborn(self):
        """Reads dielectric const and born effective charges from the files"""
        eps_eborn = np.genfromtxt('./infile.lotosplitting')
        eps = eps_eborn[:3]
        eborn = eps_eborn[3:]
        eborn = eborn.reshape((eborn.shape[0]//3, 3, 3))

        return eps, eborn

    def _get_GAMMA(self, npts = 40, tol = 1e-20):
        """Get GAMMA list"""

        Gcut = self._get_GAMMA_cutoff(npts = 40, tol = tol)
        #print(Gcut)
        bd = np.ones(3)
        m0 = np.array([(2*bd[i] + 1)*self.xtl.rc[i] for i in range(3)])
        f0 = inscribed_sphere_in_box(m0)

        while f0 < Gcut:
            #print(f0)
            bd = increment_dimensions(bd, self.xtl.rc)
            m0 = np.array([(2*bd[i] + 1)*self.xtl.rc[i] for i in range(3)])
            f0 = inscribed_sphere_in_box(m0)

        #print(bd)
        grid = np.c_[np.meshgrid(np.arange(-bd[0], bd[0] + 1), np.arange(-bd[1], bd[1] + 1), np.arange(-bd[2], bd[2] + 1))].reshape(3, -1)
        grid = (self.xtl.rc @ grid).T
        grid = grid[np.linalg.norm(grid, axis = 1) < np.linalg.norm(Gcut)]
        return grid
        
    
    def _get_GAMMA_cutoff(self, npts = 40, tol = 1e-20):
        """Get GAMMA grid"""

        pts = points_on_a_sphere(npts, seed = 0)


        f0 = inscribed_sphere_in_box(self.xtl.rc)
        # Check if can find minimum
        root = optimize.brentq(self.rc_cutoff_func, 1e-9*f0, 1e9*f0, args = (pts, tol))
    
        return root + bounding_sphere_in_box(self.xtl.rc)

    def rc_cutoff_func(self, x, pts, tol):
        """Func that should be zero on G cutoff"""
        inv4lambdasq = 1/4./self.LAMBDA**2

        y = 0
        for pt in pts:
            v = x*pt*2*np.pi
            norm_v = v.dot(self.eps).dot(v)
            y += np.exp(-norm_v*inv4lambdasq)*norm_v

        return y - tol
    
    
    ##############################################################################################
    
    def ewald_H_thingy(self, x, y):
        """Until func for real part of el stat corr"""
        
        H = np.zeros((3, 3))
        if np.abs(y) < 1e-10:
            return H
        
        twooversqrtpi = 1.128379167095513
        
        
        erfcy = erfc(y)
        invy2 = 1.0/(y*y)
        invy3 = 1.0/(y*y*y)
        expminvy2 = np.exp(-y*y)
        f0 = invy2*(3*erfcy*invy3 + twooversqrtpi*expminvy2*(3*invy2 + 2))
        f1 = erfcy*invy3 + twooversqrtpi*expminvy2*invy2

        # And get the H-thingy
        return x[:, None]*x[None, :]*f0 - self.inveps*f1
        
    
    def _chi_partial(self, K):
        """Util function for reciprocal part of el stat corr"""
        K_norm = K.dot(self.eps).dot(K)
        inv4lambdasq = 1/4./self.LAMBDA**2
        
        return np.exp(-K_norm*inv4lambdasq)/K_norm
    
    
    def _add_born_proj(self, Phi_tmp):
        """Adds born effective charges to forceconsts"""
        Phi = np.zeros((self.xtl.Na*3, self.xtl.Na*3), dtype = np.complex128)
        for i in range(self.xtl.Na):
            for j in range(self.xtl.Na):
                Phi[i*3:(i+1)*3, j*3:(j+1)*3] = self.eborn[i]@Phi_tmp[i*3:(i+1)*3, j*3:(j+1)*3]@self.eborn[j]
        
        return Phi

    def _add_born_proj_naive(self, Phi_tmp):
        """Adds born effective charges to forceconsts"""
        
        Phi = np.zeros((self.xtl.Na*3, self.xtl.Na*3), dtype = np.complex128)
        for a1 in range(self.xtl.Na):
            for a2 in range(self.xtl.Na):
                for i in range(3):
                    for j in range(3):
                        for ii in range(3):
                            for jj in range(3):
                                Phi[a1*3:(a1 + 1)*3, a2*3:(a2+1)*3][i, j] += self.eborn[a1][i, ii]*self.eborn[a2][j, jj]*Phi_tmp[a1*3:(a1 + 1)*3, a2*3:(a2+1)*3][ii, jj]
                        
        return Phi
    
    def _get_born_onsite(self):
        """Get correction to preserve sum rule"""
        Phi_tmp = self.Phi_q(np.array([0.0, 0.0, 0.0]))
        #print(Phi_tmp)
        
        born_onsite = np.zeros((self.xtl.Na, 3, 3))
        
        for i in range(self.xtl.Na):
            m = np.zeros((3, 3))
            for j in range(self.xtl.Na):
                m += np.real(Phi_tmp[i*3:(i+1)*3, j*3:(j+1)*3])
            born_onsite[i] = -m
        
        return born_onsite
        
    def Phi_q(self, q, qdir = np.array([1, 0, 0])):
        """Reciprocal space part of el stat corr"""
        Phi_out = np.zeros((3*self.xtl.Na, 3*self.xtl.Na), dtype = np.complex128)

        for g in self.GAMMA:
            K = (g + q)*2*np.pi
            if np.linalg.norm(K)**2 < self.trash**2:
                continue
            else:
                chipart = self._chi_partial(K)
                kk = K[None, :]*K[:, None]
                for key in self.xtl.tau:
                    ikr = -np.dot(K, self.xtl.tau[key])
                    Chi= chipart*np.exp(1.0j*ikr)
                    Phi_out[key[0]*3:(key[0] + 1)*3, key[1]*3:(key[1] + 1)*3] += kk*Chi
                
        #print(Phi_out)        
        Phi_out = self.add_Hermitian(Phi_out)
        
        Phi_out = self._add_born_proj(Phi_out)
        
        for i in range(self.xtl.Na):
            Phi_out[i*3:(i+1)*3, i*3:(i+1)*3] += self.born_onsite[i]
        #Phi_out = self.add_Hermitian(Phi_out)
        return Phi_out

    def add_Hermitian(self, Phi):
        Phi_full = np.zeros((3*self.xtl.Na, 3*self.xtl.Na), dtype = np.complex128)
        
        for a1 in range(self.xtl.Na):
            for a2 in range(a1, self.xtl.Na):
                Phi_full[a1*3:(a1+1)*3, a2*3:(a2+1)*3] = Phi[a1*3:(a1+1)*3, a2*3:(a2+1)*3]
                for i in range(3):
                    for j in range(3):
                        Phi_full[a2*3:(a2+1)*3, a1*3:(a1+1)*3][j, i] = np.conjugate(Phi_full[a1*3:(a1+1)*3, a2*3:(a2+1)*3][i, j])
                        
        return Phi_full
    
    def _H(self, x_vec, y, eps_inv):
        """Util function for real part of el corr"""
        return x_vec[:, None]*x_vec[None, :]/y**2*(3*erfc(y)/y**3 + 2*np.exp(-y**2)/np.pi**(1/2.)*(3/y**2 + 2)) - eps_inv*(erfc(y)/y**3 + 2*np.exp(-y)/np.pi**(1/2.)/y**2)

    def Phi_r(self, q, fct_R):
        """Real space part of el corr
           LAMBDA - ewald summation parameter"""
        eps_inv = np.linalg.inv(self.eps)
        eps_det = np.linalg.det(self.eps)
        
        
        Phi_out = np.zeros((3*self.xtl.Na, 3*self.xtl.Na), dtype = np.complex128)
        const_mult = -self.LAMBDA**3/eps_det**(1/2.)
        for key in fct_R:
            for R in fct_R[key]:
                r = R + self.xtl.tau[key]
                DELTA = eps_inv.dot(r)
                DELTA_norm = DELTA.dot(r)**(1/2.)

                x_vec = self.LAMBDA*DELTA
                y = self.LAMBDA*DELTA_norm
                Phi_out[(key[0] - 1)*3:key[0]*3, (key[1] - 1)*3:key[1]*3] += self._H(x_vec, y, eps_inv)*np.exp(1.0j*q.dot(R))

        return self.add_born_proj(Phi_out*const_mult)
    
    def Phi_c(self):
        """Connecting part of el corr"""
        Phi_out = np.zeros((3*self.xtl.Na, 3*self.xtl.Na))
        for i in range(N):
            Phi_out[i*3:(i+1)*3, i*3*3:(i+1)*3] = self.LAMBDA**3/3/np.pi**(3/2.)/np.linalg.det(self.eps)**(1/2.)*np.linalg.inv(self.eps)
 
        return self.add_born_proj(Phi_out)
    
    
    def Phi_nac(self, qdir):
        """Nonanalytical correction for small q"""
        Phi_out = np.zeros((3*self.xtl.Na, 3*self.xtl.Na))
        if isinstance(qdir, type(None)):
            return Phi_out
        else: 
            qnorm = qdir.dot(self.eps).dot(qdir)
            for i in range(self.xtl.Na):
                for j in range(self.xtl.Na):
                    Phi_out[i*3:(i+1)*3, j*3:(j+1)*3] += (self.eborn[i].dot(qdir))[:, None]*(self.eborn[j].dot(qdir))[None, :]/qnorm
        return Phi_out
    
    def long_range_electrostatics(self, q, case = 1, **kwargs):
        """Correction to dynamical matrics from long range electrostatics
           returns dynamical matrix Phi(q) in eV/A^2"""
        
        if case == 1: # only reciprocal space correction
            Phi_out = self.Phi_q(q) 
            if np.linalg.norm(q) < 1e-16: # nonanalytical correction for small q
                Phi_out +=  self.Phi_nac(np.array([1, 0, 0]))# returns forceconsts in Ry/Bohr^2
            return 4*np.pi/self.xtl.V*Hartree_to_eV/bohr_to_A**2*Phi_out
        
        if case == 2: # all three parts
            print('Hello')
            fct_R = kwargs['fct_R']
            return 4*np.pi/self.xtl.V*(Phi_nac + self.Phi_q(q) + self.Phi_r(q, fct_R) + self.Phi_c()) # returns forceconsts in eV/A^2