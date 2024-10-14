import numpy as np


class crystal():
    def __init__(self, mass):
        """Class for defining crystal lattice"""
        self.uc, self.V, self.tau, self.Na, self.r = self._get_uc()
        self.rc = self._gen_rc()
        self.mass = mass
        
    def _get_uc(self):
        """Get basis for TDEP structures"""
        with open('./infile.ucposcar', 'r') as f:
            f.readline()
            scaler = float(f.readline().rstrip().split()[0])
        
        basis = np.genfromtxt('./infile.ucposcar', skip_header=2, max_rows = 3)*scaler
        V = np.linalg.det(basis)
        
        r_uc = np.genfromtxt('./infile.ucposcar', skip_header=8)
        r = r_uc @ basis
        
        Na = r.shape[0]
        tau = {}
        for i in range(Na):
            for j in range(i, Na):
                tau[(i, j)] = (r[j] - r[i])
        
        return basis, V, tau, Na, r
        

    def _gen_rc(self):
        """Generates reciprocal lattice using TDEP input files"""
        return np.linalg.inv(self.uc)
    
    def find_1nn_dist(self):
        """Find nearest neighbour distance in the lattice"""
        
        f0 = bounding_sphere_in_box(self.uc)
        nrep = 1
        m0 = self.uc*(2*nrep+1)
        
        while inscribed_sphere_in_box(m0) <= f0:
            nrep += 1
            m0 = self.uc*(2*nrep+1)
        
        
        r = np.inf
        for a1 in range(self.Na):
            for a2 in range(self.Na):
                for i in range(-nrep, nrep+1):
                    for j in range(-nrep, nrep+1):
                        for k in range(-nrep, nrep + 1):
                            f0 = np.linalg.norm(i*self.uc[0] + j*self.uc[1] + k*self.uc[2] + self.r[a2] - self.r[a1])
                            if f0 > 0: r = min(r, f0)
        
        return r
    
    
class crystal_ss(crystal):
    def __init__(self, mass):
        """Class for defining crystal lattice"""
        self.uc, self.V, self.tau, self.Na, self.r, self.Ntypes, self.idtypes = self._get_uc()
        self.rc = self._gen_rc()
        self.mass = mass
        
        
    def _get_uc(self):
        """Get basis for TDEP structures"""
        with open('./infile.ssposcar', 'r') as f:
            f.readline()
            scaler = float(f.readline().rstrip().split()[0])
            [f.readline() for i in range(4)]
            types = list(map(int, f.readline().rstrip().split()))
            Ntypes = len(types)
        
        basis = np.genfromtxt('./infile.ssposcar', skip_header=2, max_rows = 3)*scaler
        V = np.linalg.det(basis)
        
        r_uc = np.float64(np.genfromtxt('./infile.ssposcar', skip_header=8)[:, :3])
        print(r_uc.shape)
        r = r_uc @ basis
        
        Na = r.shape[0]
        tau = r[:, None] - r[None, :]
        
        start = 0
        idtypes = np.zeros(Na, dtype = np.int32)
        
        for i, el in enumerate(types):
            idtypes[start:start + el] = i
            start += el
        return basis, V, tau, Na, r, Ntypes, idtypes