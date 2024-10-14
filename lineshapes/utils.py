import numpy as np

def gauss(sigma, dE, tresh):
    """Gaussiang distribution on a cut"""
    E = np.arange(-tresh, tresh, dE)
    return 1./(sigma)/np.pi**(1/2.)*np.exp(-(E/sigma)**2)

def plank_distr(T, w):
    """Plank distribution on a cut"""
    return 1/(np.exp(w/T) - 1)

def points_on_a_sphere(n, seed = 32):
    """Generates n points on a sphere"""
    
    pts = np.zeros((n, 3))
    
    offset = 2./n
    increment = np.pi*(3. - 5**(1/2.))
    
    for i in range(n):
        y = (i*offset - 1) + offset*0.5
        rad = max(1.0 - y**2, 0.0)**(1/2.)
        phi = (i+1 + seed)%n
        pts[i] = np.array([np.cos(phi)*rad, y, np.sin(phi)*rad])
        pts[i] /= np.linalg.norm(pts[i])
        
    return pts

def inscribed_sphere_in_box(cell):
    """Determines maximum redius of the sphere
       that can be inscribed in a box"""
    
    na = np.cross(cell[1], cell[2])
    nb = np.cross(cell[2], cell[0])
    nc = np.cross(cell[0], cell[1])
    
    na /= np.linalg.norm(na); nb /= np.linalg.norm(nb); nc /= np.linalg.norm(nc)
    
    return min(np.abs(na.dot(cell[0])), np.abs(nb.dot(cell[1])), np.abs(nc.dot(cell[2])))/2


def bounding_sphere_in_box(cell):
    """Gives radius of the smallest sphere 
       that containes the box"""
    
    a, b, c = cell
    
    v = a - b - c
    diag1 = np.dot(v, v)
    
    v = -a + b - c
    diag2 = np.dot(v, v)
    
    v = -a - b + c

    diag3 = np.dot(v, v)

    v = a + b + c
    diag4 = np.dot(v, v)
    
    return max(diag1, diag2, diag3, diag4)**(1/2.)/2

def increment_dimensions(bd, cell):
    """Increments one of the bounds based on 
       best filling of the sphere"""
    
    m0 = np.array([(2*bd[i] + 1)*cell[i] for i in range(3)])
    
    ff0 = inscribed_sphere_in_box(m0)
    
    f0 = 0
    final_ind = -1
    for i in range(3):
        bd_new = np.copy(bd)
        bd_new[i] += 1
        
        m0_new = np.array([(2*bd_new[j] + 1)*cell[j] for j in range(3)])
        f1 = inscribed_sphere_in_box(m0_new)
        
        if f1 > f0 and f1 != ff0:
            final_ind = i
            f0 = f1
    
    #print(final_ind)
    if final_ind == -1:
        final_ind = np.argmin(bd)
    
    bd[final_ind] += 1
        
    return bd