import numpy as np

def get_infos2Stokes_2D(in_dim=2, out_dim=1):
    mu = 1
    f = lambda  x,y : np.zeros_like(x)
    u1_true = lambda x, y : 20*x*(y**3)
    u2_true = lambda x, y: 5*(x**4 - y**4)
    p_true = lambda x, y: 60*(x**2)*y - 20*(y**3)
    ux_left = lambda x, y: np.zeros_like(x)
    ux_right = lambda x, y: np.zeros_like(x)
    uy_bottom = lambda x, y: np.zeros_like(x)
    uy_top = lambda x, y: np.zeros_like(x)

    return u1_true, u2_true, p_true, ux_right, ux_left, uy_top, uy_bottom,f

