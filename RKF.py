import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm, trange

import Positions as pos
import Object_Data as dat
import Timestep as ts
import Acceleration as acc

######

#def rkf_1(r,t,dt):

#    def deriv(r_,t_):
#        pos = r_[:3]
#        vel = r_[3:]
#        a = acc.accel(pos,t_)
#        return np.concatenate((vel,a))
    
#    k1 = deriv(r,t)
#    k2 = deriv()
#    k3 = deriv()
#    k4 = deriv()
#    k5 = deriv()
#    k6 = deriv()




######

def rkf_2(r,t,dt, tol):

    def deriv(r_,t_):
        pos = r_[:3]
        vel = r_[3:]
        a = acc.accel(pos,t_)
        return np.concatenate((vel,a))
    
    k1 = deriv(r,t)
    k2 = deriv(r + 0.25*dt*k1, t + 0.25*dt)
    k3 = deriv(r + dt*(3/32*k1 + 9/32*k2), t + 3/8*dt)
    k4 = deriv(r + dt*(1932/2197*k1 + -7200/2197*k2 + 7296/2197*k3), t + 12/13*dt)
    k5 = deriv(r + dt*(439/216*k1 + -8*k2 + 3680/513*k3 + -845/4104*k4), t + dt)
    k6 = deriv(r + dt*(-8/27*k1 + 2*k2 + -3544/2565*k3 + 1859/4104*k4 + -11/40*k5), t + 0.5*dt)

    r4 = r + dt*(25/216*k1 + 1408/2565*k3 + 2197/4104*k4 + -1/5*k5)

    r5 = r + dt*(16/135*k1 + 6656/12825*k3 + 28561/56430*k4 + -9/50*k5 + 2/55*k6)

    err = np.linalg.norm(r5 - r4, ord=np.inf)

    safety = 0.9

    if err == 0:
        scale = 2.0
    else:
        scale = safety * (tol/err)**0.2

    dt_new = dt*scale

    return r5, dt_new