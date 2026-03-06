import numpy as np
import Acceleration as acc
from scipy.integrate import solve_ivp

def dop853(r,t,dt):
    def deriv(t_,r_):
        pos = r_[:3]
        vel = r_[3:]
        a = acc.accel1(pos,t_)
        return np.concatenate((vel,a))
    
    sol = solve_ivp(deriv,
                    (t,t+dt),
                    r,
                    method ='DOP853',
                    t_eval=[t+dt],
                    rtol=1e-9,
                    atol=1e-12)
    
    return sol.y[:,-1]