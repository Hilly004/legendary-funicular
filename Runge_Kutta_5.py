import numpy as np
import Acceleration as acc

def rk5(r,t,dt):
    def deriv(r_,t_):
        pos = r_[:3]
        vel = r_[3:]
        a = acc.accel(pos,t_)
        return np.concatenate((vel,a))
    
    k1=deriv(r,t)

    k2=deriv(r + k1*dt/3, t + dt/3)

    k3=deriv(r + dt*(4/25*k1 + 6/25*k2), t + 2/5*dt)

    k4=deriv(r + dt*(1/4*k1 - 3*k2 + 15/4*k3), t + dt)

    k5=deriv(r + dt*(2/27*k1 + 10/9*k2 - 50/81*k3 + 8/81*k4), t + 2/3*dt)

    k6=deriv(r + dt*(2/25*k1 + 12/25*k2 + 2/15*k3 + 8/75*k4), t + 4/5*dt)

    return(r + (dt/192)*(23*k1 + 125*k3 - 81*k5 + 125*k6))