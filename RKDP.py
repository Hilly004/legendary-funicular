import numpy as np
import Acceleration as acc

def rkdp(r,t,dt):
    def deriv(r_,t_):
        pos = r_[:3]
        vel = r_[3:]
        a = acc.accel1(pos,t_)
        return np.concatenate((vel,a))
    
    k1 = deriv(r,t)

    k2 = deriv(r + 0.2*dt*k1,t + 0.2*dt)

    k3 = deriv(r + dt*(3/40*k1 + 9/40*k2), t + 3/10*dt)

    k4 = deriv(r + dt*(44/45*k1 + -56/15*k2 + 32/9*k3), t + 4/5*dt)

    k5 = deriv(r + dt*(19372/6561*k1 + -25360/2187*k2 + 64448/6561*k3 + -212/729*k4), t + 8/9*dt)

    k6 = deriv(r + dt*(9017/3168*k1 + -355/33*k2 + 46732/5247*k3 + 49/176*k4 + -5103/18656*k5), t + dt)

    r_next = r + dt*(35/384*k1 + 500/1113*k3 + 125/192*k4 + -2187/6784*k5 + 11/84*k6)

    return r_next