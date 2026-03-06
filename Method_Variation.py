import numpy as np
import Runge_Kutta_4 as rk4
import Runge_Kutta_5 as rk5
import RKDP as rkdp
import RKF as rkf
import DOP853 as dop
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import time

######

es = [0,0.5,0.75,0.6615]
rk4_res = {}
rk5_res = {}
rkdp_res = {}
dop_res = {}
true_res = {}

######

G = 6.6726e-11
mass1 = 1e23
r0=1e9 #semi-major axis

t0=0
#t1 = 1e8
dt=5000


def v_init(e):
    return np.sqrt((G*mass1*(1+e))/(r0*(1-e)))

def r_init(e):
    return r0*(1-e) #this gives periapsis distance

def per(e):
    return 2*np.pi*np.sqrt((r0**3)/(G*mass1))

def true_pos_(t,e,a):
    n=np.sqrt(G*mass1/a**3)
    M=n*t

    E = M
    for _ in range(10):
        E = E - (E - e*np.sin(E) - M) / (1 - e*np.cos(E))

    x = a * (np.cos(E) - e)
    y = a * np.sqrt(1 - e**2) * np.sin(E)

    return np.array([x, y, 0])
######
start = time.perf_counter()

for e in tqdm(es,desc='Eccentricities'):

    r_0 = r_init(e)
    v_0 = v_init(e)
    r = np.array([r_0,0,0,0,v_0,0])
    r_rk4 = r.copy()
    r_rk5 = r.copy()
    r_rkdp = r.copy()
    r_dop = r.copy()
    #r_true = r[:3].copy()

    rk4_pos = []
    rk5_pos = []
    rkdp_pos = []
    dop_pos=[]
    true_pos=[]

    times = []
    t = t0
    t1 = 5*per(e)
    a=r0
    while t<t1:
        r_true = true_pos_(t,e,a)

        rk4_pos.append(r_rk4[:3])
        rk5_pos.append(r_rk5[:3])
        rkdp_pos.append(r_rkdp[:3])
        dop_pos.append(r_dop[:3])
        true_pos.append(r_true)

        times.append(t)

        r_rk4 = rk4.rk4(r_rk4,t,dt)
        r_rk5 = rk5.rk5(r_rk5,t,dt)
        r_rkdp = rkdp.rkdp(r_rkdp,t,dt)
        r_dop = dop.dop853(r_dop,t,dt)
        
        
        t+=dt

    rk4_res[e] = {'positions': np.array(rk4_pos),
                    'times': np.array(times)}
    rk5_res[e] = {'positions': np.array(rk5_pos),
                    'times': np.array(times)}
    rkdp_res[e] = {'positions': np.array(rkdp_pos),
                    'times': np.array(times)}
    dop_res[e] = {'positions': np.array(dop_pos),
                    'times': np.array(times)}
    true_res[e] = {'positions': np.array(true_pos),
                    'times': np.array(times)}
    
np.savez_compressed(
    'Method_Variation_Results',
    rk4 = rk4_res,
    rk5 = rk5_res,
    rkdp = rkdp_res,
    dop = dop_res,
    true = true_res
)
end = time.perf_counter()

elapsed = end - start

print(elapsed)