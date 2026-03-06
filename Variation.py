import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm, trange

import Positions as pos
import Object_Data as dat
import Timestep as ts
import Acceleration as acc
import Runge_Kutta_4 as rk4
import Runge_Kutta_5 as rk5
import RKF as rkf
import RKDP as rkdp
import DOP853 as dop

######

start = time.perf_counter()

######
dt = 3600
dt_rkf = 3600
t = 0
t_rkf = 0
t_final = pos.positions['Sun'][0][-1]

tolerances = [1e-8, 1e-9,1e-10]
results = {}
horizons = {}

r_rk4=np.array(pos.asteroid[1:7,0])
r_Hor_1=r_rk4.copy()
r_Hor_2=r_rk4.copy()
r_rkf=r_rk4.copy()
r_rk5=r_rk4.copy()
r_rkdp=r_rk4.copy()
r_dop=r_rk4.copy()

rk4_pos = []
rk5_pos = []
rkf_pos = []
rkdp_pos = []
hor_pos_1 = []
hor_pos_2 = []
dop_pos=[]

times = []
times_rkf = []




while t < t_final:

    r_Hor_1 = acc.planetary_position_1(t)

    rk4_pos.append(r_rk4[:3])
    rk5_pos.append(r_rk5[:3])
    rkdp_pos.append(r_rkdp[:3])
    dop_pos.append(r_dop[:3])
    hor_pos_1.append(r_Hor_1[:3])

    times.append(t)

    r_rk4 = rk4.rk4(r_rk4,t,dt)
    r_rk5 = rk5.rk5(r_rk5,t,dt)
    r_rkdp = rkdp.rkdp(r_rkdp,t,dt)
    r_dop = dop.dop853(r_dop,t,dt)
    
    
    t +=dt

for tol in tqdm(tolerances):
    t_rkf = 3600
    dt_rkf=3600
    r_rkf = np.array(pos.asteroid[1:7,0])
    r_Hor_2 = r_rkf.copy()
    
    rkf_pos = []
    times_rkf = []
    hor_pos_2 = []
    while t_rkf < t_final:
        

        rkf_pos.append(r_rkf[:3])
        hor_pos_2.append(r_Hor_2[:3])

        times_rkf.append(t_rkf)

        r_rkf, dt_new = rkf.rkf_2(r_rkf,t_rkf,dt_rkf, tol)
        r_Hor_2 = acc.planetary_position_1(t_rkf)

        dt_rkf = dt_new
        t_rkf+=dt_rkf

    results[tol] = {'positions': np.array(rkf_pos),
                    'times': np.array(times_rkf)}
    
    horizons[tol] = {'positions': np.array(hor_pos_2),
                    'times': np.array(times_rkf)}
######


rk4_pos = np.array(rk4_pos)
rk5_pos = np.array(rk5_pos)
rkdp_pos = np.array(rkdp_pos)
hor_pos_1 = np.array(hor_pos_1)
dop_pos = np.array(dop_pos)

times = np.array(times)

######

rkf_pos = np.array(rkf_pos)
hor_pos_2 = np.array(hor_pos_2)

times_rkf = np.array(times_rkf)

######

plt.figure(num=1)
plt.xlabel('Time (s)', size = 'large')
plt.ylabel('Separation between integration method\nand Horizons model (m)', size = 'large')
plt.plot(times, np.linalg.norm(hor_pos_1-rk4_pos,axis=1), label = 'RK4')
plt.plot(times, np.linalg.norm(hor_pos_1-rk5_pos,axis=1), label = 'RK5')
plt.plot(times, np.linalg.norm(hor_pos_1-rkdp_pos,axis=1), label= 'RKDP')
plt.plot(times,np.linalg.norm(hor_pos_1-dop_pos,axis=1), label='DOP853')
for tol in results:
    t = results[tol]['times']
    pos_rkf = results[tol]['positions']
    pos_hor = horizons[tol]['positions']
    plt.plot(t, np.linalg.norm(pos_hor-pos_rkf,axis=1), label =f"tol={tol}")

plt.legend()

######

plt.figure(num=2)
plt.xlabel('Time (s)', size = 'large')
plt.ylabel('log of separation between integration method\nand Horizons model (m)', size = 'large')
rk4_err = np.linalg.norm(hor_pos_1-rk4_pos,axis=1)
rk5_err = np.linalg.norm(hor_pos_1-rk5_pos,axis=1)
rkdp_err = np.linalg.norm(hor_pos_1-rkdp_pos,axis=1)
dop_err = np.linalg.norm(hor_pos_1-dop_pos,axis=1)

rk4_err[rk4_err==0]=1e-30
rk5_err[rk5_err==0]=1e-30
rkdp_err[rkdp_err==0]=1e-30
dop_err[dop_err==0]=1e-30

plt.plot(times, np.log10(rk4_err), label = 'RK4')
plt.plot(times, np.log10(rk5_err), label = 'RK5')
plt.plot(times, np.log10(rkdp_err), label = 'RKDP')
plt.plot(times,np.log10(dop_err), label='DOP853')

plt.legend()

######

plt.figure(num=3)
plt.xlabel('Time (s)', size = 'large')
plt.ylabel('Separation between\ndifferent integration methods', size = 'large')
plt.plot(times, np.linalg.norm(rk4_pos-rk5_pos,axis=1), label='RK4 - RK5')
plt.plot(times, np.linalg.norm(rk4_pos-rkdp_pos, axis=1), label ='RK4 - RKDP')
plt.plot(times, np.linalg.norm(rk5_pos-rkdp_pos, axis=1), label ='RK5 - RKDP')
plt.legend()

######

end = time.perf_counter()
elapsed = end-start
print(f"{elapsed:.6f} seconds")

######
plt.show()