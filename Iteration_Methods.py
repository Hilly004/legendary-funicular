import numpy as np
import Runge_Kutta_4 as rk4
import Runge_Kutta_5 as rk5
import RKDP as rkdp
import RKF as rkf
import DOP853 as dop
import Acceleration as acc
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

G = 6.6726e-11
dt = 10000
dt_rkf = 10000
t = 0
t_rkf = 0
t_final = 1e9

tolerances = [1e-11,1e-12,1e-13]
results = {}

r0 = 1e9
v0 = np.sqrt(G*acc.mass1/r0)


r_rk4=np.array([r0,0,0,0,v0,0])
r_rkf=r_rk4.copy()
r_rk5=r_rk4.copy()
r_rkdp=r_rk4.copy()
r_dop=r_rk4.copy()

rk4_pos = []
rk5_pos = []
rkf_pos = []
rkdp_pos = []

dop_pos=[]

times = []
times_rkf = []


for tol in tqdm(tolerances):
    t_rkf = 0
    dt_rkf=10000
    r_rkf = r_rk4.copy()
    
    rkf_pos = []
    times_rkf = []
    while t_rkf < t_final:
        

        rkf_pos.append(r_rkf[:3])


        times_rkf.append(t_rkf)

        r_rkf, dt_new = rkf.rkf_2(r_rkf,t_rkf,dt_rkf, tol)

        dt_rkf = dt_new
        t_rkf+=dt_rkf

    results[tol] = {'positions': np.array(rkf_pos),
                    'times': np.array(times_rkf)}

######


while t < t_final:


    rk4_pos.append(r_rk4[:3])
    rk5_pos.append(r_rk5[:3])
    rkdp_pos.append(r_rkdp[:3])
    dop_pos.append(r_dop[:3])


    times.append(t)

    r_rk4 = rk4.rk4(r_rk4,t,dt)
    r_rk5 = rk5.rk5(r_rk5,t,dt)
    r_rkdp = rkdp.rkdp(r_rkdp,t,dt)
    r_dop = dop.dop853(r_dop,t,dt)
    
    
    t +=dt

rk4_pos = np.array(rk4_pos)
rk5_pos = np.array(rk5_pos)
rkdp_pos = np.array(rkdp_pos)
dop_pos = np.array(dop_pos)

times = np.array(times)

######

rkf_pos = np.array(rkf_pos)

times_rkf = np.array(times_rkf)

######

plt.figure(num=1)
ax = plt.axes(projection='3d')
s = (-1.5e9,1.5e9)
ax.set_xlim(s)
ax.set_ylim(s)
ax.set_zlim(s)
ax.set_xlabel('x(m)')
ax.set_ylabel('y(m)')
ax.set_zlabel('z(m)')
ax.plot(rk4_pos[:,0],rk4_pos[:,1],rk4_pos[:,2],label='RK4')
ax.plot(rk5_pos[:,0],rk5_pos[:,1],rk5_pos[:,2],label='RK5')
ax.plot(rkdp_pos[:,0],rkdp_pos[:,1],rkdp_pos[:,2],label='RKDP')
ax.plot(rkf_pos[:,0],rkf_pos[:,1],rkf_pos[:,2],label='RKF')
ax.plot(dop_pos[:,0],dop_pos[:,1],dop_pos[:,2],label='DOP853')

ax.scatter(0,0,0)
ax.legend()

plt.figure(num=2)

rs = np.full(len(times),r0)

plt.xlabel('Time (s)',fontsize = 'large')
plt.ylabel('Separation of iteration method\nradii from true radius (m)',fontsize = 'large')

rk4_mag = np.linalg.norm(rk4_pos,axis=1)
rk5_mag = np.linalg.norm(rk5_pos,axis=1)
rkdp_mag = np.linalg.norm(rkdp_pos,axis=1)
dop_mag = np.linalg.norm(dop_pos,axis=1)
plt.axhline(y=0,color = 'black',linestyle=':')
plt.plot(times,rs-rk4_mag, label='RK4')
plt.plot(times,rs-rk5_mag, label='RK5')
plt.plot(times,rs-rkdp_mag, label='RKDP')
plt.plot(times,rs-dop_mag, label='DOP853')
for tol in results:
    t = results[tol]['times']
    rs_ = np.full(len(t),r0)
    rkf_ = results[tol]['positions']
    mag =np.linalg.norm(rkf_,axis=1)
    plt.plot(t,rs_-mag,label=f"tol={tol}")
plt.legend()

plt.figure(num=3)
plt.xlabel('Time (s)')
plt.ylabel('Separation of methods (m)')
plt.plot(times,np.linalg.norm(rk4_pos-rk5_pos,axis=1),label='RK4-RK5')
plt.plot(times,np.linalg.norm(rk4_pos-rkdp_pos,axis=1),label='RK4-RKDP')
plt.plot(times,np.linalg.norm(rk4_pos-dop_pos,axis=1),label='RK4-DOP853')
plt.legend()

plt.figure(num=4)
plt.plot(times,np.log10(np.linalg.norm(rk4_pos-rk5_pos,axis=1)),label='RK4-RK5')
plt.plot(times,np.log10(np.linalg.norm(rk4_pos-rkdp_pos,axis=1)),label='RK4-RKDP')
plt.plot(times,np.log10(np.linalg.norm(rk4_pos-dop_pos,axis=1)),label='RK4-DOP853')

plt.show()
