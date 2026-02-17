import Runge_Kutta_4 as rk4
import numpy as np
import Positions as pos
import matplotlib.pyplot as plt
import Object_Data as dat
import Runge_Kutta_5 as rk5
import Acceleration as acc
from tqdm import tqdm,trange

steps = np.arange(400,3700,400)
p = pos.asteroid[1:4]

plt.figure(num=1)
ax = plt.axes(projection = '3d')
ax.plot(p[0],p[1],p[2])

plt.figure(num=2)
ax1 = plt.axes()
#ax1.set_xlim(-3.6e9,-2.8e9)
#ax1.set_ylim(1.47e11,1.478e11)
ax1.set_xlabel('x(m)')
ax1.set_ylabel('y(m)')
for i in range(-36,-31,1):
    ax1.scatter(p[0,i],p[1,i], color = 'red', label = 'Horizons')
    p_E = pos.positions['Earth'][1:4]
    p_M = pos.positions['Moon'][1:4]
    ax1.scatter(p_E[0,i],p_E[1,i], color = dat.colours['Earth'])
    ax1.scatter(p_M[0,i],p_M[1,i], color = dat.colours['Moon'])

plt.figure(num=3)
ax2 = plt.axes()

plt.figure(num=4)
ax3=plt.axes()

for step in tqdm(steps):
    times = np.arange(0,pos.positions['Sun'][0][-1],step)
    pos_Hor = np.zeros((len(times),3))
    pos_rk4 = np.zeros((len(times),3))
    pos_rk5 = np.zeros((len(times),3))

    r_Hor = np.array(pos.asteroid[1:7,0])
    r_rk4 = np.array(pos.asteroid[1:7,0])
    r_rk5 = np.array(pos.asteroid[1:7,0])

    for i,t in enumerate(tqdm(times)):
        pos_Hor[i] = r_Hor[:3]
        pos_rk4[i] = r_rk4[:3]
        pos_rk5[i] = r_rk5[:3]

        r_Hor = acc.planetary_position_1(t)
        r_rk4 = rk4.rk4(r_rk4,t,step)
        r_rk5 = rk5.rk5(r_rk5,t,step)

    
    ax.plot(pos_rk4[:,0],pos_rk4[:,1],pos_rk4[:,2])
    ax.plot(pos_rk5[:,0],pos_rk5[:,1],pos_rk5[:,2])

    #######

    ax1.scatter(pos_rk4[-100:,0],pos_rk4[-100:,1], label = ('dt = '+str(step)))
    ax1.scatter(pos_rk5[-100:,0],pos_rk5[-100:,1], label = ('dt = '+str(step)))

    #######

    d_Hor = np.empty(len(times))
    d_rk4 = np.empty(len(times))
    d_rk5 = np.empty(len(times))
 
    for i in trange(len(times)):
        d_Hor[i] = np.sqrt(pos_Hor[i,0]**2+pos_Hor[i,1]**2+pos_Hor[i,2]**2)
        d_rk4[i] = np.sqrt(pos_rk4[i,0]**2+pos_rk4[i,1]**2+pos_rk4[i,2]**2)
        d_rk5[i] = np.sqrt(pos_rk5[i,0]**2+pos_rk5[i,1]**2+pos_rk5[i,2]**2)
    
    ax2.plot(times,d_Hor)
    ax2.plot(times,d_rk4, label = 'RK4: '+str(step))
    ax2.plot(times,d_rk5, label =  str(step))

    ax3.plot(times,(d_Hor-d_rk4),label=str(step))
    

    ########



plt.legend()
plt.show()