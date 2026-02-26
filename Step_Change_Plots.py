import Step_Change as st
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm,trange
import Positions as pos
#######


plt.figure(num=1)
ax=plt.axes(projection='3d')

plt.figure(num=2)
ax1=plt.axes()
p_m = pos.positions['Moon'][1:4]
p_e = pos.positions['Earth'][1:4]
ax1.scatter(p_m[0,-40:-30],p_m[1,-40:-30])
ax1.scatter(p_e[0,-40:-30],p_e[1,-40:-30])

plt.figure(num=3)
ax2=plt.axes()

plt.figure(num=4)
ax3=plt.axes()

plt.figure(num=5)
ax4=plt.axes()

for step in tqdm(st.steps):
    times = np.arange(0,pos.positions['Sun'][0][-1],step)
    pos_Hor = st.positions_Hor[step]
    pos_rk4 = st.positions_rk4[step]
    pos_rk5 = st.positions_rk5[step]
    ax.plot(pos_Hor[:,0],pos_Hor[:,1],pos_Hor[:,2],label = ('Horizons - '+str(step)))
    ax.plot(pos_rk4[:,0],pos_rk4[:,1],pos_rk4[:,2],label = ('RK4 - '+str(step)))
    ax.plot(pos_rk5[:,0],pos_rk5[:,1],pos_rk5[:,2],label = ('RK5 - '+str(step)))
    

#######

    ax1.scatter(pos_rk4[-100:,0],pos_rk4[-100:,1], label = ('RK4 - '+str(step)))
    ax1.scatter(pos_rk5[-100:,0],pos_rk5[-100:,1], label = ('RK5 - '+str(step)))

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

    ax3.plot(times,(d_Hor-d_rk4),label='RK4 - '+str(step))
    ax3.plot(times,(d_Hor-d_rk5),label='RK5 - '+str(step))
    
    ax4.plot(times,(d_rk4-d_rk5), label=str(step))
########


ax.legend()
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()

plt.show()