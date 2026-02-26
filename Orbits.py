import Runge_Kutta_4 as rk4
import numpy as np
import Positions as pos
import matplotlib.pyplot as plt
import Object_Data as dat
import Runge_Kutta_5 as rk5
import time
import Timestep as ts
import Acceleration as acc
from tqdm import tqdm, trange

start = time.perf_counter()

dt = 3600
N=0
times = np.arange(N*dt,pos.positions['Sun'][0][-1],dt)
posAst = np.zeros((len(times),3))
posAst_2 = np.zeros((len(times),3))
posHor = np.zeros((len(times),3))
#print(pos.positions['Sun'][0][-33])

r=np.array(pos.asteroid[1:7,N])
r_2=np.array(pos.asteroid[1:7,N])
r_Hor_2 = np.array(pos.asteroid[1:7,N])

#r = np.array([-3.22749253e+11,  6.53523922e+10, -1.90546027e+10, -1.68467588e+04, -1.29695567e+04, -1.02237685e+03])
#r = np.array([0.5e13,0.5e13,0.5e13,-1e4,-1e4,0])

for i,t in enumerate(tqdm(times)):
    posAst[i] = r[:3] #RK4
    posAst_2[i] = r_2[:3]  #RK5
    posHor[i] = r_Hor_2[:3]
    r = rk4.rk4(r,t,dt)
    r_2 = rk5.rk5(r_2,t,dt)
    r_Hor_2 = acc.planetary_position_1(t)
    


#print(len(posAst))

r_3=np.array(pos.asteroid[1:7,0])
r_Hor=np.array(pos.asteroid[1:7,0])
ti = 0
ast_pos = []
hor_pos = []
times_1 = []

while ti < pos.positions['Sun'][0][-1]:
    acceleration = acc.accel(r_3[:3],ti)
    dt = ts.timestep(acceleration)
    ast_pos.append(r_3[:3])
    hor_pos.append(r_Hor[:3])
    r_3 = rk4.rk4(r_3,ti,dt)
    r_Hor = acc.planetary_position_1(ti)
    times_1.append(ti)
    ti +=dt

ast_pos = np.array(ast_pos)
hor_pos = np.array(hor_pos)
times_1 = np.array(times_1)

length = len(ast_pos)
d_ast=np.empty(length)
d_hor=np.empty(length)

for i in range(length):
    d_ast[i] = np.sqrt(ast_pos[i,0]**2 + ast_pos[i,1]**2 + ast_pos[i,2]**2)
    d_hor[i] = np.sqrt(hor_pos[i,0]**2 + hor_pos[i,1]**2 + hor_pos[i,2]**2)

plt.figure(num=1)
ax = plt.axes(projection = '3d')
s = (-1e12,1e12)
ax.set_xlim(s)
ax.set_ylim(s)
ax.set_zlim(s)
ax.set_xlabel('x(m)')
ax.set_ylabel('y(m)')
ax.set_zlabel('z(m)')
ax.set_xticks([-1e12,-0.5e12,0,0.5e12,1e12])
ax.set_yticks([-1e12,-0.5e12,0,0.5e12,1e12])
ax.set_zticks([-1e12,-0.5e12,0,0.5e12,1e12])

for name in dat.names:
    p = pos.positions[name][1:4]
    lines = ax.plot(p[0],p[1],p[2],label=[name])#,color = dat.colours[name],#linewidth=dat.linewidths[name])

p1 = posHor
ax.plot(p1[:,0],p1[:,1],p1[:,2], color ='black',linewidth = 1.2, label=['2024YR4 - Horizons'])

ax.plot(posAst[:,0],posAst[:,1],posAst[:,2], color = 'orange',linewidth = 1, linestyle = '--', label=['2024YR4 - RK4'])
#ax.plot(posAst_2[:,0],posAst_2[:,1],posAst_2[:,2], color = 'red', linestyle = '--', linewidth = 0.8)
ax.scatter(posAst[0,0],posAst[0,1],posAst[0,2])
ax.scatter(p1[0,0],p1[0,1],p1[0,2])
ax.legend()#fontsize='small')
####


d=np.empty(len(times))
d1=np.empty(len(times))
d2=np.empty(len(times))
for i in range(len(times)):
    d1[i] = np.sqrt(p1[i,0]**2 + p1[i,1]**2 + p1[i,2]**2) #Horizons
    d[i] = np.sqrt(posAst[i,0]**2 + posAst[i,1]**2 + posAst[i,2]**2) #RK4
    d2[i] = np.sqrt(posAst_2[i,0]**2 + posAst_2[i,1]**2 + posAst_2[i,2]**2) #RK5

print(d1[0],d1[1])
print(d[0],d[1])

plt.figure(num=2)
plt.xlabel('Time (s)')
plt.ylabel('Distance from SSB (m)')
plt.plot(times,d,'r', label = 'RK4')
plt.plot(times,d1,'black',linestyle=':', label='Horizons')
plt.plot(times,d2, 'cyan',linestyle='--', label='RK5')
plt.legend()

plt.figure(num=3)
plt.xlabel('Elapsed time (s)', fontsize = 'large')
plt.ylabel('Separation between Horizons\nand RK4 orbits (m)', fontsize = 'large')
plt.plot(times,(d1-d),color = 'r')
#plt.vlines(pos.positions['Sun'][0][-33],-2.5e8,2.5e8,linestyle='dashed')
#plt.plot(times,(d1-d2), color = 'blue')
#plt.plot(times,(d-d2))

#plt.figure(num=4)
#plt.xlabel('Elapsed time (s)')
#plt.ylabel('')
#l = np.empty(len(times))
#for i in range(len(times)):
#    l[i] = np.sqrt(
#    (p1[i,0]-posAst[i,0])**2+
#    (p1[i,1]-posAst[i,1])**2+
#    (p1[i,2]-posAst[i,2])**2
#    )
#plt.plot(times,l)

plt.figure(num=5)
ax1 = plt.axes(projection = '3d')
s = (-5e12,5e12)
ax1.set_xlim(-0.9e10,-0.6e10)
ax1.set_ylim(1.47e11,1.49e11)
ax1.set_zlim(-3e9,3e9)
ax1.set_xlabel('x(m)')
ax1.set_ylabel('y(m)')
ax1.set_zlabel('z(m)')
for name in dat.names:
    p = pos.positions[name][1:4]
    ax1.scatter(p[0,-1],p[1,-1],p[2,-1],color = dat.colours[name],linewidth=dat.linewidths[name])

p1 = pos.asteroid[1:4]
ax1.scatter(p1[0,-1],p1[1,-1],p1[2,-1], color ='black',linewidth = 1)
ax1.scatter(posAst[-1,0],posAst[-1,1],posAst[-1,2], color = 'r',linestyle=':',linewidth = 1)
ax1.scatter(posAst_2[-1,0],posAst_2[-1,1],posAst_2[-1,2], color = 'g',marker='x',linewidth = 0.8)


######
plt.figure(num=6)
ax2 = plt.axes(projection='3d')
ax2.set_xlim(-3.6e9,-2.8e9)
ax2.set_ylim(1.47e11,1.478e11)
ax2.set_zlim(-4e8,4e8)
ax2.set_xlabel('x(m)')
ax2.set_ylabel('y(m)')
ax2.set_zlabel('z(m)')
for name in dat.names:
    p2 = pos.positions[name][1:4]
    p3 = pos.asteroid[1:4]
    for i in range(-36,-31,1):
        ax2.scatter(p2[0,i],p2[1,i],p2[2,i],color = dat.colours[name],linewidth=dat.linewidths[name])
        ax2.scatter(p3[0,i],p3[1,i],p3[2,i])


plt.figure(num=7)
ax3 = plt.axes(projection='3d')
s = (-1e12,1e12)
ax3.set_xlim(s)
ax3.set_ylim(s)
ax3.set_zlim(s)
for name in dat.names:
    p = pos.positions[name][1:4]
    lines = ax3.plot(p[0],p[1],p[2],label=[name])#,color = dat.colours[name],#linewidth=dat.linewidths[name])

p4 = pos.asteroid[1:4]
ax3.plot(p4[0],p4[1],p4[2], color ='black',linewidth = 1.2, label=['2024YR4 - Horizons'])

ax3.plot(ast_pos[:,0],ast_pos[:,1],ast_pos[:,2], color = 'red',linewidth = 1, linestyle = ':', label=['2024YR4 - RK4'])

plt.figure(num=8)
plt.xlabel('Elapsed time (s)', fontsize = 'large')
plt.ylabel('Separation between Horizons\nand RK4 orbits (m)', fontsize = 'large')
plt.plot(times_1,(d_hor-d_ast), label = 'Adaptive timestep')
plt.plot(times,(d1-d), label = 'Timestep = 1 hour')
plt.legend()
end = time.perf_counter()

elapsed = end - start
print(f"{elapsed:.6f} seconds")

plt.show()

