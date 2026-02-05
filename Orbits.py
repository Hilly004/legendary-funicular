import Runge_Kutta_4 as rk4
import numpy as np
import Positions as pos
import matplotlib.pyplot as plt
import Object_Data as dat
import Runge_Kutta_5 as rk5

dt = 3600
times = np.arange(0,pos.positions['Sun'][0][-1],dt)
posAst = np.zeros((len(times),3))
posAst_2 = np.zeros((len(times),3))

r=np.array(pos.asteroid[1:7,0])
r_2=np.array(pos.asteroid[1:7,0])
#r = np.array([-3.22749253e+11,  6.53523922e+10, -1.90546027e+10, -1.68467588e+04, -1.29695567e+04, -1.02237685e+03])
#r = np.array([0.5e13,0.5e13,0.5e13,-1e4,-1e4,0])
for i,t in enumerate(times):
    posAst[i] = r[:3]
    posAst_2[i] = r_2[:3]

    r = rk4.rk4(r,t,dt)
    r_2 = rk5.rk5(r,t,dt)

#print(len(posAst))
#print(len(posAst_2))


plt.figure(num=1)
ax = plt.axes(projection = '3d')
s = (-1e12,1e12)
ax.set_xlim(s)
ax.set_ylim(s)
ax.set_zlim(s)
ax.set_xlabel('x(m)')
ax.set_ylabel('y(m)')
ax.set_zlabel('z(m)')
for name in dat.names:
    p = pos.positions[name][1:4]
    ax.plot(p[0],p[1],p[2],color = dat.colours[name],linewidth=dat.linewidths[name])

p1 = pos.asteroid[1:4]
ax.plot(p1[0],p1[1],p1[2], color ='black',linewidth = 0.8)

ax.plot(posAst[:,0],posAst[:,1],posAst[:,2], color = 'r',linestyle=':',linewidth = 0.8)
ax.plot(posAst_2[:,0],posAst_2[:,1],posAst_2[:,2], color = 'blue', linestyle = '--', linewidth = 0.6)

####

d=np.empty(len(times))
d1=np.empty(len(times))
d2=np.empty(len(times))
for i in range(len(times)):
    d[i] = np.sqrt(posAst[i,0]**2+posAst[i,1]**2+posAst[i,2]**2) #RK4
    d1[i] = np.sqrt(p1[0,i]**2+p1[1,i]**2+p1[2,i]**2) #Horizons
    d2[i] = np.sqrt(posAst_2[i,0]**2+posAst_2[i,1]**2+posAst_2[i,2]**2) #RK5

plt.figure(num=2)
plt.xlabel('Time (s)')
plt.ylabel('Distance from SSB (m)')
plt.plot(times,d,'r')
plt.plot(times,d1,'black',linestyle=':')
plt.plot(times,d2, 'cyan',linestyle='--')

plt.figure(num=3)
plt.xlabel('Time (s)')
plt.ylabel('Separation (m)')
plt.plot(times,(d1-d),color = 'r')
plt.plot(times,(d1-d2), color = 'blue')

plt.figure(num=4)
l = np.empty(len(times))
for i in range(len(times)):
    l[i] = np.sqrt(
    (p1[0,i]-posAst[i,0])**2+
    (p1[1,i]-posAst[i,1])**2+
    (p1[2,i]-posAst[i,2])**2
    )
plt.plot(times,l)

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
print(d-d2)

plt.show()