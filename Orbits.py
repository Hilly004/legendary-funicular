import Runge_Kutta_4 as rk4
import numpy as np
import Positions as pos
import matplotlib.pyplot as plt
import Object_Data as dat

dt = 6000
times = np.arange(0,pos.positions['Sun'][0][-1],dt)
posAst = np.zeros((len(times),3))

r=np.array(pos.asteroid[1:7,0])

#r = np.array([-3.22749253e+11,  6.53523922e+10, -1.90546027e+10, -1.68467588e+04, -1.29695567e+04, -1.02237685e+03])
#r = np.array([0.5e13,0.5e13,0.5e13,-1e4,-1e4,0])
for i,t in enumerate(times):
    posAst[i] = r[:3]
    r = rk4.rk4(r,t,dt)

plt.figure(num=1)
ax = plt.axes(projection = '3d')
s = (-5e12,5e12)
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

ax.plot(posAst[:,0],posAst[:,1],posAst[:,2], color = 'r', linewidth = 0.8)

####

d=np.empty(len(times))
d1=np.empty(len(times))
for i in range(len(times)):
    d[i] = np.sqrt(posAst[i,0]**2+posAst[i,1]**2+posAst[i,2]**2)
    d1[i] = np.sqrt(p1[0,i]**2+p1[1,i]**2+p1[2,i]**2)

plt.figure(num=2)
plt.xlabel('Time (s)')
plt.ylabel('Distance from SSB (m)')
plt.plot(times,d,'r')
plt.plot(times,d1,'black',linestyle=':')

plt.figure(num=3)
plt.xlabel('Time (s)')
plt.ylabel('Separation (m)')
plt.plot(times,(d1-d))

plt.figure(num=4)
l = np.empty(len(times))
for i in range(len(times)):
    l[i] = np.sqrt(
    (p1[0,i]-posAst[i,0])**2+
    (p1[1,i]-posAst[i,1])**2+
    (p1[2,i]-posAst[i,2])**2
    )
plt.plot(times,l)


plt.show()

