import Runge_Kutta_4 as rk4
import numpy as np
import Positions as pos
import matplotlib.pyplot as plt
import Object_Data as dat
import Runge_Kutta_5 as rk5
import Multivariate_Gaussian as gaus
from tqdm import trange, tqdm
import time

start = time.perf_counter()
  
dt = 3600
n = gaus.N
times = np.arange(0,pos.positions['Sun'][0][-1],dt)

posAst = np.zeros((n,n,len(times),3))

#posAst_2 = np.zeros((len(times),3))

vel_0 = np.array(pos.asteroid[4:7,0])
for j in trange(n):
    for l in range(n):
        r = np.concatenate((gaus.p[j],gaus.p_v[l]))
        for i,t in enumerate(tqdm(times)):
            posAst[j,l,i] = r[:3]
            #posAst_2[i] = r_2[:3]

            r = rk4.rk4(r,t,dt)
            #r_2 = rk5.rk5(r,t,dt)

plt.figure(num=1)
ax = plt.axes(projection='3d')

for k in trange(n):
    for p in range(n):
        ax.plot(posAst[k,p,:,0],posAst[k,p,:,1],posAst[k,p,:,2])

######

plt.figure(num=2,dpi =200)
ax1 = plt.axes(projection = '3d')
s = (-5e12,5e12)

ax1.set_xlim(-3.6e9,-2.8e9)
ax1.set_ylim(1.47e11,1.478e11)
ax1.set_zlim(-4e8,4e8)
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
ax1.set_zlabel('z (m)')
ax1.set_xticks([-3.5e9,-3.4e9,-3.3e9,-3.2e9,-3.1e9,-3.0e9,-2.9e9])


p1 = pos.positions['Earth'][1:4]
p2 = pos.positions['Moon'][1:4]
p_ast = pos.asteroid[1:4]

for i in trange(-36,-33,1):
        ax1.scatter(p1[0,i],p1[1,i],p1[2,i],color = dat.colours['Earth'])
        ax1.scatter(p2[0,i],p2[1,i],p2[2,i],color = dat.colours['Moon'])
        ax1.scatter(posAst[:,:,i,0],posAst[:,:,i,1],posAst[:,:,i,2])
        ax1.scatter(p_ast[0,i],p_ast[1,i],p_ast[2,i], color = 'r')
ax1.scatter(p1[0,-35],p1[1,-35],p1[2,-35],color = dat.colours['Earth'],label = 'Earth')
ax1.scatter(p2[0,-35],p2[1,-35],p2[2,-35],color = dat.colours['Moon'], label = 'Moon')
ax1.scatter(p_ast[0,-35],p_ast[1,-35],p_ast[2,-35], color = 'r', label = '2024 YR4 - Horizons')
ax1.legend()
#ax1.scatter(p1[0,-1],p1[1,-1],p1[2,-1], color ='black',linewidth = 1)

plt.figure(num=3)
dist = np.linalg.norm(posAst,axis=3)
for l in trange(n):
     for m in range(n):
          plt.plot(times, dist[l,m])


plt.xlabel('Time (s)')
plt.ylabel('Distance from SSB(m)')

plt.figure(num=4)
plt.xlabel('Time (s)')
plt.ylabel('Distance from average radial distance (m)')
av = np.mean(dist,axis=(0,1))
for f in trange(n):
    for g in range(n):
         plt.plot(times,(dist[f,g]-av))
#plt.figure(num=3)

#d=np.empty((n,len(times)))
#for j in range(n):
#    for i in range(len(times)):
#        d[j,i] = np.sqrt(posAst[j,i,0]**2+posAst[j,i,1]**2+posAst[j,i,2]**2)
#
#for k in range(n):
#    plt.plot(times,d[k])
#

end = time.perf_counter()

elapsed = end - start

print(f"{elapsed:.6f} seconds")

plt.show()