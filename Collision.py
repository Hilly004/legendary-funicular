import numpy as np
import Positions as pos
import Runge_Kutta_4 as rk4
import matplotlib.pyplot as plt

#######
dt = 3600
times = np.arange(0,pos.positions['Sun'][0][-1],dt)

posAst = np.zeros((len(times),3))

r=np.array(pos.asteroid[1:7,0])

#r = np.array([-3.22749253e+11,  6.53523922e+10, -1.90546027e+10, -1.68467588e+04, -1.29695567e+04, -1.02237685e+03])
#r = np.array([0.5e13,0.5e13,0.5e13,-1e4,-1e4,0])
for i,t in enumerate(times):
    posAst[i] = r[:3]
    r = rk4.rk4(r,t,dt)

r_m_a = np.empty(len(times))
vec_moon = pos.positions['Moon'][1:4]
for i in range(len(times)):
    r_m_a[i] = np.sqrt(
        vec_moon[0,i]**2+vec_moon[1,i]**2+vec_moon[2,i]**2)
    -np.sqrt(
        posAst[i,0]**2+posAst[i,1]**2+posAst[i,2]**2)
print(min(r_m_a))

plt.plot(times,r_m_a)

plt.show()