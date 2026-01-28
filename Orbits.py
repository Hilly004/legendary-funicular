import Runge_Kutta_4 as rk4
import numpy as np
import Positions as pos
import matplotlib.pyplot as plt
import Object_Data as dat

dt = 2
times = np.arange(0,2000,dt)

posAst = np.zeros((len(times),3))

r0 = [-3.227492530630914E+11,6.535239218874960E+10,-1.905460268196982E+10,-1.684675875793478E+04,-1.296955665373355E+04,-1.022376845314192E+03]

r = np.array(r0)

for i,t in enumerate(times):
    posAst[i] = r[:3]
    r = rk4.rk4(r,t,dt)


ax = plt.axes(projection = '3d')
s = (-1e12,1e12)
ax.set_xlim(s)
ax.set_ylim(s)
ax.set_zlim(s)

for name in dat.names:
    p = pos.positions[name][1:4]
    ax.plot(p[0],p[1],p[2])

ax.scatter(posAst[0],posAst[1],posAst[2], color = 'r')

plt.show()