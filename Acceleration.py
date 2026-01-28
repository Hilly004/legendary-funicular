import Object_Data as dat
import Positions as pos
import numpy as np

G = 6.6726e-11

def accel(r,t):
    a = np.zeros(3)
    rt = divmod(t,1)
    it = int(rt[0])
    rem = rt[1]
    for name in dat.names:
        arr1 = np.array(pos.positions[name])
        arr2 = arr1[1:4,it]+((arr1[1:4,it+1]-arr1[1:4,it])*rem)
        vec = arr2[:3] - np.array(r)
        d = np.sqrt(vec[0]**2+vec[1]**2+vec[2]**2)
        a += G*dat.masses[name]*vec/d**3
    return a

