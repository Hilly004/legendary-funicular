import Object_Data as dat
import Positions as pos
import numpy as np
import scipy as sc
from scipy.interpolate import CubicSpline

G = 6.6726e-11

planet_splines = {}

for name in dat.names:
    arr = pos.positions[name]

    t = arr[0]   # seconds
    x = arr[1]
    y = arr[2]
    z = arr[3]

    planet_splines[name] = {
        'x': CubicSpline(t, x, extrapolate=False),
        'y': CubicSpline(t, y, extrapolate=False),
        'z': CubicSpline(t, z, extrapolate=False)
    }


def planetary_position(name,t):
        spl = planet_splines[name]
        return np.array([
             spl['x'](t),
             spl['y'](t),
             spl['z'](t)
        ])

def accel(r,t): #t in seconds
    a = np.zeros(3)

    for name in dat.names:
        pp = planetary_position(name,t)
        vec = pp - r
        d = np.linalg.norm(vec)
        a += G*dat.masses[name]*vec/d**3
    return a

