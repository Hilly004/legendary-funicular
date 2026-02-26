import numpy as np
import Acceleration as acc

dt =10

def timestep(acc):
    a_mag = np.linalg.norm(acc)
    return dt/np.sqrt(a_mag)