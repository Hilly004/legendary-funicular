import Multivariate_Gaussian as mg 
import Runge_Kutta_4 as rk4
import numpy as np
import Positions as pos
import matplotlib.pyplot as plt
import Object_Data as dat
import Runge_Kutta_5 as rk5
from tqdm import trange, tqdm
import time

start = time.perf_counter()

###

data = np.load('Multivariate_Gaussian_N_2.npz')

n=2

times = np.arange(0,pos.positions['Sun'][0][-1],dt)

p_m = pos.positions['Moon'][1:4]
p_e = pos.positions['Earth'][1:4]

ps = data['positions'].item()
vs = data['velocities'].item()

for i in range(n):
    for j in range(n):
        r0 = np.concatenate((ps[i],vs[j]))
        for i, t in enumerate(times):
            
