import Runge_Kutta_4 as rk4
import numpy as np
import Positions as pos
import matplotlib.pyplot as plt
import Object_Data as dat
import Runge_Kutta_5 as rk5
import Acceleration as acc
from tqdm import tqdm,trange

steps = np.arange(200,2001,200)

positions_Hor = {}
positions_rk4 = {}
positions_rk5 = {}

def propagate_positions(step):
    times = np.arange(0,pos.positions['Sun'][0][-1],step)
    pos_Hor = np.zeros((len(times),3))
    pos_rk4 = np.zeros((len(times),3))
    pos_rk5 = np.zeros((len(times),3))

    r_Hor = np.array(pos.asteroid[1:7,0])
    r_rk4 = np.array(pos.asteroid[1:7,0])
    r_rk5 = np.array(pos.asteroid[1:7,0])

    for i,t in enumerate(tqdm(times, desc='Propagation - '+str(step))):
        pos_Hor[i] = r_Hor[:3]
        pos_rk4[i] = r_rk4[:3]
        pos_rk5[i] = r_rk5[:3]

        r_Hor = acc.planetary_position_1(t)
        r_rk4 = rk4.rk4(r_rk4,t,step)
        r_rk5 = rk5.rk5(r_rk5,t,step)
    return pos_Hor, pos_rk4, pos_rk5



for step in tqdm(steps, desc='Step size'):
    positions_Hor[step], positions_rk4[step], positions_rk5[step] = propagate_positions(step)
