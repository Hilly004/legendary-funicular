import Data_Extraction as datex
import Object_Data as dat

folder = '/Users/finleyhill/Documents/University/Level 3/CP/Planetary Data/dt=1hour/'
positions = {}

def load(name):
    return datex.out(folder+name+'.txt')

for name in dat.names:
    positions[name] = load(name)

def load1(name):
    return datex.out1(folder+name+'.txt')

asteroid = load('2024YR4')
asteroid1 = load1('2024YR4_RTN')