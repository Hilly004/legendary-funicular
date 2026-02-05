import numpy as np
import Positions as pos
import matplotlib.pyplot as plt

######

vec_0 = pos.asteroid[1:13][:,0]

mu_r = vec_0[0:3]
sigma_r = vec_0[6:9]
var_r = sigma_r**2

mu_v = vec_0[3:6]
sigma_v = vec_0[9:12]
var_v = sigma_v**2

N=1000

l = np.random.randn(N,3)
#######

cov_r = np.array([[var_r[0],0,0],
                  [0,var_r[1],0],
                  [0,0,var_r[2]]])

L_r = np.linalg.cholesky(cov_r)

p = mu_r + l @ L_r.T

cov_inv = np.linalg.inv(cov_r)

df = p - mu_r

d = np.einsum('ij,jk,ik->i',df,cov_inv,df)

chi2_threshold = 7.81
k = np.sqrt(chi2_threshold)

eigenvals, eigenvectors = np.linalg.eigh(cov_r)

fil = p[d<=chi2_threshold]

#######

u = np.linspace(0,2*np.pi,60)
v = np.linspace(0, np.pi, 30)

x = np.outer(np.cos(u),np.sin(v)) 
y = np.outer(np.sin(u),np.sin(v))
z = np.outer(np.ones_like(u),np.cos(v))

axes_lengths = k*np.sqrt(eigenvals)

ellipsoid = np.stack([x,y,z], axis=-1)
ellipsoid = ellipsoid@np.diag(axes_lengths)
ellipsoid = ellipsoid@eigenvectors.T+ mu_r


#######

plt.figure(num=1)
ax = plt.axes(projection = '3d')

for i in range(len(fil)):
    ax.scatter(fil[i,0],fil[i,1],fil[i,2])

ax.plot_surface(
    ellipsoid[...,0],
    ellipsoid[...,1],
    ellipsoid[...,2],
    alpha = 0.4
)
ax.scatter(mu_r[0],mu_r[1],mu_r[2], color = 'red', marker = 'x', linewidths = 1.2)

plt.show()