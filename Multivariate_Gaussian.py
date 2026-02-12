import numpy as np
import Positions as pos
import matplotlib.pyplot as plt
from tqdm import trange

######

vec_0 = pos.asteroid1[1:19][:,0]    #x,y,x, vx,vy,vz
                                    #errors
                                    #RTN errors


mu_r = np.asarray(vec_0[0:3])
sigma_RTN_r = vec_0[12:15]

mu_v = np.asarray(vec_0[3:6])
sigma_RTN_v = vec_0[15:18]

R_r_hat = mu_r/np.linalg.norm(mu_r)
h_r = np.cross(mu_r,mu_v)
N_r_hat = h_r/np.linalg.norm(h_r)

T_r_hat = np.cross(N_r_hat,R_r_hat)

R_r = np.column_stack((R_r_hat,T_r_hat,N_r_hat))

sigma_r_mat = np.diag([sigma_RTN_r[0]**2,sigma_RTN_r[1]**2,sigma_RTN_r[2]**2])
sigma_v_mat = np.diag([sigma_RTN_v[0]**2,sigma_RTN_v[1]**2,sigma_RTN_v[2]**2])

N=100

l = np.random.randn(N,3)
#######

cov_r = R_r@sigma_r_mat@R_r.T

L_r = np.linalg.cholesky(cov_r)

p = mu_r + l @ L_r.T

cov_inv = np.linalg.inv(cov_r)

df = p - mu_r

d = np.einsum('ij,jk,ik->i',df,cov_inv,df)

chi2_threshold = 7.81
k = np.sqrt(chi2_threshold)

eigenvals, eigenvectors = np.linalg.eigh(cov_r)

fil = p#p[d<=chi2_threshold]

print(sum(d>chi2_threshold))
#######

cov_v = R_r@sigma_v_mat@R_r.T

L_v = np.linalg.cholesky(cov_v)

p_v = mu_v + l @ L_v.T

cov_v_inv = np.linalg.inv(cov_v)

df_v = p_v - mu_v

d_v = np.einsum('ij,jk,ik->i',df_v,cov_v_inv,df_v)

chi2_threshold = 7.81
k = np.sqrt(chi2_threshold)

eigenvals_v, eigenvectors_v = np.linalg.eigh(cov_v)

fil_v = p_v#p[d<=chi2_threshold]

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
ps = fil-mu_r

ax = plt.axes(projection = '3d')
ax.set_aspect('equal')

for i in trange(len(ps)):
    ax.scatter(ps[i,0],ps[i,1],ps[i,2])

ax.plot_surface(
    ellipsoid[...,0],
    ellipsoid[...,1],
    ellipsoid[...,2],
    alpha = 0.4
)
#ax.scatter(mu_r[0],mu_r[1],mu_r[2], color = 'red', marker = 'x', s = 30)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')


plt.figure(num=2)
ax1 = plt.axes(projection='3d')
ax1.set_aspect('equal')

for j in trange(len(fil_v)):
    ax1.scatter(fil_v[j,0],fil_v[j,1],fil_v[j,2])

plt.show()