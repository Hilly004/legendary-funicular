import numpy as np
import matplotlib.pyplot as plt

######
data = np.load('Method_Variation_Results.npz',allow_pickle=True)

rk4_res = data['rk4'].item()
rk5_res = data['rk5'].item()
rkdp_res = data['rkdp'].item()
dop_res = data['dop'].item()
true_res = data['true'].item()

fig = plt.figure(figsize = (14,8))
gs = fig.add_gridspec(4, 2, hspace=0.1, wspace=0.005)

es = [0,0.5,0.75,0.6615]

orbit_axes = [
    fig.add_subplot(gs[0,0]),
    fig.add_subplot(gs[1,0]),
    fig.add_subplot(gs[2,0]),
    fig.add_subplot(gs[3,0])
]

ax_err0 = fig.add_subplot(gs[0,1])

error_axes = [
    ax_err0,
    fig.add_subplot(gs[1,1], sharex=ax_err0),
    fig.add_subplot(gs[2,1], sharex=ax_err0),
    fig.add_subplot(gs[3,1], sharex=ax_err0)
]

for (ax_orbit, ax_err, e) in zip(orbit_axes, error_axes, es):
    t = rk4_res[e]['times']
    rk4_ = rk4_res[e]['positions']
    rk5_ = rk5_res[e]['positions']
    rkdp_ = rkdp_res[e]['positions']
    dop_ = dop_res[e]['positions']
    true_ = true_res[e]['positions']

    s = 1e9

    ax_orbit.plot(true_[:,0]/s,true_[:,1]/s,label='True', color = 'black')
    ax_orbit.plot(rk4_[:,0]/s,rk4_[:,1]/s,label='RK4', color = 'red')
    ax_orbit.plot(rk5_[:,0]/s,rk5_[:,1]/s,label='RK5', color = 'blue', linestyle = '-.')        
    ax_orbit.plot(rkdp_[:,0]/s,rkdp_[:,1]/s,label='RKDP', color='green',linestyle = '--')
    ax_orbit.plot(dop_[:,0]/s,dop_[:,1]/s,label='DOP853', color = 'orange',linestyle=':')
    ax_orbit.scatter(0,0)

    ax_orbit.set_xlim((-2e9/s,2e9/s))
    ax_orbit.set_ylim((-1.5e9/s,1.5e9/s))
    ax_orbit.set_xlabel(r'$\mathrm{x} (\times 10^{9}$ m)')
    ax_orbit.set_ylabel(r'$\mathrm{y} (\times 10^{9}$ m)')
    #ax_orbit.label_outer()
    ax_orbit.set_aspect('equal')
    #ax_orbit.set_title(f'e = {e}')

    for ax in orbit_axes[:-1]:
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        ax.set_xlabel('')



    ######

    err_rk4 = np.linalg.norm(rk4_-true_,axis=1)
    err_rk5 = np.linalg.norm(rk5_-true_,axis=1)
    err_rkdp = np.linalg.norm(rkdp_-true_,axis=1)
    err_dop = np.linalg.norm(dop_-true_,axis=1)

    ax_err.plot(t,err_rk4,label='RK4')
    ax_err.plot(t,err_rk5,label='RK5')
    ax_err.plot(t,err_rkdp,label='RKDP')
    ax_err.plot(t,err_dop,label='DOP853')

    ax_err.set_xlabel('Time (s)')
    ax_err.set_ylabel('Error (m)')

    for ax in error_axes:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        #ax.set_yscale('log')

    for ax in error_axes[:-1]:
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        ax.set_xlabel('')

    
    #ax_err.set_title(f'e = {e}')

orbit_axes[0].legend()
error_axes[0].legend()
#plt.tight_layout()
plt.subplots_adjust(wspace=0.1)
plt.show()