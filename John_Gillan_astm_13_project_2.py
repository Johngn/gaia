# %%
import numpy as np
import matplotlib.pyplot as plt

K = 4.47405

colour_range = [[-0.3, -0.1], [-0.1, -0.05], [-0.05, 0], [0, 0.05], [0.05, 0.10], [0.10, 0.15], [0.15, 0.20], [0.20, 0.25], [0.25, 0.30], [0.30, 0.35], [0.35, 0.40], [0.40, 0.45], [0.45, 0.50], [0.50, 0.55], [0.55, 0.60], [0.60, 0.65], [0.65, 0.70], [0.70, 0.75], [0.75, 0.80]]

abs_mag = G + (5 * np.log10(parallax / 100))

parallax_filter = np.logical_and(parallax_error * 10 < parallax, parallax < 20)
mag_filter = np.logical_and(abs_mag > (c * 3.5 - 1.5), (abs_mag < (c * 4.5 + 1.5)))


#%%

u_means, v_means, w_means, u_sigma, v_sigma, w_sigma, number_of_stars, T_array, colour_means, mag_mins = [], [], [], [], [], [], [], [], [], []

for i in colour_range:

    c_filter = np.logical_and(c >= i[0], c < i[1])
    star_filter = parallax_filter & c_filter & mag_filter
    
    f_l = np.radians(l[star_filter])
    f_b = np.radians(b[star_filter])
    f_p = parallax[star_filter]
    # f_mul = mul[star_filter]
    # f_mub = mub[star_filter]
    f_abs_mag = abs_mag[star_filter]
    
    min_abs_mag = np.amin(f_abs_mag)
    min_parallax = np.amin(f_p)
    mag_mins.append(min_parallax)
    
    v_u = np.transpose([np.cos(f_b) * np.cos(f_l), np.cos(f_b) * np.sin(f_l), np.sin(f_b)])
    v_l = np.transpose([- np.sin(f_l), np.cos(f_l), np.zeros(len(f_l))])
    v_b = np.transpose([- np.sin(f_b) * np.cos(f_l), - np.sin(f_b) * np.sin(f_l), np.cos(f_b)])
    
    tau = v_l * K * f_mul[:, None] / f_p[:, None] + v_b * K * f_mub[:, None] / f_p[:, None]

    T = [np.identity(3) - np.outer(v_u[j], v_u[j]) for j in range(len(v_u))]

    tau_mean = np.mean(tau, axis=(0))

    T_mean = np.mean(T, axis=(0))

    V_mean = np.dot(np.linalg.inv(np.array(T_mean)),  np.array(tau_mean))

    delta_tau = tau - np.matmul(np.array(T), V_mean)

    u_means.append(V_mean[0])
    v_means.append(V_mean[1])
    w_means.append(V_mean[2])

    B = np.mean([np.outer(delta_tau[i], delta_tau[i]) for i in range(len(delta_tau))], axis=(0))
   
    T_kmln = [np.tensordot(np.array(T[j]), np.array(T[j]).T, axes=0) for j in range(len(T))]
    T_kmln_mean = np.mean(T_kmln, axis=0)

    D = np.linalg.tensorsolve(T_kmln_mean, B, axes=(0, 2))

    u_sigma.append(D[0][0])
    v_sigma.append(D[1][1])
    w_sigma.append(D[2][2])
    
    number_of_stars.append(len(f_b))
    
    T_array.append(T_mean)
    
    colour_means.append(np.mean(c[c_filter]))
    
#%%
plot_labels = [-0.20,-0.075,-0.025,0.025,0.075,0.125,0.175,0.225,
               0.275,0.325,0.375,0.425,0.475,0.525,0.575,
               0.625,0.675,0.725,0.775 ]
plt.figure(figsize=(13, 5))
x = [1 / 2 * np.arctan(2 * D_array[i][0][1] / (D_array[i][0][0] - D_array[i][1][1]) ) for i in range(len(D_array))]
plt.plot(x)
plt.xlabel('mean BP-RP')
plt.ylabel(r'$\Theta$')
plt.title('Vertex Deviation')
plt.xticks(ticks=np.arange(0, len(u_means), 1), labels=plot_labels)
plt.savefig('vertex')
#%%
plt.figure(figsize=(13, 5))
x = np.divide(np.sqrt(v_sigma), np.sqrt(u_sigma))
plt.plot(x, label=r'$\sigma_v/\sigma_u$ (measured)')

par = np.polyfit(np.sqrt(v_sigma), np.sqrt(u_sigma), 1, full=True)
slope=par[0][0]
intercept=par[0][1]
xl = [min(u_sigma), max(u_sigma)]
yl = [slope*xx + intercept  for xx in xl]
plt.plot(xl, yl, label=f'slope = {np.round(slope, 5)}')

plt.title(r'$\sigma_v/\sigma_u$')
plt.xlabel('mean BP-RP')
plt.ylim(0, 1.5)
np.arange(20)
plt.plot(np.zeros(20) + 0.68, label=r'$\sigma_v/\sigma_u$ (predicted, 0.68)')
plt.xticks(ticks=np.arange(0, len(u_means), 1), labels=plot_labels)
plt.legend()
plt.savefig('sigma_over_sigma')
#%%
plt.figure(figsize=(12, 5))
plt.loglog(plot_labels, np.sqrt(u_sigma), label=r'$\sigma_u$')
plt.loglog(plot_labels, np.sqrt(v_sigma), label=r'$\sigma_v$')
plt.loglog(plot_labels, np.sqrt(w_sigma), label=r'$\sigma_w$')
plt.legend()
plt.xlabel('BP-RP')
plt.ylabel(r'$\sqrt{velocity dispersion}$')
plt.title(r'$\sigma$')

plt.savefig('alpha')
#%%
fig, axs = plt.subplots(3, 1, figsize=(13, 13))
par = np.polyfit(u_sigma, u_means, 1, full=True)
slope=par[0][0]
intercept=par[0][1]
axs[0].set_xlabel(r'$\sigma_u^2 (km / s)$')
axs[0].set_ylabel(r'$\langle u \rangle (km / s)$')
xl = [0, max(u_sigma)]
yl = [slope*xx + intercept  for xx in xl]
axs[0].plot(xl, yl, label=f'slope = {np.round(slope, 5)}')
axs[0].scatter(u_sigma, u_means)
axs[0].legend()

par = np.polyfit(u_sigma,v_means , 1, full=True)
slope=par[0][0]
intercept=par[0][1]
xl = [0, max(u_sigma)]
yl = [slope*xx + intercept  for xx in xl]
axs[1].plot(xl, yl, label=f'slope = {np.round(slope, 5)}')
axs[1].set_xlabel(r'$\sigma_u^2 (km / s)$')
axs[1].set_ylabel(r'$\langle v \rangle (km / s)$')
axs[1].scatter(u_sigma, v_means)
axs[1].legend()

par = np.polyfit(u_sigma, w_means, 1, full=True)
slope=par[0][0]
intercept=par[0][1]
xl = [0, max(u_sigma)]
yl = [slope*xx + intercept  for xx in xl]
axs[2].set_xlabel(r'$\sigma_u^2 (km / s)$')
axs[2].set_ylabel(r'$\langle w \rangle (km / s)$')
axs[2].plot(xl, yl, label=f'slope = {np.round(slope, 5)}')
axs[2].scatter(u_sigma, w_means)
axs[2].legend()

fig.savefig('sigma_slope')
#%%
plt.figure(figsize=(12, 5))
plt.plot(u_means, label=r'$\langle u \rangle $')
plt.plot(v_means, label=r'$\langle v \rangle $')
plt.plot(w_means, label=r'$\langle w \rangle $')
plt.title("Mean Velocity")
plt.legend()
plt.xticks(ticks=np.arange(0, len(u_means), 1), labels=plot_labels)
plt.xlabel(r'mean age $(10^8 yr)$')
plt.ylabel('velocity (km / s)')

plt.savefig('mean_velocity_age')
#%%

plt.figure(figsize=(12, 5))
plt.plot(u_sigma, label=r'$\sigma_u^2$')
plt.plot(v_sigma, label=r'$\sigma_v^2$')
plt.plot(w_sigma, label=r'$\sigma_w^2$')
plt.title("Velocity Dispersion")
plt.legend()
plt.xticks(ticks=np.arange(0, len(u_means), 1), labels=plot_labels)
plt.xlabel(r'mean age $(10^8 yr)$')
plt.ylabel('velocity dispersion (km / s)')
plt.savefig('velocity_dispersion_age')
#%%

x = c[mag_filter &  parallax_filter]
y = abs_mag[mag_filter &  parallax_filter]

plt.figure(figsize=(10, 10))
plt.hist2d(x, y, cmap="jet", cmin=1, bins=3000)
plt.colorbar()
#plt.xlim(-0.5, 2.0)
#plt.ylim(-4.5, 10)
plt.xlim(-0.3, 0.8)
plt.ylim(-3, 6)
plt.gca().invert_yaxis()
plt.title("HR Diagram")
plt.xlabel('BP - RP [mag]')
plt.ylabel(r'$M_{g} [mag]$')
plt.savefig('hrdiagram_after', bbox_inches='tight')
