# %%
import numpy as np
import matplotlib.pyplot as plt

K = 4.47405

colour_range = [[-0.30, 0.00], [0.00, 0.05], [0.05, 0.10], [0.10, 0.15], [0.15, 0.20], [0.20, 0.25], 
                [0.25, 0.30], [0.30, 0.35], [0.35, 0.40], [0.40, 0.45], [0.45, 0.50], [0.50, 0.55],
                [0.55, 0.60], [0.60, 0.65], [0.65, 0.70], [0.70, 0.75], [0.75, 0.80]]

distance_limits = [1200, 1100, 1000, 900, 700, 630, 560, 500, 480, 450, 430, 
                   400, 360, 310, 300, 250, 220, 200, 160, 150]

abs_mag = G + (5 * np.log10(parallax / 100))

parallax_filter = parallax_error * 10 < parallax
mag_filter = np.logical_and(abs_mag > ( c * 3.5 - 1.5), (abs_mag < (c * 4.5 + 1.5)))

# %%

curve_max_error, curve_max, rho_error, residuals_all, residuals_x, rhos, colour_means, number_of_stars = [], [], [], [], [], [], [], []

axis_lim = 400

fig, axs = plt.subplots(6, 3, figsize=(13, 18))
fig.subplots_adjust(hspace = 0.6, wspace=0.3)
axs = axs.ravel()
fig.delaxes(axs[17])

for i in range(len(colour_range)):
    
    r_max = distance_limits[i]
    distances = 1000 / parallax
    distance_filter = distances < r_max
    c_filter = np.logical_and(c >= colour_range[i][0], c < colour_range[i][1])
    star_filter = c_filter & parallax_filter & mag_filter & distance_filter

    f_l = np.radians(l[star_filter])
    f_b = np.radians(b[star_filter])
    f_p = parallax[star_filter]
    f_mul = mul[star_filter]
    f_mub = mub[star_filter]
    f_abs_mag = abs_mag[star_filter]
    f_distances = 1000 / f_p
    
    z_ranges = np.arange(-r_max, r_max + 10, 10)
    
    volumes = [(np.pi * (r_max ** 2) * (z_ranges[j + 1] - z_ranges[j])) -
               (np.pi * ((z_ranges[j + 1]) ** 3 - (z_ranges[j] ** 3)) / 3)
               for j in range(len(z_ranges) - 1)]

    z = f_distances * np.sin(f_b)
    
    num_stars_per_range = np.histogram(z, z_ranges)[0]

    num_stars_per_range[num_stars_per_range == 0] = 1

    n_z = num_stars_per_range / volumes
    
    z_range_means = np.arange(-r_max + 5, r_max + 5, 10)
    
    range_ends_filter = np.abs(z_range_means) < axis_lim
    
    x = np.array(z_range_means[range_ends_filter])
    y = np.array(np.log(n_z[range_ends_filter]))
    residuals_x.append(x)

    axs[i].plot(x, y)
    axs[i].set_title(f'{colour_range[i][0]} to {colour_range[i][1]} BP-RP')
    axs[i].set_ylabel(r'log density (pc$^{-3}$)')
    axs[i].set_xlabel('z (pc)')    

    Q, Q_cov = np.polyfit(np.array(z_range_means[np.abs(z_range_means) < axis_lim]),
                          np.array(np.log(n_z[np.abs(z_range_means) < axis_lim])), 2, cov=True)
    c1, c2, c3 = Q
    ccc = np.poly1d([c1, c2, c3])
    
    plot_max = r_max if np.abs(r_max) < axis_lim else axis_lim
    xl = np.arange(-plot_max, plot_max, 0.01)
    yl = ccc(xl)
    center_line = xl[np.where(yl == np.amax(yl))]
    curve_max.append(xl[np.where(yl == np.amax(yl))][0])
    axs[i].plot(xl, yl)
    axs[i].axvline(x=center_line, linestyle='--', color='black', label=f'${np.round_(center_line[0], decimals=2) }$ pc')   
    axs[i].legend()
    
    residuals = np.array([y[j] - yl[0::1000][j] for j in range(len(y))])
    residuals_all.append(residuals)
    curve_max_error.append(residuals[np.where(y == np.amax(y))][0])

    Q_error = np.sqrt(np.diag(Q_cov))[0]

    v_u = np.transpose([np.cos(f_b) * np.cos(f_l), np.cos(f_b) * np.sin(f_l), np.sin(f_b)])
    v_l = np.transpose([- np.sin(f_l), np.cos(f_l), np.zeros(len(f_l))])
    v_b = np.transpose([- np.sin(f_b) * np.cos(f_l), - np.sin(f_b) * np.sin(f_l), np.cos(f_b)])

    tau = v_l * K * f_mul[:, None] / f_p[:, None] + v_b * K * f_mub[:, None] / f_p[:, None]

    T = [np.identity(3) - np.outer(v_u[j], v_u[j]) for j in range(len(v_u))]

    tau_mean = np.mean(tau, axis=(0))

    T_mean = np.mean(T, axis=(0))

    V_mean = np.dot(np.linalg.inv(np.array(T_mean)),  np.array(tau_mean))

    delta_tau = tau - np.matmul(np.array(T), V_mean)

    B = np.mean([np.outer(delta_tau[i], delta_tau[i]) for i in range(len(delta_tau))], axis=(0))

    T_kmln = [np.tensordot(np.array(T[j]), np.array(T[j]).T, axes=0) for j in range(len(T))]
    T_kmln_mean = np.mean(T_kmln, axis=0)

    D = np.linalg.tensorsolve(T_kmln_mean, B, axes=(0, 2))

    w_sigma = D[2][2]

    colour_means.append(np.mean(c[c_filter]))

    rhos.append(- w_sigma / (4 * np.pi * 4.301 * 10 ** -3) * 2 * c1)
    rho_error.append(- w_sigma / (4 * np.pi * 4.301 * 10 ** -3) * 2 * (c1 + Q_error))
    number_of_stars.append(len(f_b))
    
plt.savefig('stellar_density', bbox_inches='tight')
    
# %%
    
plt.figure(figsize=(9, 6))
plt.title('Mass density for each colour range')
plt.ylabel(r'$\rho_0 (M_s pc^{-3})$')
plt.xlabel('mean BP-RP')
plt.grid('both', axis='y')
plt.xticks(np.arange(-0.3, 0.8, 0.1))
plt.scatter(colour_means, rhos)
plt.errorbar(colour_means, rhos, yerr=rho_error, fmt='o', capsize=2)
plt.savefig('rho', bbox_inches='tight')

# %%

residuals_sum = [np.sum(np.abs(residuals_all[i])) for i in range(len(residuals_all))]
    
plt.figure(figsize=(9, 6))
plt.title('Distance from galactic midplane')
plt.ylabel(r'z (pc)')
plt.xlabel('mean BP-RP')
plt.grid('both', axis='y')
plt.xticks(np.arange(-0.3, 0.8, 0.1))
plt.scatter(colour_means, curve_max)
plt.errorbar(colour_means, curve_max, yerr=residuals_sum, fmt='o', capsize=2)
plt.axhline(y = -20.8, linestyle='--', linewidth=0.8, color='r')
plt.savefig('z', bbox_inches='tight')


# %%

fig, axs = plt.subplots(6, 3, figsize=(13, 18))
fig.subplots_adjust(hspace = 0.6, wspace=0.3)
axs = axs.ravel()
fig.delaxes(axs[17])

for i in range(len(colour_range)):
    axs[i].plot(residuals_x[i], residuals_all[i])
    axs[i].set_title(f'{colour_range[i][0]} to {colour_range[i][1]} BP-RP')
    axs[i].set_ylabel(r'log(density) - fit ($pc^{-3}$)')
    axs[i].set_xlabel('z (pc)')
    axs[i].set_ylim(-2, 2)
    axs[i].axhline(y = 0, linestyle='--', linewidth=0.8, color='black')
    
plt.savefig('residuals', bbox_inches='tight')
    
# %%
    
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
plt.xticks(np.arange(-0.3, 0.8, 0.1))
plt.yticks(np.arange(-3, 6, 0.5))
plt.grid('both')
plt.savefig('hrdiagram_after', bbox_inches='tight')