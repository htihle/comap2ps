import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
from mpi4py import MPI
import corner
import h5py
import sys

import tools
import master
import comap2xs
import map_obj

try:
    mapname = sys.argv[1]
    mapname2 = sys.argv[2]
except IndexError:
    print('Missing filename!')
    print('Usage: python ps_script.py mapname mapname2')
    sys.exit(1)

maps = map_obj.MapObj(mapname)
maps2 = map_obj.MapObj(mapname2)
#for det in range(1,20):
my_xs = comap2xs.comap2xs(maps, maps2, decimate_z=256)

# my_ps.pseudo_ps = False
# my_ps.calculate_mode_mixing_matrix()
# np.save('M_inv', my_ps.M_inv)

xs, k = my_xs.calculate_xs()

#rms_mean, rms_sig = my_ps.noise_sims_from_file(mapname)
#print(rms_mean)
#print(ps)
#print(rms_sig)
rms_mean, rms_sig = my_xs.run_noise_sims(100)
# print(rms_mean)

#ps_th = np.load('ps.npy')
#k_th = np.load('k.npy')
ps_th = np.load('ps_beam.npy')
k_th = np.load('k_beam.npy')
# ps_th = np.load('ps_cube.npy')
# k_th = np.load('k_cube.npy')
h = 0.7
# cube = np.load('cube_map.npy')

# theory, k2, nmodes = tools.compute_power_spec3d(
#     cube, my_xs.k_bin_edges, my_xs.dx,
#     my_xs.dy, my_xs.dz)

# print(ps)
# print(rms_sig)

np.savetxt('xs_co2.txt', xs * 1e12)
np.savetxt('k_xs.txt', k)

np.savetxt('ps_th.txt', ps_th * h ** 3)
np.savetxt('k_th.txt', k_th / h)
xs = xs[1:] * 1e12
k = k[1:]
#theory = theory[1:]
rms_mean = rms_mean[1:] * 1e12
rms_sig = rms_sig[1:] * 1e12

lim = np.mean(np.abs(xs[4:])) * 4
    
fig = plt.figure()

ax1 = fig.add_subplot(211)
ax1.errorbar(k, xs, rms_sig, fmt='o', label=r'$\tilde{C}_{data}(k)$')
ax1.plot(k, 0 * rms_mean, 'k', label=r'$\tilde{C}_{noise}(k)$', alpha=0.4)
#ax1.plot(k, theory[1:] * 100, 'r--', label=r'$100 * P_{Theory}(k)$')
ax1.plot(k_th / h, ps_th * h ** 3 * 10, 'r--', label=r'$10 * P_{Theory}(k)$')
ax1.set_ylabel(r'$\tilde{C}(k)$ [$\mu$K${}^2$ (Mpc / $h$)${}^3$]')
ax1.set_ylim(-lim, lim)              # ax1.set_ylim(0, 0.1)
plt.legend()

ax2 = fig.add_subplot(212)
ax2.errorbar(k, xs / rms_sig, rms_sig / rms_sig, fmt='o', label=r'$\tilde{C}_{data}(k)$')
ax2.plot(k, 0 * rms_mean, 'k', alpha=0.4)
ax2.set_ylabel(r'$\tilde{C}(k) / \sigma_\tilde{C}$')
ax2.set_xlabel(r'$k$ [(Mpc / $h$)${}^{-1}$]')
ax2.set_ylim(-10, 10)
plt.legend()


plt.savefig('xs_halfmission_split.pdf', bbox_inches='tight')

# plt.figure()
# h = 0.7
# # plt.plot(k / h, ps * 100 * h ** 3)
# plt.loglog(k2 * h, k2 ** 3 * theory / (2 * np.pi ** 2))
plt.show()
# print('Done with feed ', det)
