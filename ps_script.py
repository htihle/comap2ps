import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
from mpi4py import MPI
import corner
import h5py

import tools
import master
import comap2ps

det = 9
my_ps = comap2ps.comap2ps('co4_map.h5', decimate_z=32, det=det)

my_ps.calculate_mode_mixing_matrix()
np.save('M_inv', my_ps.M_inv)
# print(my_ps.w)
ps, k = my_ps.calculate_ps()
rms_mean, rms_sig = my_ps.run_noise_sims(1000)

n_cut = 4

# ps = ps[n_cut:]
# k = k[n_cut:]
# rms_mean = rms_mean[n_cut:]
# rms_sig = rms_sig[n_cut:]
# print(ps)
# print(rms_mean)
# print(rms_sig)

# print(my_ps.dx)
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.errorbar(k, ps - rms_mean, rms_sig, fmt='o', label=r'$P_{data}(k) - P_{noise}(k)$, feed %i' % det)
# plt.plot(k, - rms_sig, '--k', alpha=0.4)
ax1.plot(k, 0 * rms_mean, 'k', alpha=0.4)
ax2 = fig.add_subplot(212)
ax2.errorbar(k, (ps - rms_mean) / rms_sig, rms_sig / rms_sig, fmt='o', label=r'$P_{data}(k) - P_{noise}(k)$, feed %i' % det)
# plt.plot(k, - rms_sig, '--k', alpha=0.4)
ax2.plot(k, 0 * rms_mean, 'k', alpha=0.4)
# plt.plot(k, + rms_sig, '--k', alpha=0.4)
ax1.set_ylabel(r'$P(k)$ [K${}^2$ (Mpc / $h$)${}^3$]')
ax2.set_ylabel(r'$P(k) / \sigma_P$')
ax2.set_xlabel(r'$k$ [(Mpc / $h$)${}^{-1}$]')
ax1.set_xscale('log')
ax2.set_xscale('log')
ax1.set_xlim(1e-2, 1.6e-1)
ax2.set_xlim(1e-2, 1.6e-1)
plt.legend()
plt.savefig('ps_test_%2i.pdf' % det, bbox_inches='tight')
plt.show()


# my_ps = comap2ps.comap2ps('map.h5', decimate_z=32)

# my_ps.calculate_mode_mixing_matrix()
# np.save('M_inv', my_ps.M_inv)
# # print(my_ps.w)
# ps, k = my_ps.calculate_ps()
# rms_mean, rms_sig = my_ps.run_noise_sims(1000)
# print(ps)
# print(rms_mean)
# print(rms_sig)

# plt.plot(k, ps - rms_mean, label=r'$P_{data}(k) - P_{noise}(k)$')
# plt.plot(k, - rms_sig, '--k', alpha=0.4)
# plt.plot(k, 0 * rms_mean, 'k', alpha=0.4)
# plt.plot(k, + rms_sig, '--k', alpha=0.4)
# plt.ylabel(r'$P(k)$ [$\mu$K${}^2$ (Mpc / $h$)${}^3$]')
# plt.xlabel(r'$k$ [(Mpc / $h$)${}^3$]')
# plt.legend()
# plt.savefig('ps_test.pdf')
# plt.show()