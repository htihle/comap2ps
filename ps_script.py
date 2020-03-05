import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
from mpi4py import MPI
import corner
import h5py
import sys

import tools
import master
import comap2ps
import map_obj

try:
    mapname = sys.argv[1]
except IndexError:
    print('Missing filename!')
    print('Usage: python ps_script.py mapname')
    sys.exit(1)

maps = map_obj.MapObj(mapname)
# maps.rms_beam[1:] = 0.0
# maps.rms_beam[0, 3:] = 0.0
#for det in range(1,20):
rms_min = np.min(maps.rms_beam[np.where(maps.rms_beam > 0.0)].flatten())
print(rms_min)

maps.rms_beam[np.where(maps.rms_beam > 75e-6)] = 0.0
#maps.map_beam = np.random.randn(*maps.map_beam.shape) * maps.rms_beam
my_ps = comap2ps.comap2ps(maps, decimate_z=256)

# my_ps.pseudo_ps = False
# my_ps.calculate_mode_mixing_matrix()
# np.save('M_inv', my_ps.M_inv)

ps, k, nmodes = my_ps.calculate_ps()


#rms_mean, rms_sig = my_ps.noise_sims_from_file(mapname)
#print(rms_mean)
#print(ps)
#print(rms_sig)
rms_mean, rms_sig = my_ps.run_noise_sims(10)
# print(rms_mean)
# print(ps)
# print(rms_sig)

np.savetxt('ps_diagonal_cut.txt', ps)
np.savetxt('k_diagnoal_cut.txt', k)
np.savetxt('rms_mean_diagonal_cut.txt',rms_mean)
np.savetxt('rms_sig_diagonal_cut.txt',rms_sig)

ps_th = np.load('ps.npy')
k_th = np.load('k.npy')
# h = 0.7
my_ps.make_h5()
lim = np.mean(np.abs(ps[6:]))  #3


# ps *= K2muK ** 2

kPk = False
# print(my_ps.nmodes)
    
fig = plt.figure()

if kPk:
    ax1 = fig.add_subplot(211)
    ax1.errorbar(k,  k * ps, k * rms_sig, fmt='o', label=r'$\tilde{P}_{data}(k)$')
    ax1.plot(k, k * rms_mean, 'k', label=r'$\tilde{P}_{noise}(k)$', alpha=0.4)
    ax1.plot(k_th, k_th * ps_th, 'r--', label=r'$P_{Theory}(k)$')
    ax1.set_ylabel(r'$k\tilde{P}(k)$ [K${}^2$ (Mpc)${}^2$]')
    ax1.set_ylim(1e-0, 1e4)  # ax1.set_ylim(0, 0.1)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    plt.grid()
    plt.legend()
else: 
    ax1 = fig.add_subplot(211)
    ax1.errorbar(k,  ps, rms_sig, fmt='o', label=r'$\tilde{P}_{data}(k)$')
    ax1.plot(k, rms_mean, 'k', label=r'$\tilde{P}_{noise}(k)$', alpha=0.4)
    # ax1.plot(k_th, ps_th * 10, 'r--', label=r'$10 * P_{Theory}(k)$')
    ax1.set_ylabel(r'$\tilde{P}(k)$ [K${}^2$ (Mpc)${}^3$]')
    ax1.set_ylim(1e5, 1e8)  # ax1.set_ylim(0, 0.1)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    plt.grid()
    plt.legend()

ax2 = fig.add_subplot(212)
ax2.errorbar(k, (ps - rms_mean) / rms_sig, rms_sig / rms_sig, fmt='o', label=r'$\tilde{P}_{data}(k) - \tilde{P}_{noise}(k)$')
ax2.plot(k, 0 * rms_mean, 'k', alpha=0.4)
ax2.set_ylabel(r'$\tilde{P}(k) / \sigma_\tilde{P}$')
ax2.set_xlabel(r'$k$ [Mpc${}^{-1}$]')
#ax2.set_ylim(-7, 20)
ax2.set_ylim(-7, 100)
ax2.set_xscale('log')
plt.legend()


plt.savefig('ps_co6_rmscut.png', bbox_inches='tight')
plt.show()
# print('Done with feed ', det)
