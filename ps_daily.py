import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
from mpi4py import MPI
import h5py
import sys
import multiprocessing

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
save_folder = '/mn/stornext/u3/haavarti/www_docs/diag/ps/'

prefix = mapname[:-6].rpartition("/")[-1]

maps = map_obj.MapObj(mapname, all_feeds=True)

my_ps = comap2ps.comap2ps(maps, decimate_z=256)

ps, k, nmodes = my_ps.calculate_ps()

rms_mean, rms_sig = my_ps.run_noise_sims(100)
    
fig = plt.figure()

    # ax1 = fig.add_subplot(211)
    # ax1.errorbar(k,  ps, rms_sig, fmt='o', label=r'$\tilde{P}_{data}(k)$')
    # ax1.plot(k, rms_mean, 'k', label=r'$\tilde{P}_{noise}(k)$', alpha=0.4)
    # # ax1.plot(k_th, ps_th * 10, 'r--', label=r'$10 * P_{Theory}(k)$')
    # ax1.set_ylabel(r'$\tilde{P}(k)$ [K${}^2$ (Mpc)${}^3$]')
    # # ax1.set_ylim(1e3, 1e8)  # ax1.set_ylim(0, 0.1)
    # ax1.set_yscale('log')
    # ax1.set_xscale('log')
    # plt.grid()
    # plt.legend()

print(ps)
print(rms_mean)

ax1 = fig.add_subplot(211)
ax1.errorbar(k, ps, rms_sig, fmt='o', label=r'$\tilde{P}_{data}(k)$')
ax1.plot(k, rms_mean, 'k', label=r'$\tilde{P}_{noise}(k)$', alpha=0.4)
ax1.set_ylabel(r'$\tilde{P}(k)$ [$\mu$K${}^2$ Mpc${}^3$]')
#ax1.set_ylim(0, 0.012)#rms_mean.mean() * 3)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.grid()
ax1.legend()

ax2 = fig.add_subplot(212)

ax2.errorbar(k, (ps - rms_mean) / rms_sig, rms_sig / rms_sig, fmt='o', label=r'$\tilde{P}_{data}(k) - \tilde{P}_{noise}(k)$')
ax2.plot(k, 0 * rms_mean, 'k', alpha=0.4)
ax2.set_ylabel(r'$\tilde{P}(k) / \sigma_\tilde{P}$')
ax2.set_xlabel(r'$k$ [Mpc${}^{-1}$]')
ax2.set_ylim(-7, 20)
ax2.set_xscale('log')
plt.legend()

plt.savefig(save_folder + prefix + 'ps.pdf', bbox_inches='tight')

cols = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

linetype = 'o'

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
# for det in range(1,3):

def get_det_ps(det):
    my_ps = comap2ps.comap2ps(maps, decimate_z=256, det=det)

    ps, k, nmodes = my_ps.calculate_ps()
    rms_mean, rms_sig = my_ps.run_noise_sims(100)
    return ps, k, rms_mean, rms_sig

pool = multiprocessing.Pool(20)
ps_arr, k_arr, rms_mean_arr, rms_sig_arr = zip(*pool.map(get_det_ps, range(1, 20)))

for det in range(1,20):
    if det == 8:
        linetype = 'x'
    if det == 15: 
        linetype = 'v'
    ps = ps_arr[det - 1]
    k = k_arr[det - 1]
    rms_mean = rms_mean_arr[det - 1]
    rms_sig = rms_sig_arr[det -1]
    # my_ps = comap2ps.comap2ps(maps, decimate_z=256, det=det)

    # ps, k = my_ps.calculate_ps()
    # rms_mean, rms_sig = my_ps.run_noise_sims(100)

    col = cols[(det-1) % 7]
    
    ax1.errorbar(k, ps, rms_sig, color=col, fmt=linetype, label='Feed %02i' % det)  # label=r'$P_{data}(k)$')
    ax1.plot(k, rms_mean, color=col, alpha=0.4)  # label=r'$P_{noise}(k)$', alpha=0.4)
    ax1.set_ylabel(r'$\tilde{P}(k)$ [$\mu$K${}^2$ Mpc${}^3$]')
#    ax1.set_ylim(0, 0.012)#rms_mean.mean() * 3)
    
    ax2.errorbar(k, (ps - rms_mean) / rms_sig, rms_sig / rms_sig, color=col, fmt=linetype, label=r'$\tilde{P}_{data}(k) - \tilde{P}_{noise}(k)$')
    ax2.set_ylabel(r'$\tilde{P}(k) / \sigma_\tilde{P}$')
    ax2.set_xlabel(r'$k$ [Mpc${}^{-1}$]')
    ax2.set_ylim(-7, 20)

    # print('Done with feed ', det)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.grid()
ax2.plot(k, 0 * rms_mean, 'k', alpha=0.4)
ax2.set_xscale('log')
plt.figlegend(*ax1.get_legend_handles_labels(), loc='upper right')
# ax1.legend(loc='upper right')
# ax2.legend()
plt.savefig(save_folder + prefix + 'allpix_ps.pdf', bbox_inches='tight')
#plt.show()
