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

prefix = mapname[:-6]

maps = map_obj.MapObj(mapname, all_feeds=True)

my_ps = comap2ps.comap2ps(maps, decimate_z=256)

ps, k = my_ps.calculate_ps()

rms_mean, rms_sig = my_ps.run_noise_sims(100)
    
fig = plt.figure()

ax1 = fig.add_subplot(211)
ax1.errorbar(k, ps, rms_sig, fmt='o', label=r'$\tilde{P}_{data}(k)$')
ax1.plot(k, rms_mean, 'k', label=r'$\tilde{P}_{noise}(k)$', alpha=0.4)
ax1.set_ylabel(r'$\tilde{P}(k)$ [K${}^2$ (Mpc / $h$)${}^3$]')
ax1.set_ylim(0, 0.012)#rms_mean.mean() * 3)
ax1.legend()

ax2 = fig.add_subplot(212)

ax2.errorbar(k, (ps - rms_mean) / rms_sig, rms_sig / rms_sig, fmt='o', label=r'$\tilde{P}_{data}(k) - \tilde{P}_{noise}(k)$')
ax2.plot(k, 0 * rms_mean, 'k', alpha=0.4)
ax2.set_ylabel(r'$\tilde{P}(k) / \sigma_\tilde{P}$')
ax2.set_xlabel(r'$k$ [(Mpc / $h$)${}^{-1}$]')
ax2.set_ylim(-7, 20)
plt.legend()

plt.savefig(prefix + 'ps.pdf', bbox_inches='tight')

cols = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

linetype = 'o'

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
# for det in range(1,3):
for det in range(1,20):
    if det == 8:
        linetype = 'x'
    if det == 15: 
        linetype = 'v'

    my_ps = comap2ps.comap2ps(maps, decimate_z=256, det=det)

    ps, k = my_ps.calculate_ps()
    col = cols[(det-1) % 7]
    rms_mean, rms_sig = my_ps.run_noise_sims(100)
    
    ax1.errorbar(k, ps, rms_sig, color=col, fmt=linetype, label='Feed %02i' % det)  # label=r'$P_{data}(k)$')
    ax1.plot(k, rms_mean, color=col, alpha=0.4)  # label=r'$P_{noise}(k)$', alpha=0.4)
    ax1.set_ylabel(r'$\tilde{P}(k)$ [K${}^2$ (Mpc / $h$)${}^3$]')
    ax1.set_ylim(0, 0.012)#rms_mean.mean() * 3)

    ax2.errorbar(k, (ps - rms_mean) / rms_sig, rms_sig / rms_sig, color=col, fmt=linetype, label=r'$\tilde{P}_{data}(k) - \tilde{P}_{noise}(k)$')
    ax2.set_ylabel(r'$\tilde{P}(k) / \sigma_\tilde{P}$')
    ax2.set_xlabel(r'$k$ [(Mpc / $h$)${}^{-1}$]')
    ax2.set_ylim(-7, 20)

    # print('Done with feed ', det)
ax2.plot(k, 0 * rms_mean, 'k', alpha=0.4)
plt.figlegend(*ax1.get_legend_handles_labels(), loc='upper right')
# ax1.legend(loc='upper right')
# ax2.legend()
plt.savefig(prefix + 'ps_allpix.pdf', bbox_inches='tight')
plt.show()
