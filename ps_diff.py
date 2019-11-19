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
    mapname2 = sys.argv[2]
except IndexError:
    print('Missing filename!')
    print('Usage: python ps_script.py mapname mapname2')
    sys.exit(1)

maps = map_obj.MapObj(mapname)
maps2 = map_obj.MapObj(mapname2)

tot_rms = np.zeros_like(maps.rms_beam)
where = np.where((maps.rms_beam * maps2.rms_beam) != 0)
#tot_rms[where] = 1 / np.sqrt(1 / maps.rms_beam[where] ** 2 + 1 / maps2.rms_beam[where] ** 2)
tot_rms[where] = np.sqrt(maps.rms_beam[where] ** 2 + maps2.rms_beam[where] ** 2) 
summap = np.zeros_like(maps.map_beam)
summap[where] = (maps.map_beam[where] + maps2.map_beam[where])
diffmap = np.zeros_like(maps.map_beam)
diffmap[where] = (maps.map_beam[where] - maps2.map_beam[where])
#for det in range(1,20):
maps.rms_beam = tot_rms
maps.map_beam = summap
sum_ps = comap2ps.comap2ps(maps, decimate_z=256)
ps_sum, k = sum_ps.calculate_ps()

maps.map_beam = diffmap
diff_ps = comap2ps.comap2ps(maps, decimate_z=256)
ps_diff, k = diff_ps.calculate_ps()


rms_mean, rms_sig = sum_ps.run_noise_sims(100)


lim = np.mean(np.abs(ps_diff[6:])) * 3


fig = plt.figure()

ax1 = fig.add_subplot(211)
ax1.errorbar(k, ps_sum, rms_sig, fmt='o', label=r'$\tilde{P}_{sum}(k)$')
ax1.errorbar(k, ps_diff, rms_sig, fmt='o', label=r'$\tilde{P}_{diff}(k)$')
ax1.plot(k, rms_mean, 'k', label=r'$\tilde{P}_{noise}(k)$', alpha=0.4)
# ax1.plot(k_th / h, ps_th * h ** 3 * 10, 'r--', label=r'$10 * P_{Theory}(k)$')
ax1.set_ylabel(r'$\tilde{P}(k)$ [K${}^2$ (Mpc / $h$)${}^3$]')
ax1.set_ylim(0, lim)  # ax1.set_ylim(0, 0.1)
plt.legend()

ax2 = fig.add_subplot(212)
ax2.errorbar(k, (ps_sum - rms_mean) / rms_sig, rms_sig / rms_sig, fmt='o', label=r'$\tilde{P}_{sum}(k) - \tilde{P}_{noise}(k)$')
ax2.errorbar(k, (ps_diff - rms_mean) / rms_sig, rms_sig / rms_sig, fmt='o', label=r'$\tilde{P}_{diff}(k) - \tilde{P}_{noise}(k)$')
ax2.plot(k, 0 * rms_mean, 'k', alpha=0.4)
ax2.set_ylabel(r'$\tilde{P}(k) / \sigma_\tilde{P}$')
ax2.set_xlabel(r'$k$ [(Mpc / $h$)${}^{-1}$]')
ax2.set_ylim(-7, 20)
plt.legend()


plt.savefig('ps_halfmission_diff.pdf', bbox_inches='tight')
plt.show()
# print('Done with feed ', det)
