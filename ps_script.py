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
#for det in range(1,20):
my_ps = comap2ps.comap2ps(maps, decimate_z=256)

# my_ps.pseudo_ps = False
# my_ps.calculate_mode_mixing_matrix()
# np.save('M_inv', my_ps.M_inv)

ps, k = my_ps.calculate_ps()

#rms_mean, rms_sig = my_ps.noise_sims_from_file(mapname)
#print(rms_mean)
#print(ps)
#print(rms_sig)
rms_mean, rms_sig = my_ps.run_noise_sims(100)
# print(rms_mean)
# print(ps)
# print(rms_sig)
    
fig = plt.figure()

ax1 = fig.add_subplot(211)
ax1.errorbar(k, ps, rms_sig, fmt='o', label=r'$\tilde{P}_{data}(k)$')
ax1.plot(k, rms_mean, 'k', label=r'$\tilde{P}_{noise}(k)$', alpha=0.4)
ax1.set_ylabel(r'$\tilde{P}(k)$ [K${}^2$ (Mpc / $h$)${}^3$]')
ax1.set_ylim(0, 5)  # ax1.set_ylim(0, 0.1)
plt.legend()

ax2 = fig.add_subplot(212)
ax2.errorbar(k, (ps - rms_mean) / rms_sig, rms_sig / rms_sig, fmt='o', label=r'$\tilde{P}_{data}(k) - \tilde{P}_{noise}(k)$')
ax2.plot(k, 0 * rms_mean, 'k', alpha=0.4)
ax2.set_ylabel(r'$\tilde{P}(k) / \sigma_\tilde{P}$')
ax2.set_xlabel(r'$k$ [(Mpc / $h$)${}^{-1}$]')
ax2.set_ylim(-7, 10)
plt.legend()


plt.savefig('ps_co7.pdf', bbox_inches='tight')
plt.show()
# print('Done with feed ', det)
