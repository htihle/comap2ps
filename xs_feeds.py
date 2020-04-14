import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
import corner
import h5py
import sys

import tools
import map_cosmo
import power_spectrum

try:
    mapname = sys.argv[1]
    mapname2 = sys.argv[2]
except IndexError:
    print('Missing filename!')
    print('Usage: python ps_script.py mapname mapname2')
    sys.exit(1)
feeds = [1, 2, 3, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

for i in feeds:
    for j in feeds:
        my_map = map_cosmo.MapCosmo(mapname, feed=i)
        my_map2 = map_cosmo.MapCosmo(mapname2, feed=j)

        my_xs = power_spectrum.CrossSpectrum(my_map, my_map2)

        xs, k, nmodes = my_xs.calculate_xs()

        rms_mean, rms_sig = my_xs.run_noise_sims(10)

        my_xs.make_h5()

        lim = np.mean(np.abs(xs[4:])) * 4

        fig = plt.figure()
        fig.suptitle('feed %02i x feed %02i' % (i, j))
        ax1 = fig.add_subplot(211)
        ax1.errorbar(k, xs, rms_sig, fmt='o', label=r'$\tilde{C}_{data}(k)$')
        ax1.plot(k, 0 * rms_mean, 'k', label=r'$\tilde{C}_{noise}(k)$', alpha=0.4)
        #ax1.plot(k, theory[1:] * 100, 'r--', label=r'$100 * P_{Theory}(k)$')
        #ax1.plot(k_th, ps_th * 10, 'r--', label=r'$10 * P_{Theory}(k)$')
        ax1.set_ylabel(r'$\tilde{C}(k)$ [$\mu$K${}^2$ Mpc${}^3$]')
        ax1.set_ylim(-lim, lim)              # ax1.set_ylim(0, 0.1)
        #ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax1.grid()
        plt.legend()

        ax2 = fig.add_subplot(212)
        ax2.errorbar(k, xs / rms_sig, rms_sig / rms_sig, fmt='o', label=r'$\tilde{C}_{data}(k)$')
        ax2.plot(k, 0 * rms_mean, 'k', alpha=0.4)
        ax2.set_ylabel(r'$\tilde{C}(k) / \sigma_\tilde{C}$')
        ax2.set_xlabel(r'$k$ [Mpc${}^{-1}$]')
        ax2.set_ylim(-12, 12)
        ax2.set_xscale('log')
        ax2.grid()
        plt.legend()

        folder = '/mn/stornext/u3/haavarti/www_docs/files/xs/'
        plt.savefig(folder + 'xs' + my_map.save_string + '_' + my_map2.map_string + '_%02i.png' % j, bbox_inches='tight')
        print('Done with %02i, %02i!' % (i, j))



