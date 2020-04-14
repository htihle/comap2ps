import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
import corner
import h5py
import sys

import tools
import map_cosmo
import power_spectrum


n_sim = 10
n_k = 14
n_feed = 19

xs = np.zeros((n_feed, n_feed, n_k))
# rms_xs_mean = np.zeros_like(xs)
rms_xs_std = np.zeros_like(xs)

rms_xs = np.zeros((n_feed, n_feed, n_k, n_sim))

chi2 = np.zeros((n_feed, n_feed, n_k))

for i in range(n_feed):
    for j in range(n_feed):
        try:
            filepath = 'spectra/xs_feeds_co6_half_0_good_hp_map_%02i_map_%02i.h5' % (i+1, j+1)
            with h5py.File(filepath, mode="r") as my_file: 
                xs[i, j] = np.array(my_file['xs'][:])
                rms_xs[i, j] = np.array(my_file['rms_xs'][:])
                rms_xs_std[i, j] = np.array(my_file['rms_xs_std'][:])
                k = np.array(my_file['k'][:])
            print(i+1, j+1)
        except:
            xs[i, j] = np.nan
            rms_xs[i, j] = np.nan
            rms_xs_std[i, j] = np.nan

        chi3 = np.sum((xs[i,j] / rms_xs[i,j]) ** 3)

        chi2[i, j] = np.sign(chi3) * np.sum((xs[i,j] / rms_xs[i,j]) ** 2 - n_k) / np.sqrt(2 * n_k)

plt.imshow(chi2, interpolation='none')


