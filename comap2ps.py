import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
import h5py
from mpi4py import MPI


import tools
import master


class comap2ps():
    def __init__(self, mapfile, decimate_z=32, use_mpi=False, det=None):
        self.pseudo_ps = True

        self.mapfile = mapfile
        
        deg2mpc = 76.22  # at redshift 2.9
        dz2mpc = 699.62  # redshift 2.4 to 3.4
        #freq = np.linspace(26, 34, 257)
        dz = (1+2.9) ** 2 * 32.2e-3 / 115
        n_f = 256#64
        redshift = np.linspace(2.9 - n_f/2*dz, 2.9 + n_f/2*dz, n_f + 1)
        if det is not None:
            self.det = det
        with h5py.File(mapfile, mode="r") as my_file:
            self.x = np.array(my_file['x'][:]) * deg2mpc#tools.cent2edge(np.array(my_file['x'][:])) * deg2mpc
            self.y = np.array(my_file['y'][:]) * deg2mpc#tools.cent2edge(np.array(my_file['y'][:])) * deg2mpc

        self.z = tools.edge2cent(redshift * dz2mpc)

        self.nz = len(self.z)
        self.nx = len(self.x)
        self.ny = len(self.y)

        self.decimate_in_frequency(decimate_z)

        self.remove_whitespace()

        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.dz = self.z[1] - self.z[0]
        self.nz = len(self.z)
        self.nx = len(self.x)
        self.ny = len(self.y)
        # print(self.nx, self.ny, self.nz)
        kmax = np.sqrt(
            np.max(np.abs(fft.fftfreq(len(self.x), self.dx))) ** 2
            + np.max(np.abs(fft.fftfreq(len(self.y), self.dy))) ** 2
            + np.max(np.abs(fft.fftfreq(len(self.z), self.dz))) ** 2
        )

        n_k = 15
        # print(kmax)
        # kmax = 0.23#0.155
        self.k_bin_edges = np.linspace(0 - 1e-5, kmax + 1e-5, n_k)  # np.logspace(-3, np.log10(kmax + 1e-5), n_k) #np.linspace(0 - 1e-5, kmax + 1e-5, n_k)
        self.k = tools.edge2cent(self.k_bin_edges)


    def decimate_in_frequency(self, n_end):
        with h5py.File(self.mapfile, mode="r") as my_file:
            # self.datamap = np.array(my_file['map_beam'][:]).transpose() / 0.5616
            # self.rms = np.array(my_file['rms_beam'][:]).transpose()
            self.datamap = np.array(my_file['map'][self.det-1]).transpose()
            self.rms = np.array(my_file['rms'][self.det-1]).transpose()
            # self.mask = np.array(my_file['mask'][:])
        sh = self.datamap.shape
        # print(sh)
        self.datamap = self.datamap.reshape((sh[0] * sh[1], sh[2], sh[3]))
        self.rms = self.rms.reshape((sh[0] * sh[1], sh[2], sh[3]))
        self.mask = np.zeros_like(self.rms)
        self.mask[(self.datamap != 0.0)] = 1.0
        where = (self.mask == 1.0)
        self.w = np.zeros_like(self.rms)
        self.w[where] = np.mean(self.rms[where].flatten() ** 2) / self.rms[where] ** 2

        n_dec = self.nz // n_end

        self.datamap = np.sum(
            (self.datamap * self.w).reshape((self.nx, self.ny, self.nz // n_dec, n_dec)), axis=3
        ) / np.sum(
            self.w.reshape((self.nx, self.ny, self.nz // n_dec, n_dec)), axis=3
        )
        
        self.w = np.sum(
            self.w.reshape((self.nx, self.ny, self.nz // n_dec, n_dec)), axis=3
        )

        self.rms = 1 / np.sqrt(np.sum(
            (1 / self.rms ** 2).reshape((self.nx, self.ny, self.nz // n_dec, n_dec)), axis=3
        ))

        self.datamap[(self.w == 0)] = 0.0
        self.nz = self.nz // n_dec
        self.mask = np.zeros_like(self.w)
        self.mask[(self.w > 0)] = 1.0
        self.z = np.mean(self.z.reshape((self.nz, n_dec)), axis=1)

    def remove_whitespace(self):
        used_x = np.where(self.mask.sum(axis=(1, 2)))[0]
        used_y = np.where(self.mask.sum(axis=(0, 2)))[0]
        used_z = np.where(self.mask.sum(axis=(0, 1)))[0]
        try: 
            self.datamap = self.datamap[used_x[0]:used_x[-1], used_y[0]:used_y[-1], used_z[0]:used_z[-1]]
            self.w = self.w[used_x[0]:used_x[-1], used_y[0]:used_y[-1], used_z[0]:used_z[-1]]
            self.rms = self.rms[used_x[0]:used_x[-1], used_y[0]:used_y[-1], used_z[0]:used_z[-1]]
            self.mask = self.mask[used_x[0]:used_x[-1], used_y[0]:used_y[-1], used_z[0]:used_z[-1]]
            self.x = self.x[used_x[0]:used_x[-1]]
            self.y = self.y[used_y[0]:used_y[-1]]
            self.z = self.z[used_z[0]:used_z[-1]]
        except IndexError:
            print('Empty map', len(used_x), len(used_y), len(used_z))
            pass


    def calculate_mode_mixing_matrix(self, det=None):
        self.M_inv, k = master.mode_coupling_matrix_3d(
            self.w, k_bin_edges=self.k_bin_edges, dx=self.dx,
            dy=self.dy, dz=self.dz, insert_edge_bins=False
        )
    
    def calculate_ps(self, det=None):
        ps_with_weight, k, _ = tools.compute_power_spec3d(
            self.datamap * self.w, self.k_bin_edges, dx=self.dx, dy=self.dy, dz=self.dz)
        if self.pseudo_ps: 
            self.data_ps = ps_with_weight
        else:
            self.data_ps = np.dot(self.M_inv, ps_with_weight)
        return self.data_ps, k

    def run_noise_sims(self, n_sims):
        rms_ps = np.zeros((len(self.k_bin_edges) - 1, n_sims))
        for i in range(n_sims):
            randmap = self.rms * np.random.randn(*self.rms.shape)
            randmap = randmap - randmap.flatten().mean()
            if self.pseudo_ps: 
                rms_ps[:, i] = tools.compute_power_spec3d(
                randmap * self.w, self.k_bin_edges, dx=self.dx, 
                dy=self.dy, dz=self.dz)[0]
            else:
                rms_ps[:, i] = np.dot(self.M_inv, tools.compute_power_spec3d(
                    randmap * self.w,self.k_bin_edges, dx=self.dx, 
                    dy=self.dy, dz=self.dz)[0])
        self.rms_ps_mean = np.mean(rms_ps, axis=1)
        self.rms_ps_std = np.std(rms_ps, axis=1)
        return self.rms_ps_mean, self.rms_ps_std
