import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
import h5py
from mpi4py import MPI
import glob


import tools
import master


class comap2ps():
    def __init__(self, maps, decimate_z=32, use_mpi=False, det=None, perp=False):
        self.pseudo_ps = True
        self.perp = perp
        h = 0.7
        deg2mpc = 76.22 / h  # at redshift 2.9
        dz2mpc = 699.62 / h # redshift 2.4 to 3.4
        K2muK = 1e6
        dz = (1+2.9) ** 2 * 32.2e-3 / 115  # conversion 
        n_f = 256  # 64 * 4
        redshift = np.linspace(2.9 - n_f/2*dz, 2.9 + n_f/2*dz, n_f + 1)
        
        if det is not None:
            self.det = det
            self.datamap = maps.maps[self.det-1].transpose(3, 2, 0, 1) * K2muK
            self.rms = maps.rms[self.det-1].transpose(3, 2, 0, 1) * K2muK
        else:
            self.datamap = maps.map_beam.transpose(3, 2, 0, 1) * K2muK #maps.maps[self.det-1].transpose(3, 2, 0, 1)   # now the order is (x, y, sb, freq)
            self.rms = maps.rms_beam.transpose(3, 2, 0, 1) * K2muK #maps.rms[self.det-1].transpose(3, 2, 0, 1)
            
        meandec = np.mean(maps.y)
        self.x = maps.x * deg2mpc * np.cos(meandec * np.pi / 180) #tools.cent2edge(np.array(my_file['x'][:])) * deg2mpc  # ra
        self.y = maps.y * deg2mpc #tools.cent2edge(np.array(my_file['y'][:])) * deg2mpc  # dec
        # print(maps.x)
        # print(maps.y)
        self.z = tools.edge2cent(redshift * dz2mpc)
        # self.datamap = np.zeros_like(self.datamap) + np.arange(64)[None, None, None, :]
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
        
        kmax = np.sqrt(
            np.max(np.abs(fft.fftfreq(len(self.x), self.dx)) * 2 * np.pi) ** 2
            + np.max(np.abs(fft.fftfreq(len(self.y), self.dy)) * 2 * np.pi) ** 2
            + np.max(np.abs(fft.fftfreq(len(self.z), self.dz)) * 2 * np.pi) ** 2
        )
#        print(np.max(np.abs(fft.fftfreq(len(self.z), self.dz)) * 2 * np.pi) ** 2)

        n_k = 15

        
        # self.k_bin_edges = np.linspace(0 - 1e-5, kmax + 1e-5, n_k)  # np.logspace(-3, np.log10(kmax + 1e-5), n_k) #np.linspace(0 - 1e-5, kmax + 1e-5, n_k)
        self.k_bin_edges = np.logspace(-2.0, np.log10(1.5), n_k) #np.linspace(0 - 1e-5, kmax + 1e-5, n_k)
        #self.k = tools.edge2cent(self.k_bin_edges)


        #### this lead to nan in some cases test this thoroughly:
        # mean = np.nansum(self.datamap * self.w, axis=(0,1)) / np.nansum(self.w, axis=(0,1))
        # self.datamap = self.datamap - mean[None, None, :]



        #vmax = 200
        #plt.imshow(self.datamap[:,:,13], interpolation='none', vmin=-vmax, vmax=vmax)
        #plt.show()
        #sys.exit()


    def decimate_in_frequency(self, n_end): ##### understand this!!
        sh = self.datamap.shape
        self.datamap = self.datamap.reshape((sh[0], sh[1], sh[2] * sh[3])) 
        self.rms = self.rms.reshape((sh[0], sh[1], sh[2] * sh[3]))
        self.mask = np.zeros_like(self.rms)
        self.mask[(self.rms != 0.0)] = 1.0
        where = (self.mask == 1.0)
        self.w = np.zeros_like(self.rms)
        self.w[where] = 1 / np.sqrt(np.mean(1 / self.rms[where].flatten() ** 4)) / self.rms[where] ** 2 
        #np.mean(self.rms[where].flatten() ** 2) / self.rms[where] ** 2

        n_dec = self.nz // n_end
        self.datamap = np.sum(
            (self.datamap * self.w).reshape((self.nx, self.ny, self.nz // n_dec, n_dec)), axis=3
        ) / np.sum(
            self.w.reshape((self.nx, self.ny, self.nz // n_dec, n_dec)), axis=3
        )
        # print('after: ', self.datamap.shape)
        self.w = np.sum(
            self.w.reshape((self.nx, self.ny, self.nz // n_dec, n_dec)), axis=3
        )

        self.rms = 1 / np.sqrt(np.sum(
            (1 / self.rms ** 2).reshape((self.nx, self.ny, self.nz // n_dec, n_dec)), axis=3
        ))

        self.datamap[(self.w == 0)] = 0.0
        # print(self.datamap[40, 50, :])
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
            print('Empty map', len(used_x), len(used_y), len(used_z))#, 'Feed: ', self.det)
            pass
        self.used_x = used_x
        self.used_y = used_y
        self.used_z = used_z
        
            
        

    def calculate_mode_mixing_matrix(self, det=None):
        self.M_inv, self.k = master.mode_coupling_matrix_3d(
            self.w, k_bin_edges=self.k_bin_edges, dx=self.dx,
            dy=self.dy, dz=self.dz, insert_edge_bins=False
        )
    
    def calculate_ps(self, det=None):
        if self.perp:
            ps_with_weight, self.k, self.nmodes = tools.compute_power_spec_perp_vs_par(
                self.datamap * self.w, self.k_bin_edges, dx=self.dx, dy=self.dy, dz=self.dz)
        else:
            ps_with_weight, self.k, self.nmodes = tools.compute_power_spec3d(
                self.datamap * self.w, self.k_bin_edges, dx=self.dx, dy=self.dy, dz=self.dz)
        if self.pseudo_ps: 
            self.data_ps = ps_with_weight
        else:
            self.data_ps = np.dot(self.M_inv, ps_with_weight)
        return self.data_ps, self.k, self.nmodes

    def run_noise_sims(self, n_sims):
        rms_ps = np.zeros((len(self.k_bin_edges) - 1, n_sims))
        for i in range(n_sims):
            randmap = self.rms * np.random.randn(*self.rms.shape)
#            randmap = randmap - randmap.flatten().mean()
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

    def noise_sims_from_file(self, mapfile):
        prefix, _, mapname = mapfile.rpartition("/")
         #+ '*.h5')
        sim_list = glob.glob(prefix + '/sims/' + mapname[:-3].replace('map', 'sim') + '*.h5')
        n_sims = len(sim_list)
        print(n_sims)
        rms_ps = np.zeros((len(self.k_bin_edges) - 1, n_sims))
        for i in range(n_sims):
            print(sim_list[i])
            with h5py.File(sim_list[i], mode="r") as my_file:
                randmap = np.array(my_file['map_sim']).transpose(3, 2, 0, 1) #.transpose(0, 4, 3, 1, 2)  
                sh = randmap.shape
                randmap = randmap.reshape((sh[0], sh[1], sh[2] * sh[3]))
                randmap = randmap[self.used_x[0]:self.used_x[-1],
                                  self.used_y[0]:self.used_y[-1],
                                  self.used_z[0]:self.used_z[-1]]
            if self.pseudo_ps:
                rms_ps[:, i] = tools.compute_power_spec3d(
                    randmap * self.w, self.k_bin_edges, dx=self.dx, 
                    dy=self.dy, dz=self.dz)[0]
            else:
                rms_ps[:, i] = np.dot(self.M_inv, tools.compute_power_spec3d(
                    randmap * self.w,self.k_bin_edges, dx=self.dx, 
                    dy=self.dy, dz=self.dz)[0])
        self.rms_ps_mean = np.mean(rms_ps, axis=1)
        self.rms_ps_std = np.std(rms_ps, axis=1, ddof=1)
        return self.rms_ps_mean, self.rms_ps_std


    def make_h5(self, outname='ps.h5'):
        f1 = h5py.File(outname, 'w')
        try: 
            f1.create_dataset('ps', data=self.data_ps)
            f1.create_dataset('k', data=self.k)
            f1.create_dataset('k_bin_edges', data=self.k_bin_edges)
            f1.create_dataset('nmodes', data=self.nmodes)
        except:
            print('No power spectrum calculated.')
            return 
        try:
            f1.create_dataset('rms_ps_mean', data=self.rms_ps_mean)
            f1.create_dataset('rms_ps_std', data=self.rms_ps_std)
        except:
            pass
        f1.close()
    # def noise_sims_from_file(self, mapfile):
    #     with h5py.File(mapfile, mode="r") as my_file:
    #         randmap = np.array(my_file['map_sim'][:, self.det-1]).transpose(0, 4, 3, 1, 2)  # order is now (nsim, x, y, sb, freq)
    #     sh = randmap.shape
    #     n_sims = sh[0]
    #     randmap = randmap.reshape((sh[0], sh[1], sh[2], sh[3] * sh[4]))
    #     try: 
    #         randmap = randmap[:, self.used_x[0]:self.used_x[-1],
    #                           self.used_y[0]:self.used_y[-1],
    #                           self.used_z[0]:self.used_z[-1]]
    #     except IndexError:
    #         ##print('Empty map', len(self.used_x), len(self.used_y), len(self.used_z), 'Feed: ', self.det)
    #         randmap = np.zeros((n_sims, self.nx, self.ny, self.nz))
        
    #     rms_ps = np.zeros((len(self.k_bin_edges) - 1, n_sims))
    #     for i in range(n_sims):
    #         if self.pseudo_ps:
    #             rms_ps[:, i] = tools.compute_power_spec3d(
    #                 randmap[i] * self.w, self.k_bin_edges, dx=self.dx, 
    #                 dy=self.dy, dz=self.dz)[0]
    #         else:
    #             rms_ps[:, i] = np.dot(self.M_inv, tools.compute_power_spec3d(
    #                 randmap[i] * self.w,self.k_bin_edges, dx=self.dx, 
    #                 dy=self.dy, dz=self.dz)[0])
    #     self.rms_ps_mean = np.mean(rms_ps, axis=1)
    #     self.rms_ps_std = np.std(rms_ps, axis=1, ddof=1)
    #     return self.rms_ps_mean, self.rms_ps_std
