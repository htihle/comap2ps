import numpy as np
import h5py
import tools
import map_cosmo


class PowerSpectrum():
    def __init__(self, my_map):
        self.map = my_map
        self.weights_are_normalized = False

    def normalize_weights(self):
        self.map.w = self.map.w / np.sqrt(np.mean(self.map.w.flatten() ** 2))
        self.weights_are_normalized = True
    
    def calculate_ps(self, do_2d=False):
        n_k = 15

        if not self.weights_are_normalized: self.normalize_weights()
        if do_2d:
            self.k_bin_edges_par = np.logspace(-2.0, np.log10(1.0), n_k)
            self.k_bin_edges_perp = np.logspace(-2.0 + np.log10(2), np.log10(1.5), n_k)
            
            self.ps_2d, self.k, self.nmodes = tools.compute_power_spec_perp_vs_par(
                self.map.map * self.map.w, (self.k_bin_edges_perp, self.k_bin_edges_par),
                dx=self.map.dx, dy=self.map.dy, dz=self.map.dz
            )
            return self.ps_2d, self.k, self.nmodes
        else:
            self.k_bin_edges = np.logspace(-2.0, np.log10(1.5), n_k)
            self.ps, self.k, self.nmodes = tools.compute_power_spec3d(
                self.map.map * self.map.w, self.k_bin_edges,
                dx=self.map.dx, dy=self.map.dy, dz=self.map.dz
            )
            return self.ps, self.k, self.nmodes
    
    def run_noise_sims(self, n_sims):
        if not self.weights_are_normalized: self.normalize_weights()
        
        rms_ps = np.zeros((len(self.k_bin_edges) - 1, n_sims))
        for i in range(n_sims):
            randmap = self.map.rms * np.random.randn(*self.map.rms.shape)

            rms_ps[:, i] = tools.compute_power_spec3d(
                randmap * self.map.w, self.k_bin_edges,
                dx=self.map.dx, dy=self.map.dy, dz=self.map.dz
                )[0]
        self.rms_ps_mean = np.mean(rms_ps, axis=1)
        self.rms_ps_std = np.std(rms_ps, axis=1)
        return self.rms_ps_mean, self.rms_ps_std
    
    def make_h5(self, outname=None):
        if outname is None:
            tools.ensure_dir_exists('spectra')
            outname = 'spectra/ps' + self.map.save_string + '.h5'            

        f1 = h5py.File(outname, 'w')
        try:
            f1.create_dataset('mappath', data=self.map.mappath)
            f1.create_dataset('ps', data=self.ps)
            f1.create_dataset('k', data=self.k)
            f1.create_dataset('k_bin_edges', data=self.k_bin_edges)
            f1.create_dataset('nmodes', data=self.nmodes)
        except:
            print('No power spectrum calculated.')
            return 
        try:
            f1.create_dataset('ps_2d', data=self.ps_2d)
        except:
            pass
        
        try:
            f1.create_dataset('rms_ps_mean', data=self.rms_ps_mean)
            f1.create_dataset('rms_ps_std', data=self.rms_ps_std)
        except:
            pass
        f1.close()


class CrossSpectrum():
    def __init__(self, my_map, my_map2):
        self.maps = []
        self.maps.append(my_map)
        self.maps.append(my_map2)
        self.weights_are_normalized = False

    def normalize_weights(self):
        norm = np.sqrt(np.mean((self.maps[0].w * self.maps[1].w).flatten()))
        self.maps[0].w = self.maps[0].w / norm
        self.maps[1].w = self.maps[1].w / norm
        self.weights_are_normalized = True
    
    def calculate_xs(self):
        n_k = 15

        self.k_bin_edges = np.logspace(-2.0, np.log10(1.5), n_k)
        
        if not self.weights_are_normalized: self.normalize_weights()

        self.xs, self.k, self.nmodes = tools.compute_cross_spec3d(
            (self.maps[0].map * self.maps[0].w, self.maps[1].map * self.maps[1].w),
            self.k_bin_edges, dx=self.maps[0].dx, dy=self.maps[0].dy, dz=self.maps[0].dz)
        return self.xs, self.k, self.nmodes

    def run_noise_sims(self, n_sims, seed=None):
        if not self.weights_are_normalized: self.normalize_weights()

        if seed is not None:
            if self.maps[0].feed is not None:
                feeds = np.array([self.maps[0].feed, self.maps[1].feed])
            else:
                feeds = np.array([1, 1])
        
        self.rms_xs = np.zeros((len(self.k_bin_edges) - 1, n_sims))
        for i in range(n_sims):
            randmap = [np.zeros(self.maps[0].rms.shape), np.zeros(self.maps[0].rms.shape)]
            for j in range(len(self.maps)):
                if seed is not None:
                    np.random.seed(seed * i * j * feeds[j])
                randmap[j] = np.random.randn(*self.maps[j].rms.shape) * self.maps[j].rms

            self.rms_xs[:, i] = tools.compute_cross_spec3d(
                (randmap[0] * self.maps[0].w, randmap[1] * self.maps[1].w),
                self.k_bin_edges, dx=self.maps[0].dx, dy=self.maps[0].dy, dz=self.maps[0].dz)[0]
                
        self.rms_xs_mean = np.mean(self.rms_xs, axis=1)
        self.rms_xs_std = np.std(self.rms_xs, axis=1)
        return self.rms_xs_mean, self.rms_xs_std
    
    def make_h5(self, outname=None, save_noise_realizations=False):
        if outname is None:
            tools.ensure_dir_exists('spectra')
            outname = 'spectra/xs' + self.maps[0].save_string + '_' + self.maps[1].map_string + '.h5'            

        f1 = h5py.File(outname, 'w')
        try:
            f1.create_dataset('mappath1', data=self.maps[0].mappath)
            f1.create_dataset('mappath2', data=self.maps[1].mappath)
            f1.create_dataset('xs', data=self.xs)
            f1.create_dataset('k', data=self.k)
            f1.create_dataset('k_bin_edges', data=self.k_bin_edges)
            f1.create_dataset('nmodes', data=self.nmodes)
        except:
            print('No power spectrum calculated.')
            return 
        try:
            f1.create_dataset('rms_xs_mean', data=self.rms_xs_mean)
            f1.create_dataset('rms_xs_std', data=self.rms_xs_std)
        except:
            pass
        if save_noise_realizations:
            try:
                f1.create_dataset('rms_xs', data=self.rms_xs)
            except:
                pass
        f1.close()
