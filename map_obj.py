import numpy as np
import h5py

class MapObj():
    def __init__(self, mapfile):
        with h5py.File(mapfile, mode="r") as my_file:
            self.x = np.array(my_file['x'][:])
            self.y = np.array(my_file['y'][:])
            self.maps = np.array(my_file['map_beam'][:])
            self.rms = np.array(my_file['rms_beam'][:]) * 2
