import numpy as np
import h5py

class MapObj():
    def __init__(self, mapfile, all_feeds=False):
        with h5py.File(mapfile, mode="r") as my_file:
            self.x = np.array(my_file['x'][:])
            self.y = np.array(my_file['y'][:])
            self.map_beam = np.array(my_file['map_beam'][:])
            self.rms_beam = np.array(my_file['rms_beam'][:]) * np.sqrt(8) #* 2.4 #2.7
            if all_feeds:
                self.maps = np.array(my_file['map'][:])
                self.rms = np.array(my_file['rms'][:]) #* 2
