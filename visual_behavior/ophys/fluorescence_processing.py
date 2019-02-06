import numpy as np
import h5py

def get_corrected_fluorescence_traces(roi_metrics, demix_file):
    g = h5py.File(demix_file)
    return np.asarray(g['data'])
