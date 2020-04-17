import numpy as np
import os


def test_get_ica_traces(ica_obj):
    for pkey in ica_obj.pkeys:
        assert ica_obj.raws[pkey]['roi'].shape == ica_obj.raws[pkey]['np'].shape, f"Number of traces for ROI and Neuropil doens't align for plane {pkey}"
        assert np.all(ica_obj.rois_names[pkey]['roi'] == ica_obj.rois_names[pkey]['np']), 'Roi IDs for roi and np for {pkey} are not aligned'
        for tkey in ica_obj.tkeys:
            assert os.path.isfile(ica_obj.raw_paths[pkey][tkey]), f'input traces not found for plane {pkey}, {tkey}'

