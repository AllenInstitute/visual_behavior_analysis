import numpy as np
import os
import visual_behavior.ophys.mesoscope.crosstalk_unmix as ica
CACHE = '/media/rd-storage/Z/MesoscopeAnalysis/'


def test_get_ica_traces():
	session = 786144371
	ica_obj = ica.MesoscopeICA(session_id=session, cache=CACHE, roi_name="ica_traces", np_name="ica_neuropil")
	pairs = ica_obj.dataset.get_paired_planes()
	pair = pairs[0]
	ica_obj.set_exp_ids(pair)
	ica_obj.get_ica_traces()

	# check if alla ttributes have been set
	for pkey in ica_obj.pkeys:
		for tkey in ica_obj.tkeys:
			assert ica_obj.raws[pkey][tkey] is not None, f"Failed to set attributes self.raws for {pkey}, {tkey}"
			assert ica_obj.raw_paths[pkey][tkey] is not None, f"Failed to set attributes self.raw_paths for {pkey}, {tkey}"
			assert ica_obj.rois_names[pkey][tkey] is not None, f"Failed to set attributes self.rois_names for {pkey}, {tkey}"
	for pkey in ica_obj.pkeys:
		assert ica_obj.raws[pkey]['roi'].shape == ica_obj.raws[pkey]['np'].shape, f"Number of traces for ROI and Neuropil doens't align for plane {pkey}"
		assert np.all(ica_obj.rois_names[pkey]['roi'] == ica_obj.rois_names[pkey]['np']), f'Roi IDs for roi and np for {pkey} are not aligned'
		for tkey in ica_obj.tkeys:
			assert os.path.isfile(ica_obj.raw_paths[pkey][tkey]), f'input traces not found for plane {pkey}, {tkey}'
