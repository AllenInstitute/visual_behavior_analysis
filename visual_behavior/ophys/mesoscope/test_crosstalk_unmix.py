import numpy as np
import os
import visual_behavior.ophys.mesoscope.crosstalk_unmix as ica
CACHE = '/media/rd-storage/Z/MesoscopeAnalysis/'


def test_get_ica_traces(test_session = None):
	"""
	test for visual_behavior.ophys.mesoscope.crosstalk_unmix.MesoscopeICA.get_ica_traces()
	Testing that:
		1. all attributes are set
		2. outputs are written to disk
		3. number of traces for neuropil and roi for each palne agrees
		4. roi names are the same for ROI and Neuropil set of traces of each plane
	:return:
	"""
	if not test_session:
		session = 786144371
	else:
		session = test_session
	ica_obj = ica.MesoscopeICA(session_id=session, cache=CACHE, roi_name="ica_traces", np_name="ica_neuropil")
	pairs = ica_obj.dataset.get_paired_planes()
	pair = pairs[0]
	ica_obj.set_exp_ids(pair)
	ica_obj.get_ica_traces()

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


def test_validate_trtaces(test_session = None):
	"""
	visual_behavior.ophys.mesoscope.crosstalk_unmix.MesoscopeICA.validate_traces()
	Testing:
		1. reading existing valid jsons, confirming the shape, confirming that roi and neuropil valid jsons are the same
		2. validating tarces: confirming
		3. validate against VBA flag = True : confirm roi and np traces valids are teh same, confrim shape
	:return:
	"""
	if not test_session:
		session = 786144371
	else:
		session = test_session

	ica_obj = ica.MesoscopeICA(session_id=session, cache=CACHE, roi_name="ica_traces", np_name="ica_neuropil")
	pairs = ica_obj.dataset.get_paired_planes()
	pair = pairs[0]
	ica_obj.set_exp_ids(pair)
	ica_obj.get_ica_traces()
	self = ica_obj

	#remove jsons if they exist:
	ica_obj.validate_traces()
	for pkey in self.pkeys:
		for tkey in self.tkeys:
			os.remove(self.rois_valid_paths[pkey][tkey])

	# return_vba = False (default value)
	ica_obj.validate_traces()
	for pkey in self.pkeys:
		roi_names = [roi for roi, _ in self.rois_valid[pkey]['roi'].items()]
		np_names = [roi for roi, _ in self.rois_valid[pkey]['np'].items()]
		assert roi_names == np_names, f"roi names are not the same for ROI and NP in {pkey}"
		for roi in roi_names:
			assert self.rois_valid[pkey]['roi'][roi] == self.rois_valid[pkey]['np'][
				roi], f"roi valid flag is not the same for ROI and NP cell {roi} in {pkey}"

	#jsons exist, re-run validate to read them and test the same:
	ica_obj.validate_traces()
	self = ica_obj
	for pkey in self.pkeys:
		roi_names = [roi for roi, _ in self.rois_valid[pkey]['roi'].items()]
		np_names = [roi for roi, _ in self.rois_valid[pkey]['np'].items()]
		assert roi_names == np_names, f"roi names are not the same for ROI and NP in {pkey}"
		for roi in roi_names:
			assert self.rois_valid[pkey]['roi'][roi] == self.rois_valid[pkey]['np'][
				roi], f"roi valid flag is not the same for ROI and NP cell {roi} in {pkey}"

	#remove jsons:
	for pkey in self.pkeys:
		for tkey in self.tkeys:
			os.remove(self.rois_valid_paths[pkey][tkey])

	# rerun test with return_vba = False (default value
	ica_obj.validate_traces(return_vba = True)
	self = ica_obj
	for pkey in self.pkeys:
		roi_names = [roi for roi, _ in self.rois_valid[pkey]['roi'].items()]
		np_names = [roi for roi, _ in self.rois_valid[pkey]['np'].items()]
		assert roi_names == np_names, f"roi names are not the same for ROI and NP in {pkey}"
		for roi in roi_names:
			assert self.rois_valid[pkey]['roi'][roi] == self.rois_valid[pkey]['np'][
				roi], f"roi valid flag is not the same for ROI and NP cell {roi} in {pkey}"


