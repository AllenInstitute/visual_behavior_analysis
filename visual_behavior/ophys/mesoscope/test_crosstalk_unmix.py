import numpy as np
import os
import visual_behavior.ophys.mesoscope.crosstalk_unmix as ica
from scipy import linalg
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

	# 1. Test if all attributes ahve been set:
	for pkey in ica_obj.pkeys:
		for tkey in ica_obj.tkeys:
			assert ica_obj.raws[pkey][tkey] is not None, f"Failed to set attributes self.raws for {pkey}, {tkey}"
			assert ica_obj.raw_paths[pkey][tkey] is not None, f"Failed to set attributes self.raw_paths for {pkey}, {tkey}"
			assert ica_obj.rois_names[pkey][tkey] is not None, f"Failed to set attributes self.rois_names for {pkey}, {tkey}"

	# 2. Test wether Number of traces in ROI and NP aligns, files have been saved to disk
	for pkey in ica_obj.pkeys:
		assert ica_obj.raws[pkey]['roi'].shape == ica_obj.raws[pkey]['np'].shape, f"Number of traces for ROI and Neuropil doens't align for plane {pkey}"
		assert np.all(ica_obj.rois_names[pkey]['roi'] == ica_obj.rois_names[pkey]['np']), f'Roi IDs for ROI and Neuropil for {pkey} are not aligned'
		for tkey in ica_obj.tkeys:
			assert os.path.isfile(ica_obj.raw_paths[pkey][tkey]), f'input traces not found for plane {pkey}, {tkey}'


def test_validate_traces(test_session = None):
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
	ica_obj.validate_traces()
	self = ica_obj

	# 1. Test if all attributes ahve been set:
	for pkey in self.pkeys:
		for tkey in self.tkeys:
			assert self.rois_valid[pkey][tkey] is not None, f"Failed to set attributes self.rois_valid for {pkey}, {tkey}"
			assert self.rois_valid_paths[pkey][tkey] is not None, f"Failed to set attributes self.rois_valid_paths for {pkey}, {tkey}"

	# remove jsons if they exist - for the case whne they have been read fomr disk
	for pkey in self.pkeys:
		for tkey in self.tkeys:
			os.remove(self.rois_valid_paths[pkey][tkey])

	# testing with return_vba = False (default value)

	self.validate_traces()
	for pkey in self.pkeys:  # test if ROi names and NP names align:
		roi_names = [roi for roi, _ in self.rois_valid[pkey]['roi'].items()]
		np_names = [roi for roi, _ in self.rois_valid[pkey]['np'].items()]
		assert roi_names == np_names, f"roi names are not the same for ROI and NP in {pkey}"
		for roi in roi_names:  # test is valid values are the same for ROI and nueropil
			assert self.rois_valid[pkey]['roi'][roi] == self.rois_valid[pkey]['np'][
				roi], f"roi valid flag is not the same for ROI and NP cell {roi} in {pkey}"

	# jsons exist, re-run validate to read them and test the same:
	self.validate_traces()
	for pkey in self.pkeys:  # test if ROi names and NP names align:
		roi_names = [roi for roi, _ in self.rois_valid[pkey]['roi'].items()]
		np_names = [roi for roi, _ in self.rois_valid[pkey]['np'].items()]
		assert roi_names == np_names, f"roi names are not the same for ROI and NP in {pkey}"
		for roi in roi_names:  # test is valid values are the same for ROI and nueropil
			assert self.rois_valid[pkey]['roi'][roi] == self.rois_valid[pkey]['np'][
				roi], f"roi valid flag is not the same for ROI and NP cell {roi} in {pkey}"

	# remove jsons:
	for pkey in self.pkeys:
		for tkey in self.tkeys:
			os.remove(self.rois_valid_paths[pkey][tkey])

	# rerun test with return_vba = True
	self.validate_traces(return_vba=True)
	for pkey in self.pkeys:  # test if ROi names and NP names align:
		roi_names = [roi for roi, _ in self.rois_valid[pkey]['roi'].items()]
		np_names = [roi for roi, _ in self.rois_valid[pkey]['np'].items()]
		assert roi_names == np_names, f"roi names are not the same for ROI and NP in {pkey}"
		for roi in roi_names:  # test is valid values are the same for ROI and nueropil
			assert self.rois_valid[pkey]['roi'][roi] == self.rois_valid[pkey]['np'][
				roi], f"roi valid flag is not the same for ROI and NP cell {roi} in {pkey}"


def test_debias_traces(test_session=None):
	"""
	testing trace debiasing:
	:param test_session:
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
	ica_obj.validate_traces()
	ica_obj.debias_traces()
	self = ica_obj

	# remove outputs if they exist:
	for pkey in self.pkeys:
		for tkey in self.tkeys:
			os.remove(self.ins_paths[pkey][tkey])

	self.debias_traces()

	# 1. test whether all attributes have been set
	for pkey in self.pkeys:
		for tkey in self.tkeys:
			assert self.ins[pkey][tkey] is not None, f"Failed to set attributes self.ins for {pkey}, {tkey}"
			assert self.ins_paths[pkey][tkey] is not None, f"Failed to set attributes self.ins_paths for {pkey}, {tkey}"

	for pkey in self.pkeys:
		for tkey in self.tkeys:
			raw_sig = self.raws[pkey][tkey][0]
			raw_ct = self.raws[pkey][tkey][1]
			rois_valid = self.rois_valid[pkey][tkey]
			# filter raws :
			valid_mask = [valid for _, valid in rois_valid.items()]
			raw_sig = raw_sig[valid_mask, :]
			raw_ct = raw_ct[valid_mask, :]
			inp_sig = self.ins[pkey][tkey][0]
			inp_ct = self.ins[pkey][tkey][1]
			sig_offset = self.offsets[pkey][tkey]['sig_offset']
			ct_offset = self.offsets[pkey][tkey]['ct_offset']
			valid_rois = [roi for roi, valid in rois_valid.items() if valid]
			# 2. test if number of ica input traces in the output is equal to number of valid rois:
			assert inp_sig.shape[0] == len(
				valid_rois), f"Number of traces in debiasing output doesn't agree with valid dict for plane: {pkey}, trace:{tkey}"
			assert inp_ct.shape[0] == len(
				valid_rois), f"Number of traces in debiasing output doesn't agree with valid dict for plane: {pkey}, trace:{tkey}"
			# 3. test if number offsets in the output is equal to number of valid rois:
			assert sig_offset.shape[0] == len(
				valid_rois), f"Number of traces in sig offset output doesn't agree with valid dict for plane: {pkey}, trace:{tkey}"
			assert ct_offset.shape[0] == len(
				valid_rois), f"Number of traces in ct offset output doesn't agree with valid dict for plane: {pkey}, trace:{tkey}"
			# 4. test filtered_raws.shape[0]  = inp.shape[0]
			assert raw_sig.shape[0] == inp_sig.shape[0], f"shape of filtered raw traces SIGNAL does not agree with the shape of ica input traces for plane: {pkey}, trace:{tkey}"
			assert raw_ct.shape[0] == inp_ct.shape[0], f"shape of filtered raw traces CROSSTALK does not agree with the shape of ica input traces for plane: {pkey}, trace:{tkey}"
			# 5. test that debiasing is correct: outputs = input + offset
			for i in range(len(valid_rois)):
				assert np.all(np.isclose(inp_sig[i] + sig_offset[i], raw_sig[
					i])), f'Debiasing went wrong for plane: {pkey}, trace:{tkey}, roi #: {i}, id: {valid_rois[i]}'
				assert np.all(np.isclose(inp_ct[i] + ct_offset[i], raw_ct[
					i])), f'Debiasing went wrong :) for plane: {pkey}, trace:{tkey}, roi #: {i}, id: {valid_rois[i]}'


def test_unmix_pair(test_session=None):
	"""
	testing trace debiasing:
	:param test_session:
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
	ica_obj.validate_traces()
	ica_obj.debias_traces()
	ica_obj.unmix_pair()
	self = ica_obj

	# remove outputs if they were read from disk :
	for pkey in self.pkeys:
		for tkey in self.tkeys:
			os.remove(self.ins_paths[pkey][tkey])

	self.unmix_pair()
	# 1. test that all attributes have been set
	for pkey in self.pkeys:
		for tkey in self.tkeys:
			assert self.outs[pkey][tkey] is not None, f"Failed to set attributes self.outs for {pkey}, {tkey}"
			assert self.outs_paths[pkey][tkey] is not None, f"Failed to set attributes self.outs_paths for {pkey}, {tkey}"
			assert self.crosstalk[pkey][tkey] is not None, f"Failed to set attributes self.crosstalk for {pkey}, {tkey}"
			assert self.mixing[pkey][tkey] is not None, f"Failed to set attributes self.mixing for {pkey}, {tkey}"
			assert self.a_mixing[pkey][tkey] is not None, f"Failed to set attributes self.a_mixing for {pkey}, {tkey}"

	# 2. testing shape ouf outputs
	for pkey in self.pkeys:
		for tkey in self.tkeys:
			out_sig = self.outs[pkey][tkey][0]
			out_ct = self.outs[pkey][tkey][1]
			rois_valid = self.rois_valid[pkey][tkey]
			inp_sig = self.ins[pkey][tkey][0]
			inp_ct = self.ins[pkey][tkey][1]
			mixing = self.mixing[pkey][tkey]
			only_valid_rois = [roi for roi, valid in rois_valid.items() if valid]
			# 2.a Test that number of traces in unmixing output is equal number of valid rois
			assert out_sig.shape[0] == len(
				only_valid_rois), f"Number of traces in unmixing output doesn't agree with number of valid rois for plane: {pkey}, trace:{tkey}"
			assert out_ct.shape[0] == len(
				only_valid_rois), f"Number of traces in unmixing output doesn't agree with number of valid rois for plane: {pkey}, trace:{tkey}"
			assert mixing.shape[0] == len(
				only_valid_rois), f"Number of entries in self.mixing  doesn't agree with number of valid rois for plane: {pkey}, trace:{tkey}"
			# 2.b. test taht number of output traces is equal to number input traces
			assert out_sig.shape[0] == inp_sig.shape[
				0], f"shape of filtered output traces SIGNAL does not agree with the shape of ica input traces for plane: {pkey}, trace:{tkey}"
			assert out_ct.shape[0] == inp_ct.shape[
				0], f"shape of filtered output traces CROSSTALK does not agree with the shape of ica input traces for plane: {pkey}, trace:{tkey}"

	# 3. test that unmixing is correct: unmixing_out - offset = unmixing_in / unmixing_matrix
