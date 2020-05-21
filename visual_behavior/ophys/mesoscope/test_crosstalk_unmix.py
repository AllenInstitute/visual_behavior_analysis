import numpy as np
import os
import visual_behavior.ophys.mesoscope.crosstalk_unmix as ica
import sciris as sc
import pylab as pl
import h5py
from visual_behavior.ophys.mesoscope.crosstalk_unmix import run_ica
import copy
CACHE = '/media/rd-storage/Z/MesoscopeAnalysis/'


def test_get_ica_traces(test_session=None):
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

	# 2. Test if number of rois is the same as nuber of traces
	for pkey in ica_obj.pkeys:
		for tkey in ica_obj.tkeys:
			assert len(ica_obj.rois_names[pkey][tkey]) == ica_obj.raws[pkey][tkey].shape[1], f"Failed : Number of rois " \
				f"doesn't align wiht number of traces for exp: {ica_obj.exp_ids[pkey]}"

	# 3. Test wether # of traces in ROI and NP aligns, files have been saved to disk
	for pkey in ica_obj.pkeys:
		assert ica_obj.raws[pkey]['roi'].shape == ica_obj.raws[pkey]['np'].shape, f"Number of traces for ROI and Neuropil doens't align for plane {pkey}"
		assert np.all(ica_obj.rois_names[pkey]['roi'] == ica_obj.rois_names[pkey]['np']), f'Roi IDs for ROI and Neuropil for {pkey} are not aligned'
		for tkey in ica_obj.tkeys:
			assert os.path.isfile(ica_obj.raw_paths[pkey][tkey]), f'input traces not found for plane {pkey}, {tkey}'


def test_validate_raw_traces(test_session = None):
	"""
	visual_behavior.ophys.mesoscope.crosstalk_unmix.MesoscopeICA.validate_raw_traces()
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
	ica_obj.validate_raw_traces()
	self = ica_obj

	# 1. Test if all attributes have been set:
	for pkey in self.pkeys:
		for tkey in self.tkeys:
			assert self.rois_valid[pkey] is not None, f"Failed to set attributes self.rois_valid for {pkey}, {tkey}"
			assert self.rois_valid_paths[pkey] is not None, f"Failed to set attributes self.rois_valid_paths for {pkey}, {tkey}"

	# remove jsons if they exist - for the case when they have been read from disk
	for pkey in self.pkeys:
		for tkey in self.tkeys:
			os.remove(self.rois_valid_paths[pkey][tkey])

	# testing with return_vba = False (default value)
	self.validate_raw_traces()
	for pkey in self.pkeys:  # test if ROi names and NP names align:
		roi_names = self.rois_names_valid[pkey]['roi']
		np_names = self.rois_names_valid[pkey]['np']
		assert roi_names == np_names, f"roi names are not the same for ROI and NP in {pkey}"

	# jsons exist, re-run validate to read them and test the same:
	self.validate_raw_traces()
	for pkey in self.pkeys:  # test if ROi names and NP names align:
		roi_names = self.rois_names_valid[pkey]['roi']
		np_names = self.rois_names_valid[pkey]['np']
		assert roi_names == np_names, f"roi names are not the same for ROI and NP in {pkey}"

	# 3. test if roi ids in valid json came from corresponding raw output
	for pkey in self.pkeys:
		for tkey in self.tkeys:
			roi_names_valid = [str(roi) for roi in self.rois_names_valid[pkey][tkey]]
			raw_roi_names = self.rois_names[pkey][tkey]
			assert all([int(roi_v) in raw_roi_names for roi_v in roi_names_valid]), f"Some roi names in self.self.rois_names_valid are not in self.rois_names for {pkey}, {tkey}"


def test_debias_traces(test_session=None):
	"""
	testing trace debiasing:
	:param test_session: LIMS session ID to use in test
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
	ica_obj.validate_raw_traces()
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
			rois_valid = self.rois_valid[pkey]
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

			# 6. test that debiasing is correct: outputs = input + offset
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
	ica_obj.validate_raw_traces()
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
			assert len(mixing) == len(
				only_valid_rois), f"Number of entries in self.mixing  doesn't agree with number of valid rois for plane: {pkey}, trace:{tkey}"
			# 2.b. test taht number of output traces is equal to number input traces
			assert out_sig.shape[0] == inp_sig.shape[
				0], f"shape of filtered output traces SIGNAL does not agree with the shape of ica input traces for plane: {pkey}, trace:{tkey}"
			assert out_ct.shape[0] == inp_ct.shape[
				0], f"shape of filtered output traces CROSSTALK does not agree with the shape of ica input traces for plane: {pkey}, trace:{tkey}"

	# 3. test that unmixing is correct: unmixing_out - offset = unmixing_in / unmixing_matrix


def test_validate_cells_crosstalk(session):
	"""
	test function for crosstalkunmix.validate_cells_crosstalk
	:param session: LIMS session id for sessiong to use for the test
	:return: none
	"""

	if not session:
		ses = 839208243
		"""LIMS session ID to use for test,
		use a test session with ica and lims processing performed on it to avoid runnign ICA from scratch
		which takes ~ 5 hours"""
	else:
		ses = session

	ica_obj = ica.MesoscopeICA(session_id=ses, cache=CACHE, roi_name="ica_traces", np_name="ica_neuropil")
	pairs = ica_obj.dataset.get_paired_planes()
	pair = pairs[0]
	ica_obj.set_exp_ids(pair)
	ica_obj.get_ica_traces()
	ica_obj.validate_raw_traces()
	ica_obj.debias_traces()
	ica_obj.unmix_pair()
	ica_obj.validate_cells_crosstalk()
	self = ica_obj

	# 1. Test if all attributes have been set:
	for pkey in self.pkeys:
		for tkey in self.tkeys:
			assert self.rois_valid_ct[pkey] is not None, f"Failed to set attributes self.rois_valid for {pkey}, {tkey}"
			assert self.rois_valid_ct_paths[pkey] is not None, f"Failed to set attributes self.rois_valid_paths for {pkey}, {tkey}"

	# 2. remove jsons if they exist - for the case when they have been read from disk
	for pkey in self.pkeys:
		for tkey in self.tkeys:
			os.remove(self.rois_valid_ct_paths[pkey][tkey])

	self.validate_cells_crosstalk()

	# 3. test that roi and np ids are identical:
	for pkey in self.pkeys:  # test if ROi names and NP names align:
		roi_names = self.rois_names_valid_ct[pkey]['roi']
		np_names = self.rois_names_valid_ct[pkey]['np']
		assert roi_names == np_names, f"roi names are not the same for ROI and NP in {pkey}"

	# 4. now that jsons exist, test teh same whne they are read from disk:
	self.validate_cells_crosstalk()

	for pkey in self.pkeys:  # test if ROi names and NP names align:
		roi_names = self.rois_names_valid_ct[pkey]['roi']
		np_names = self.rois_names_valid_ct[pkey]['np']
		assert roi_names == np_names, f"roi names are not the same for ROI and NP in {pkey}"

	# 5. test if roi ids in valid json came from corresponding raw output
	for pkey in self.pkeys:
		for tkey in self.tkeys:
			roi_names_valid_ct = [str(roi) for roi in self.rois_names_valid_ct[pkey][tkey]]
			raw_roi_names = self.rois_names[pkey][tkey]
			assert all([roi_v in raw_roi_names for roi_v in roi_names_valid_ct])
	return


def test_filter_dff_traces_crosstalk(session=None):
    if not session:
        ses = 839208243
        """LIMS session ID to use for test,
	    use a test session with ica and lims processing performed on it to avoid runnign ICA from scratch
	    which takes ~ 5 hours"""
    else:
        ses = session

    ica_obj = ica.MesoscopeICA(session_id=ses, cache=CACHE, roi_name="ica_traces", np_name="ica_neuropil")
    pairs = ica_obj.dataset.get_paired_planes()
    pair = pairs[0]
    ica_obj.set_exp_ids(pair)
    ica_obj.get_ica_traces()
    ica_obj.validate_raw_traces(return_vba=False)
    ica_obj.debias_traces()
    ica_obj.unmix_pair()
    ica_obj.validate_cells_crosstalk()

    # 0. if dff_ct files exist - delete them first, this test will regenerate the files
    for pkey in ica_obj.pkeys:
	    dff_file = os.path.join(ica_obj.session_dir, f"{ica_obj.exp_ids[pkey]}_dff_ct.h5")
	    if os.path.isfile(dff_file):
		    os.chdir(ica_obj.session_dir)
		    print(f"Filtered dff files exist for {ica_obj.exp_ids[pkey]}, deleting for the test purposes")
		    sc.runcommand(f"rm -rf {dff_file}")

    ica_obj.filter_dff_traces_crosstalk()

    # 1. test that all attributes have been set
    for pkey in ica_obj.pkeys:
	    assert len(ica_obj.dff_files[pkey]) != 0, f"Failed to set attributes self.dff_files for {ica_obj.exp_ids[pkey]}"
	    assert len(ica_obj.dff[pkey]) != 0, f"Failed to set attributes self.dffs for {ica_obj.exp_ids[pkey]}"
	    assert len(ica_obj.dff_ct_files[pkey]) != 0, f"Failed to set attributes self.dff_ct_files {ica_obj.exp_ids[pkey]}"
	    assert len(ica_obj.dff_ct[pkey]) != 0, f"Failed to set attributes self.dffs_ct for {ica_obj.exp_ids[pkey]}"

	# 2. test that input dff files are aligned with ica_obj.rois_valid
    for pkey in ica_obj.pkeys:
	    assert len(ica_obj.dff[pkey]) == len(ica_obj.rois_names_valid[pkey]['roi']), \
		                                     f"dff traces are not alligned wiht rois_vali for exp {ica_obj.exp_ids[pkey]}"
	# 3. test if filtered dff are aligned with ica_obj.rois_valid_ct
    for pkey in ica_obj.pkeys:
        assert len(ica_obj.dff_ct[pkey]) == len(ica_obj.rois_names_valid_ct[pkey]['roi']), \
	                                            f"filtered dff traces are not alligned with rois_valid_ct for exp {ica_obj.exp_ids[pkey]}"

    # 4. test if files were written to disc and both datasets exist in hdf5 files (traces and roi names)
    for pkey in ica_obj.pkeys:
	    assert os.path.isfile(ica_obj.dff_ct_files[pkey]), f"Filtered hdf 5 file does not exist for {ica_obj.exp_ids[pkey]}"
	    f = h5py.File(ica_obj.dff_ct_files[pkey], 'r')
	    datasets = f.keys()
	    assert 'data' in datasets, f"traces are not present in dff_ct file for exp {ica_obj.exp_ids[pkey]}"
	    assert 'roi_names' in datasets, f"roi names are not present in dff_ct file for exp {ica_obj.exp_ids[pkey]}"
	    f.close()


def test_extract_active(session):
	if not session:
		ses = 839208243
		"""LIMS session ID to use for test,
		use a test session with ica and lims processing performed on it to avoid runnign ICA from scratch
		which takes ~ 5 hours"""
	else:
		ses = session

	ica_obj = ica.MesoscopeICA(session_id=ses, cache=CACHE, roi_name="ica_traces", np_name="ica_neuropil")
	pairs = ica_obj.dataset.get_paired_planes()
	pair = pairs[0]
	ica_obj.set_exp_ids(pair)
	ica_obj.get_ica_traces()
	ica_obj.validate_raw_traces(return_vba=False)
	ica_obj.debias_traces()

	pkey = 'pl1'
	tkey = 'roi'
	traces = ica_obj.ins[pkey][tkey]
	traces_sig_evs, traces_ct_evs, valid = ica.extract_active(traces)

	# 1. Check data alignment
	assert traces_sig_evs.shape[0] == traces.shape[1], f"Output of active traces SIGNAL contains different number of " \
													   f"traces (number of rois doesn't align)"
	assert traces_ct_evs.shape[0] == traces.shape[1], f"Output of active traces CROSSTALK contains different number of " \
													  f"traces (number of rois doesn't align)"
	assert len(valid) == traces.shape[1], f"Length of Valid list contains different number of elements than number of " \
										  f"cells in input traces"

	# 2. Check if valid actually has true for traces with no Nans
	for i in range(traces_sig_evs.shape[0]):
		assert valid[i] == (not np.any(np.isnan(traces_sig_evs[i])) and not np.any(np.isnan(traces_ct_evs[i])))

	return


def test_run_ica(snr=30):
	"""
	simulate noisy signals and test ICA function on them
	:param snr: singal to noise ratio to test the ICA with
	:return:
	"""
	# create simulated signals
	npts = 1000
	x = pl.arange(npts)
	sig1 = pl.sin(x / 100) ** 500 + 1 / snr * pl.randn(npts)
	sig2 = pl.sin(x / 84) ** 500 + 1 / snr * pl.randn(npts)

	# Scaling signals
	S = np.array([sig1, sig2])
	D = np.array([[1, 0], [0, 0.3]])  # sclaing matrix
	S_s = np.dot(D, S) # scaled sources

	# Mixing signals
	M = np.array([[0.9, 0.1], [0.2, 0.8]])  # mixing matrix
	O = np.dot(S_s.T, M)  # mixed observations

	mix, a_mix, a_unmix, r_source = run_ica(O[:,0], O[:,1])

	# normallize mixing matrix and compare to original
	b_mix = copy.deepcopy(a_mix)
	b_mix = b_mix / b_mix.sum(axis=0)
	b_mix = b_mix.T

	# test if ICA mixing matrix == original mixing matrix
	assert(np.all(np.isclose(b_mix, M, atol=0.05))), f"Normalized ICA mixign matrix is not similar to original mixing matrix"
	# test if unmixing matrix is inverse of mixing:
	assert(np.all(np.isclose(np.linalg.inv(a_mix), a_unmix))), f"Unmixing is not an inverse of mixing"
	# test the shape of recovered sources is same as shape of input observations:
	assert (r_source.shape == O.shape), f"output of ICA is not shaped the same as input observations"
	return


def test_get_active_traces(session):

	if not session:
		ses = 839208243
		"""LIMS session ID to use for test,
		use a test session with ica and lims processing performed on it to avoid runnign ICA from scratch
		which takes ~ 5 hours"""
	else:
		ses = session

	ica_obj = ica.MesoscopeICA(session_id=ses, cache=CACHE, roi_name="roi", np_name="neuropil")
	pairs = ica_obj.dataset.get_paired_planes()
	pair = pairs[0]
	ica_obj.set_exp_ids(pair)
	ica_obj.get_ica_traces()
	ica_obj.validate_raw_traces(return_vba=False)
	ica_obj.debias_traces()
	ica_obj.get_active_traces("Input")
	self = ica_obj
	for pkey in self.pkeys:
		for tkey in self.tkeys:

			# test if attibutes not nones
			assert len(ica_obj.ins_active[pkey]) != 0, f"Failed to set attributes self.ins_active for {ica_obj.exp_ids[pkey]}"
			assert len(ica_obj.ins_active_paths[pkey]) != 0, f"Failed to set attributes self.ins_active_paths for {ica_obj.exp_ids[pkey]}"
			# testing if output files exist:
			assert os.path.isfile(ica_obj.ins_active_paths[pkey][tkey]), f"Output file doesn't exist for {ica_obj.exp_ids[pkey]}"
			# testing to check if number of rois in ins_active is equal ot number of rois_valid
			assert len(ica_obj.ins_active[pkey][tkey]) == len(ica_obj.rois_names_valid[pkey][tkey]), f"Number of ROIs in ins_active doesnt align with number of valid rois for {ica_obj.exp_ids[pkey]} "
	return


def test_get_ica_active_events(session):
	if not session:
		ses = 839208243
		"""LIMS session ID to use for test,
		use a test session with ica and lims processing performed on it to avoid runnign ICA from scratch
		which takes ~ 5 hours"""
	else:
		ses = session

	ica_obj = ica.MesoscopeICA(session_id=ses, cache=CACHE, roi_name="roi", np_name="neuropil")
	pairs = ica_obj.dataset.get_paired_planes()
	pair = pairs[0]
	ica_obj.set_exp_ids(pair)
	ica_obj.get_ica_traces()
	ica_obj.validate_raw_traces(return_vba=False)
	ica_obj.debias_traces()
	ica_obj.get_active_traces("Input")
	self = ica_obj
	for pkey in self.pkeys:
		for tkey in self.tkeys:
			traces_active = ica_obj.ins_active[pkey][tkey]
			traces = ica_obj.ins[pkey][tkey]
			dict_traces_active = self.get_ica_active_events(traces, traces_active)

			assert len(dict_traces_active) == len(ica_obj.rois_names_valid[pkey][tkey]), f"Length of dict_traces_active is not the same as rois_names_valid for {ica_obj.exp_ids[pkey]}"

	return


def test_unmix_plane(session):
	if not session:
		ses = 839208243
		"""LIMS session ID to use for test,
		use a test session with ica and lims processing performed on it to avoid runnign ICA from scratch
		which takes ~ 5 hours"""
	else:
		ses = session

	ica_obj = ica.MesoscopeICA(session_id=ses, cache=CACHE, roi_name="ica_traces", np_name="ica_neuropil")
	pairs = ica_obj.dataset.get_paired_planes()
	pair = pairs[0]
	ica_obj.set_exp_ids(pair)
	ica_obj.get_ica_traces()
	ica_obj.validate_raw_traces(return_vba=False)
	ica_obj.debias_traces()
	self = ica_obj
	pkey = 'pl1'

	# testing for roi functionality : do full unmix
	tkey = 'roi'
	traces_in = self.ins[pkey][tkey]
	traces_in_active = self.get_ica_active_events(self.ins[pkey][tkey], self.ins_active[pkey][tkey])
	traces_out, crosstalk, mixing, a_mixing = self.unmix_plane(traces_in, traces_in_active)

	assert traces_out.shape == traces_in.shape, f"output traces are not shaped the same as input traces"
	assert len(a_mixing) != 0, f"No mixing matrix returned"
	assert len(crosstalk) != 0, f"No crosstalk data returned"
	assert len(mixing) != 0, f"No raw mixing matrices recorded"

	# testing neuropil - apply roi mixing matrix:
	tkey = 'np'
	traces_in = self.ins[pkey][tkey]
	traces_in_active = self.get_ica_active_events(self.ins[pkey][tkey], self.ins_active[pkey][tkey])
	traces_out, crosstalk, mixing, a_mixing = self.unmix_plane(traces_in, traces_in_active, mixing)

	assert traces_out.shape == traces_in.shape, f"output traces are not shaped the same as input traces"
	assert len(a_mixing) != 0, f"No mixing matrix returned"
	assert len(crosstalk) != 0, f"No crosstalk data returned"
	assert len(mixing) != 0, f"No raw mixing matrices recorded"



	return



