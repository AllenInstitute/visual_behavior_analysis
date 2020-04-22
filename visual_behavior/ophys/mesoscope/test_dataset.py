import os
import visual_behavior.ophys.mesoscope.dataset as ms


def test_get_splitting_json(test_session=None):
	"""
	test for visual_behavior.ophys.mesoscope.crosstalk_unmix.MesoscopeICA.get_ica_traces()
	Testing that:
		returned file exists
	:return:
	"""
	if not test_session:
		session = 958772311
	else:
		session = test_session
	ds = ms.MesoscopeDataset(session_id=session)
	splitting_json_path = ds.get_splitting_json()

	# 1. Test if a valid filepath is returned
	assert os.path.isfile(splitting_json_path), f"Failed find a slitting json for session {session}"
