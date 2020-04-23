import visual_behavior.ophys.mesoscope.utils as mu

# global constants used in all utils

ITER_ROI = 50 # number of iterations for demixing roi traces
ITER_NP = 50 # number of iterations for demixing neuropil traces
CACHE = '/media/rd-storage/Z/vba_testing'


def test_get_lims_done_sessions():
    # 1 test on all mesoscope sessions
    lims_done = mu.get_lims_done_sessions()
    return lims_done


def test_get_ica_done_sessions():
    ica_done = mu.get_ica_done_sessions()
    return ica_done


def test_get_demixing_done_sessions():

    demixing_done_sessions = mu.get_demixing_done_sessions()
    return demixing_done_sessions

if __name__ == "__main__":
    test_get_lims_done_sessions()
    test_get_ica_done_sessions()
    test_get_demixing_done_sessions()



