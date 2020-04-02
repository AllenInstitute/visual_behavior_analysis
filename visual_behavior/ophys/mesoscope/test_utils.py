from visual_behavior.ophys.mesoscope.utils import *

# global constants used in all utils

ITER_ROI = 50 # number of iterations for demixing roi traces
ITER_NP = 50 # number of iterations for demixing neuropil traces
CACHE = '/media/rd-storage/Z/vba_testing'


def test_get_lims_done_sessions():
    s = 'session_id'
    lims_done, lims_not_done, ica_success = get_lims_done_sessions()

    lims_done = lims_done[s].drop_duplicates().values
    lims_not_done = lims_not_done[s].drop_duplicates().values
    ica_success = ica_success[s].drop_duplicates().values

    print(f'number of lims_done sessions: {len(lims_done)}')
    print(f'number of lims_not_done sessions: {len(lims_not_done)}')
    print(f'number of crosstalk demixed sessions: {len(ica_success)}')

    assert len(lims_done) != 0, 'did not find any lims_done sessions'
    assert len(lims_not_done) != 0, 'did not find any lims_done sessions'
    assert len(lims_done)+len(lims_not_done) == len(ica_success), 'numbers of lims_done and lims_not_done sessions do not add up'

    return lims_done, lims_not_done, ica_success


def test_get_ica_done_sessions():
    s = 'session_id'
    ica_done, ica_not_done, meso_data = get_ica_done_sessions()
    ica_done = ica_done[s].drop_duplicates().values
    ica_not_done = ica_not_done[s].drop_duplicates().values
    meso_data = meso_data[s].drop_duplicates().values

    print(f'number of crosstalk demixed sessions: {len(ica_done)}')
    print(f'number of not demixed sessions: {len(ica_not_done)}')
    print(f'total number of mesoscope sessions: {len(meso_data)}')

    assert len(ica_done) != 0, 'did not find any ica_done sessions'
    assert len(ica_not_done) != 0, 'did not find any ica_done sessions'
    assert len(ica_done)+len(ica_not_done) == len(meso_data), 'numbers of ica_done and ica_not_done sessions do not add up'
    
    return ica_done, ica_not_done, meso_data


if __name__ == "__main__":
    test_get_lims_done_sessions()
    test_get_ica_done_sessions()



