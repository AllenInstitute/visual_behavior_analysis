from visual_behavior.ophys.mesoscope.utils import *

def test_get_lims_done_sessions():
    s = 'session_id'
    lims_done, lims_not_done, ica_success = get_lims_done_sessions()
    lims_done = lims_done[s].drop_duplicates().values
    lims_not_done = lims_not_done[s].drop_duplicates().values
    assert len(lims_done) != 0, 'did not find any lims_done sessions'
    assert len(lims_not_done) != 0, 'did not find any lims_done sessions'
    assert len(lims_done)+len(lims_not_done) == len(ica_success), 'numbers of lims_done and lims_not_done sessions do not add up'
    return


def test_get_ica_done_sessions():
    s = 'session_id'
    ica_done, ica_not_done, meso_data = get_ica_done_sessions()
    ica_done = ica_done[s].drop_duplicates().values
    ica_not_done = ica_not_done[s].drop_duplicates().values
    assert len(ica_done) != 0, 'did not find any ica_done sessions'
    assert len(ica_not_done) != 0, 'did not find any ica_done sessions'
    assert len(ica_done)+len(ica_not_done) == len(meso_data), 'numbers of ica_done and ica_not_done sessions do not add up'
    return ica_done, ica_not_done, meso_data


if __name__ == "__main__":
    test_get_lims_done_sessions()
    test_get_ica_done_sessions()



