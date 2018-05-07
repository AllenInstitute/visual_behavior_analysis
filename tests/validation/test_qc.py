from visual_behavior.validation.qc import compute_qc_metrics, check_session_passes

def test_check_session_passes():

    PASS = {'metric_1': True, 'metric_2': True}
    assert check_session_passes(PASS)==True

    FAIL = {'metric_1': False, 'metric_2': True}
    assert check_session_passes(FAIL)==False
