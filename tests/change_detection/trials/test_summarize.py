from pandas.testing import assert_frame_equal
from visual_behavior.change_detection.trials import summarize


def test_session_level_summary(mock_trials_fixture, session_summary):
    summary = summarize.session_level_summary(mock_trials_fixture)
    assert_frame_equal(
        summary,
        session_summary,
        check_names=False,
        check_like=True,
        check_dtype=False
    )


def test_epoch_level_summary(mock_trials_fixture, epoch_summary):
    summary = summarize.epoch_level_summary(mock_trials_fixture)
    assert_frame_equal(
        summary,
        epoch_summary,
        check_names=False,
        check_like=True,
        check_dtype=False,
        check_exact=False,
        check_less_precise=2,
    )
