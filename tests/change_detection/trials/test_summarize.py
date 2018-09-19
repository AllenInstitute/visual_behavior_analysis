from pandas.testing import assert_frame_equal
from visual_behavior.change_detection.trials import summarize


def test_session_level_summary(mock_trials_fixture, session_summary):
    summary = summarize.session_level_summary(mock_trials_fixture)
    print(summary)
    print(session_summary)
    assert_frame_equal(
        summary,
        session_summary,
        check_names=False,
        check_like=True,
    )


def test_epoch_level_summary(mock_trials_fixture,epoch_summary):
    summary = summarize.epoch_level_summary(mock_trials_fixture)
    print(summary)
    print(epoch_summary)
    assert_frame_equal(
        summary,
        epoch_summary,
        check_names=False,
        check_like=True,
    )
