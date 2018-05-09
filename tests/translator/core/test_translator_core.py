import pytest
import pandas as pd

from visual_behavior.translator import core


def test_create_extended_dataframe(
        exemplar_extended_trials_fixture,
        exemplar_trials_fixture,
        exemplar_metadata_fixture,
        exemplar_licks_fixture,
        exemplar_time_fixture
):
    extended_trials = core.create_extended_dataframe(
        exemplar_trials_fixture,
        exemplar_metadata_fixture,
        exemplar_licks_fixture,
        exemplar_time_fixture
    )

    pd.testing.assert_frame_equal(
        extended_trials,
        exemplar_extended_trials_fixture,
        check_column_type=False,
        check_index_type=False,
        check_dtype=False,
        check_like=True,
        check_names=False
    )
