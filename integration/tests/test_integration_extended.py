import pytest

import os
from visual_behavior.translator.core import create_extended_dataframe
from visual_behavior.translator.foraging2 import data_to_change_detection_core


TEST_DIR = os.path.dirname(
    os.path.abspath(__file__),
)
ASSETS_DIR = os.path.join(
    TEST_DIR,
    'assets',
)


@pytest.fixture
def core_data(
    doc_data,
):
  return data_to_change_detection_core(doc_data)


@pytest.mark.skipif(
    not os.path.exists(os.path.join(ASSETS_DIR, 'issue_482.pkl', )),
    reason="no asset to test",
)
def test_issue_482_last_trial(
    core_data,
):
    extended_data = create_extended_dataframe(
        trials=core_data['trials'],
        metadata=core_data['metadata'],
        licks=core_data['licks'],
        time=core_data['time'],
        visual_stimuli=core_data['visual_stimuli'],
    )

    assert (extended_data['trial_length'].iloc[-1]) < 100, 
        "last trial length should be less than a 'really long time'(100 seconds)"