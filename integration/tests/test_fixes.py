import pytest

from visual_behavior.translator.core import create_extended_dataframe
from visual_behavior.translator.foraging2 import data_to_change_detection_core


def test_issue_467_static_movies_present(
    ophys_data,
):
    """ensures issue 467 is resolved:
    - static movies are present in 
    extended trials object
    """
    core = data_to_change_detection_core(ophys_data)
    extended = create_extended_dataframe(
        trials=core['trials'],
        metadata=core['metadata'],
        licks=core['licks'],
        time=core['time'],
    )
    
    raise ValueError(ophys_data['items']['behavior']['items'].keys(), extended.initial_image_name.unique())