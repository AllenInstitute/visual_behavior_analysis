import pytest

import os

from visual_behavior.translator.foraging2 import \
    data_to_visual_presentation_table
from visual_behavior.schemas.visual_presentation_table import \
    VisualPresentationTable
from visual_behavior.validation.utils import assert_is_valid_dataframe


here = os.path.dirname(os.path.abspath(__file__), )

@pytest.mark.skipif(
    not (len(os.listdir(os.path.join(here, 'assets', ))) > 1),  # .gitignore will exist always hopefully?
    reason='no test assets',
)
def test_visual_presentation_table_doc(doc_data):
    assert_is_valid_dataframe(
        data_to_visual_presentation_table(doc_data),
        VisualPresentationTable()
    )


@pytest.mark.skipif(
    not (len(os.listdir(os.path.join(here, 'assets', ))) > 1),  # .gitignore will exist always hopefully?
    reason='no test assets',
)
def test_visual_presentation_table_ophys(ophys_data):
    assert_is_valid_dataframe(
        data_to_visual_presentation_table(ophys_data),
        VisualPresentationTable()
    )