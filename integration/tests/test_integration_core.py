import pytest

import os
import glob
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
    ophys_data,
):
  return data_to_change_detection_core(ophys_data)


@pytest.mark.skipif(
    not glob.glob(os.path.join(ASSETS_DIR, '*.pkl', )),
    reason="no assets to test",
)
def test_issue_467_static_movies_present(
    core_data,
):
    """ensures issue 467 is resolved:
    - static movies are present in 
    extended trials object
    """
    movie_names = []
    for image_name, image_meta in \
            core_data['image_set']['metadata'].items():
        if image_name.startswith('movie:'):
            movie_names.append(image_name.split(':')[1])  # "movie:.*" => ".*"
    
    movie_images_present = len(list(filter(
        lambda name: name in movie_names,
        list(core_data['visual_stimuli'].image_category.unique()),
    ))) > 0

    # eval order important...do not remove parens
    # movie metadata doesnt necessarily mean movie presentations but
    # it will be a fine assumption for now...?
    assert (len(movie_names) > 0 and movie_images_present), \
        "movie metadata and movie stim presentations are expected to exist together"