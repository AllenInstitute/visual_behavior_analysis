import pytest

import os
import pandas as pd


TEST_DIR = os.path.dirname(
    os.path.abspath(__file__),
)
ASSETS_DIR = os.path.join(
    TEST_DIR,
    'assets',
)


@pytest.fixture(
    scope='function',  # less efficient but easier for humans...
    ids=[
        'issue_467.pkl',
    ],
    params=[
        {'path': os.path.join(ASSETS_DIR, 'issue_467.pkl', ), }, 
    ],
)
def ophys_data(request):
    return pd.read_pickle(request.param['path'])