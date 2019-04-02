import pytest

import pandas as pd


@pytest.fixture(
    scope='function',  # less efficient but easier for humans...
    ids=[
        'issue_467.pkl',
    ],
    params=[
        {'path': './assets/issue_467.pkl', }, 
    ],
)
def ophys_data(request):
    return pd.read_pickle(request.param['path'])