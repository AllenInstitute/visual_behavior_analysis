import os
import warnings

import pytest
import pandas as pd
import numpy as np

from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset



def safe_makedirs(path):
    # exist_ok is not a thing in Python 2
    try:
        os.makedirs(path)
    except OSError as err:
        pass


    
@pytest.fixture
def ophys_metadata():
    return pd.DataFrame({
        'a': [1, 2, 3],
        'b': [4, 5, 6]
    })



@pytest.fixture
def ophys_timestamps():
    return pd.DataFrame({
        'stimulus_frames': [np.arange(10)]
    }, index=['timestamps'])
    
    
    
@pytest.fixture
def ophys_data_dir(tmpdir_factory, ophys_metadata, ophys_timestamps):
    warnings.simplefilter("ignore")
    
    eid = 12345678
    tmp_dir = str(tmpdir_factory.mktemp('ophys_dataset_01'))
    
    # analysis folder
    analysis_folder_name = '{}_analysis'.format(eid)
    analysis_folder_path = os.path.join(tmp_dir, analysis_folder_name)
    safe_makedirs(analysis_folder_path)
    
    # metadata
    md_path = os.path.join(analysis_folder_path, 'metadata.h5')
    ophys_metadata.to_hdf(md_path, key='df', format='fixed')
    
    # timestamps
    ts_path = os.path.join(analysis_folder_path,  'timestamps.h5')
    ophys_timestamps.to_hdf(ts_path, key='df', format='fixed')
    
    return tmp_dir



@pytest.fixture
def ophys_dataset(ophys_data_dir):
    return VisualBehaviorOphysDataset(12345678, ophys_data_dir)
    
    
@pytest.mark.parametrize('fn', [
    lambda ds: ds.analysis_folder,
    lambda ds: ds.get_analysis_folder()
])
def test_analysis_folder(fn, ophys_dataset):

    obtained = fn(ophys_dataset)
    assert obtained == '12345678_analysis'
    
    
@pytest.mark.parametrize('fn', [
    lambda ds: ds.analysis_dir,
    lambda ds: ds.get_analysis_dir()
])
def test_analysis_dir(fn, ophys_dataset, ophys_data_dir):
    
    obtained = fn(ophys_dataset)
    assert obtained == os.path.join(ophys_data_dir, '12345678_analysis')
    
    
@pytest.mark.parametrize('fn', [
    lambda ds: ds.metadata,
    lambda ds: ds.get_metadata()
])
def test_get_metadata(fn, ophys_dataset, ophys_metadata):
    
    obtained = fn(ophys_dataset)
    pd.testing.assert_frame_equal(obtained, ophys_metadata, check_like=True)
    
    
@pytest.mark.parametrize('fn', [
    lambda ds: ds.timestamps,
    lambda ds: ds.get_timestamps()
])
def test_get_timestamps(fn, ophys_dataset, ophys_timestamps):
    
    obtained = fn(ophys_dataset)
    pd.testing.assert_frame_equal(obtained, ophys_timestamps, check_like=True)
    
    
@pytest.mark.parametrize('fn', [
    lambda ds: ds.timestamps_stimulus,
    lambda ds: ds.get_timestamps_stimulus()
])
def test_get_timestamps_stimulus(fn, ophys_dataset, ophys_timestamps):
    
    obtained = fn(ophys_dataset)
    assert np.allclose(obtained, ophys_timestamps['stimulus_frames']['timestamps'])