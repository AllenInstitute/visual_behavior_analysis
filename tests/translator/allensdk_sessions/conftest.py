import os
import pytest
import pandas as pd
data_dir = '/home/nick.ponvert/nco_home/data/sdk_master_migration'

@pytest.fixture(scope="session")
def sfn_sdk_extended_stimulus_presentations():
    full_path = os.path.join(data_dir,'session_880753403_extended_stimulus_presentations.h5')
    return pd.read_hdf(full_path, key='df')

@pytest.fixture(scope="session")
def sfn_sdk_stimulus_presentations():
    full_path = os.path.join(data_dir,'session_880753403_stimulus_presentations.h5')
    return pd.read_hdf(full_path, key='df')

@pytest.fixture(scope="session")
def sfn_sdk_licks():
    full_path = os.path.join(data_dir,'session_880753403_licks.h5')
    return pd.read_hdf(full_path, key='df')

@pytest.fixture(scope="session")
def sfn_sdk_licks():
    full_path = os.path.join(data_dir,'session_880753403_licks.h5')
    return pd.read_hdf(full_path, key='df')[['timestamps']]

@pytest.fixture(scope="session")
def sfn_sdk_rewards():
    full_path = os.path.join(data_dir,'session_880753403_rewards.h5')
    return pd.read_hdf(full_path, key='df')

@pytest.fixture(scope="session")
def sfn_sdk_trials():
    full_path = os.path.join(data_dir,'session_880753403_trials.h5')
    return pd.read_hdf(full_path, key='df')

@pytest.fixture(scope="session")
def sfn_sdk_running_speed():
    full_path = os.path.join(data_dir,'session_880753403_running_speed.h5')
    return pd.read_hdf(full_path, key='df')
