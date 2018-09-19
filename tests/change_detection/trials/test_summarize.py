import datetime as dt
import numpy as np
import pandas as pd
import pytz
from visual_behavior.change_detection.trials import summarize
from visual_behavior.uuid_utils import make_deterministic_session_uuid


n_tr = 500
np.random.seed(42)
change = np.random.random(n_tr) > 0.8
incorrect = np.random.random(n_tr) > 0.8
detect = change.copy()
detect[incorrect] = ~detect[incorrect]

TRIALS = pd.DataFrame({
    'change': change,
    'detect': detect,
},)
TRIALS['trial_type'] = TRIALS['change'].map(lambda x: ['catch', 'go'][x])
TRIALS['response'] = TRIALS['detect']
TRIALS['change_time'] = np.sort(np.random.rand(n_tr)) * 3600
TRIALS['reward_lick_latency'] = 0.1

metadata = {}
metadata['mouse_id'] = 'M999999'
metadata['startdatetime'] = (
    dt
    .datetime(2017, 7, 19, 10, 35, 8, 369000, tzinfo=pytz.utc)
    .isoformat()
)
metadata['behavior_session_uuid'] = make_deterministic_session_uuid(
    metadata['mouse_id'],
    metadata['startdatetime'],
)
metadata['stage'] = 'test'

for k, v in metadata.items():
    TRIALS[k] = v


def test_session_level_summary():
    print(summarize.session_level_summary(TRIALS))
    assert False


def test_epoch_level_summary():
    print(summarize.epoch_level_summary(TRIALS))
    assert False
