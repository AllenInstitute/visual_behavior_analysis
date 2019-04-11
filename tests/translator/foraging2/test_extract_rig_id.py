import pytest
import datetime
import pandas as pd
from visual_behavior.translator.foraging2 import get_rig_id
from visual_behavior import devices

DEVICES_DATE = devices.VALID_BEFORE_DATE

def test_rig_id_included():
    data_with_rig_id = {'rig_id':'A1_fromfile'}
    expected = 'A1_fromfile'
    output = get_rig_id(data_with_rig_id)
    assert output == expected

def test_old_data_unknown_rig():
    experiment_date = DEVICES_DATE - datetime.timedelta(days=1)
    old_data_unknown_rig = {'start_time':experiment_date}
    expected = 'A1'
    output = get_rig_id(old_data_unknown_rig)
    assert output == expected

def test_new_data_unknown_rig(caplog):
    experiment_date = DEVICES_DATE + datetime.timedelta(days=1)
    new_data_unknown_rig = {'start_time':experiment_date}
    expected = 'unknown'
    expected_log = ("rig_id unknown, no valid mapping exists for computer "
                    "{} on {}").format('W7DTMJ19R2F', experiment_date)
    output = get_rig_id(new_data_unknown_rig)
    assert output == expected
    assert caplog.records[0].message == expected_log
