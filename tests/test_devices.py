import pytest

from visual_behavior import devices


@pytest.mark.parametrize("computer_name, rig_id", [
    ("W7DTMJ19R2F", "A1"),
    ("w7dtmj19r2f", "A1"),
    ("NOT_A_NAME", "unknown"),
])
def test_get_rid_id(computer_name, rig_id):
    assert devices.get_rig_id(computer_name) == rig_id

    
@pytest.mark.parametrize("rig_id, computer_name", [
    ("A2","w7dtmj35y0t"),
    ("NOT_A_RIG", "unknown"),
])
def test_get_computer_name(rig_id, computer_name):
    assert devices.get_computer_name(rig_id) == computer_name
