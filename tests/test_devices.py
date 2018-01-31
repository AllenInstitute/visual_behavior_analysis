import pytest

from visual_behavior import devices


@pytest.mark.parametrize("kwargs, expected", [
    ({"in_val": "W7DTMJ19R2F", "input_type": "computer_name", }, "A1", ),
    ({"in_val": "A2", "input_type": "rig_id", }, "W7DTMJ35Y0T", ),
    ({"in_val": "NOT_A_NAME", "input_type": "computer_name", }, "unknown", ),
])
def test_get_rid_id(kwargs, expected):
    assert devices.get_rig_id(**kwargs) == expected
