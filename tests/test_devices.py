import pytest

from visual_behavior import devices


@pytest.mark.parametrize("kwargs, expected", [
    ({"in_val": "W7DTMJ19R2F", "input_type": "computer_name", }, "A1", ),
    ({"in_val": "w7dtmj19r2f", "input_type": "computer_name", }, "A1", ),
    ({"in_val": "A2", "input_type": "rig_id", }, "w7dtmj35y0t", ),
    ({"in_val": "NOT_A_NAME", "input_type": "computer_name", }, "unknown", ),
])
def test_get_rid_id(kwargs, expected):
    assert devices.get_rig_id(**kwargs) == expected
