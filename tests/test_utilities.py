from visual_behavior.utilities import local_time

def test_local_time():
    EXPECTED = "2018-05-23T03:55:42.118000-07:00"

    coerced = local_time("2018-05-23T03:55:42.118000",timezone='America/Los_Angeles')
    assert coerced == EXPECTED
