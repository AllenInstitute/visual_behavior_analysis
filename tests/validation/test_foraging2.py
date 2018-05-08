from visual_behavior.validation.foraging2 import validate_frame_intervals_exists

def test_validate_frame_intervals_exists():
    test_data={}
    test_data['items']={}
    test_data['items']['behavior']={}
    #populated list should pass
    test_data['items']['behavior']['intervalsms']=[16,16]
    assert validate_frame_intervals_exists(test_data)==True

    #empty list should fail
    test_data['items']['behavior']['intervalsms']=[]
    assert validate_frame_intervals_exists(test_data)==False
