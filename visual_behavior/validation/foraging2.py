
def validate_frame_intervals_exists(data):
    '''
    ensure that frame intervals exist in PKL file
    takes full pickled data object as input
    '''
    return len(data['items']['behavior']['intervalsms']) > 0
