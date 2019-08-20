

def count_wraps(running_df, direction='forward', lower_threshold=1.5, upper_threshold=3.5):
    '''
    count instances of encoder crossing the 5V/0V threshold. A proxy for the number of rotations.
    '''

    running_df['v_sig_last'] = running_df['v_sig'].shift()

    if direction.lower() == 'forward' or direction.lower() == 'cw':
        wraps = running_df.query("v_sig < @lower_threshold and v_sig_last > @upper_threshold")
    elif direction.lower() == 'backward' or direction.lower() == 'ccw':
        wraps = running_df.query("v_sig > @upper_threshold and v_sig_last < @lower_threshold")

    return len(wraps)
