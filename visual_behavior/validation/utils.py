import numpy as np
import pandas as pd


def assert_is_valid_dataframe(df, schema_instance):
    records = df.to_dict('records')
    errors = schema_instance.validate(records, many=True)
    assert (len(errors) == 0), errors


def nanis_equal(v1, v2):
    '''
    checks equality, but deals with nulls (e.g., will return True for
    np.nan==None)
    '''
    if pd.isnull(v1):
        v1 = None
    if pd.isnull(v2):
        v2 = None
    return v1 == v2


def all_close(v, tolerance=0.01):
    '''
    adapts the numpy allclose method to work on single array
    creates two arrays that are shifted versions of the input, pads with
    initial and final values, compares. equivalent to iterating through v and
    comparing each element to the next
    '''
    a = np.concatenate(([v[0]], v))
    b = np.concatenate((v, [v[-1]]))
    return np.allclose(a, b, rtol=tolerance)


def even_sampling_mode_enabled(data):
    '''
    check the foraging2 data structure to determine whether even sampling mode is enabled
    '''
    stim = data['items']['behavior']['stimuli'].values()[0]
    if stim['obj_type'].lower() == 'docimagestimulus' and stim['sampling'] in ['even', 'file']:
        return True
    else:
        return False
