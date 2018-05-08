import numpy as np
import pandas as pd
from visual_behavior.validation.core import validate_running_data, validate_licks

def test_validate_running_data():
    #good data: length matches time and not all values the same
    GOOD_DATA={}
    GOOD_DATA['running']=pd.DataFrame({
        'speed':[3,4,5]
    })
    GOOD_DATA['time']=np.array([1,2,3])
    assert validate_running_data(GOOD_DATA)==True

    #bad data: length does not matche time
    BAD_DATA={}
    BAD_DATA['running']=pd.DataFrame({
        'speed':[3,4,5,6]
    })
    BAD_DATA['time']=np.array([1,2,3])
    assert validate_running_data(BAD_DATA)==False

    #bad data: all values of speed are zero
    BAD_DATA={}
    BAD_DATA['running']=pd.DataFrame({
        'speed':[0,0,0]
    })
    BAD_DATA['time']=np.array([1,2,3])
    assert validate_running_data(BAD_DATA)==False

def test_validate_licks():
    GOOD_DATA = {}
    GOOD_DATA['licks']=pd.DataFrame({
        'time':[10,20],
        'frame':[1,2]
    })
    assert validate_licks(GOOD_DATA)==True

    BAD_DATA = {}
    BAD_DATA['licks']=pd.DataFrame({
        'time':[],
        'frame':[]
    })
    assert validate_licks(BAD_DATA)==False
