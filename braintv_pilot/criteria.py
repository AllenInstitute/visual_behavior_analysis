
import pandas as pd
from sklearn.metrics import accuracy_score
from . import metrics

"""
functions in this module take in a mouse object & return `True` if 
the mouse meets the criterion or `False` otherwise

"""




## decorators: http://thecodeship.com/patterns/guide-to-python-function-decorators/

import pandas as pd

from functools import wraps

def requires_daily_metrics(**metrics_to_calculate):
    """
    This decorator is for functions that require the mouse object to have a session_summary dataframe with specific
    
    It takes as keyword arguments the functions that are computed on each day of the dataframe.
    
    For example, the function `two_out_of_three_aint_bad` assumes that the mouse object has session_summary dataframe
    which has a column titled 'dprime_peak'. If it does not, it will create it using the function passed in on each
    training day dataframe.
    
        @requires_daily_metrics(dprime_peak=metrics.peak_dprime)
        def two_out_of_three_aint_bad(mouse):
            criteria = (mouse.session_summary[-3:]['dprime_peak']>2).sum() > 1
            return criteria==True
    
    """
    
    def requires_metrics_decorator(func_needs_metrics):
        
        @wraps(func_needs_metrics) # <- this is important to maintain attributes of the wrapped function
        def wrapper(anymouse):
            
            try:
                def calculator(group):
                    result = {
                        metric:func(group) 
                        for metric,func 
                        in metrics_to_calculate.iteritems() 
                        if metric not in anymouse.session_summary.columns
                    }
                    return pd.Series(result, name='metrics')
                
                new_summary_data = (
                    anymouse.trials
                        .groupby('training_day')
                        .apply(calculator)
                        )
                anymouse.session_summary = anymouse.session_summary.join(new_summary_data,rsuffix='_joined')
                
            except AttributeError as e:
                print 'oops, AttributeError: {}'.format(e)
                
                def calculator(group):
                    result = {
                        metric:func(group) 
                        for metric,func 
                        in metrics_to_calculate.iteritems()
                    }
                    return pd.Series(result, name='metrics')
                
                anymouse.session_summary = (
                    anymouse.trials
                        .groupby('training_day')
                        .apply(calculator)
                        )
                
            return func_needs_metrics(anymouse)
        return wrapper
    return requires_metrics_decorator


@requires_daily_metrics(dprime_peak=metrics.peak_dprime)
def two_out_of_three_aint_bad(mouse):
    """
    the mouse meets this criterion if 2 of the last 3 days showed a peak d-prime above 2
    """
    print 'calculating criteria'
    criteria = (mouse.session_summary[-3:]['dprime_peak']>2).sum() > 1
    return criteria==True

@requires_daily_metrics(
    response_bias=lambda trials: metrics.response_bias(trials,'detect'),
    )
def no_response_bias(mouse):
    """
    the mouse meets this criterion if their last session exhibited a
    response bias between 10% and 90%
    """
    criteria = (
        (mouse.session_summary[-1]['response_bias']<0.9)
        & (mouse.session_summary[-1]['response_bias']>0.1)
        )

    return criteria==True