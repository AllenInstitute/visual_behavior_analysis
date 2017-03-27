# change detection metrics

def trial_types(trials,trial_types):
    
    if trial_types is not None and len(trial_types)>0:
        return trials['trial_type'].isin(trial_types)
    else:
        return np.ones((len(trials),),dtype=bool)

def continent_trials(trials):
    """ GO & CATCH trials only """
    return trial_types(trials,('go','catch'))

def motivated(trials,reward_rate_thresh=2.0):
    """ masks trials where the reward rate (per minute) is below some threshold.

    This de facto omits trials in which the animal was not licking for extended periods
    or periods when they were licking indiscriminantly.

    """

    mask = trials['reward_rate']>reward_rate_thresh
    return mask