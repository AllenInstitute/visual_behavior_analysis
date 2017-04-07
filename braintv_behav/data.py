import pandas as pd
import numpy as np
# from braintv_behav.utilities import load_from_folder
from braintv_behav.cohorts import basepath, load_cohort_assignment, mouse_info

from functools import wraps

def inplace(func):

    @wraps(func)
    def df_wrapper(df,*args,**kwargs):

        try:
            inplace = kwargs.pop('inplace')
        except KeyError:
            inplace = False

        if inplace==False:
            df = df.copy()

        func(df,*args,**kwargs)

        if inplace==False:
            return df
        else:
            return None

    return df_wrapper


@inplace
def annotate_parameters(trials,data,keydict=None):
    if keydict is None:
        return
    else:
        for key,value in keydict.iteritems():
            try:
                trials[key] = [data[value]]*len(trials)
            except KeyError as e:
                trials[key] = None

@inplace
def explode_startdatetime(df):
    df['date'] = df['startdatetime'].dt.date.astype(str)
    df['year'] = df['startdatetime'].dt.year
    df['month'] = df['startdatetime'].dt.month
    df['day'] = df['startdatetime'].dt.day
    df['hour'] = df['startdatetime'].dt.hour
    df['dayofweek'] = df['startdatetime'].dt.weekday

@inplace
def annotate_n_rewards(df):
    try:
        df['number_of_rewards'] = df['reward_times'].map(len)
    except KeyError:
        df['number_of_rewards'] = None

@inplace
def annotate_rig_id(df,data):
    #get the rig_id that the session was run on 
    try:
        df['rig_id'] = data['rig_id']
    except KeyError:
        df['rig_id'] = get_rig_id(df['computer_name'][0])

@inplace
def annotate_startdatetime(df,data):

    df['startdatetime'] = pd.to_datetime(data['startdatetime'])

@inplace
def annotate_cumulative_reward(trials,data):

    #calculate cumulative volume
    try:
        trials['cumulative_volume'] = trials['reward_volume'].cumsum()
    except:
        trials['reward_volume'] = data['rewardvol']*trials['number_of_rewards']
        trials['cumulative_volume'] = trials['reward_volume'].cumsum()

@inplace
def annotate_filename(df,filename):
    df['filepath'] = os.path.split(filename)[0]
    df['filename'] = os.path.split(filename)[-1]

@inplace
def fix_autorearded(df):
    df.rename(columns={'auto_rearded': 'auto_rewarded'}, inplace=True)


@inplace
def annotate_cohort_info(trials):

    cohort_assignment = load_cohort_assignment()

    trials = trials.merge(
        cohort_assignment,
        how='left',
        on='mouse_id',
    )

@inplace
def annotate_mouse_info(trials):

    mouse_df = trials['mouse_id'].unique().apply(lambda mouse: pd.Series(mouse_info(mouse)))

    trials = trials.merge(
        mouse_df,
        how='left',
        left_on='mouse',
        right_on='mouse_id'
    )

@inplace
def annotate_change_detect(trials):

    trials['change'] = trials['trial_type']=='go'
    trials['detect'] = trials['response']==1.0


def get_training_day(df_in):
    '''adds a column to the dataframe with the number of unique training days up to that point
         '''

    cohort_assignment = load_cohort_assignment()

    day_zero = {r['mouse']:r['day_zero'].strftime("%Y-%m-%d") for _,r in cohort_assignment.iterrows()}
    coh = cohort_assignment.set_index('mouse')['cohort']

    training_day_lookup = {}
    for mouse, group in df_in.groupby('mouse_id'):
        dates = np.sort(group['date'].unique())
        try:
            dz = day_zero[mouse]
            offset = np.argwhere(dates==dz)[0][0]
        except KeyError:
            print 'day zero not found for {}'.format(mouse)
            offset = 0
        except IndexError:
            print 'day zero ({}) not found in dates'.format(dz)
            offset = 0
        training_day_lookup[mouse] = {date:training_day-offset for training_day,date in enumerate(dates)}
    return df_in.apply(lambda row: training_day_lookup[row['mouse_id']][row['date']],axis=1)

@inplace
def annotate_training_day(trials):
    trials['training_day'] = get_training_day(trials)

@inplace
def fix_change_time(trials):
    trials['change_time'] = trials['change_time'].map(lambda x: np.nan if x is None else x)

@inplace
def explode_response_window(trials)
    trials['response_window_lower'] = trials['response_window'].map(lambda x: x[0])
    trials['response_window_upper'] = trials['response_window'].map(lambda x: x[1])


@inplace
def annotate_trials_jk_style(trials):

    annotate_mouse_info(trials,inplace=True)
    annotate_cohort_info(trials,inplace=True)

    annotate_training_day(trials,inplace=True)

    ## build arrays for change detection
    annotate_change_detect(trials,inplace=True)

    ## calculate reaction times
    fix_change_time(trials,inplace=True)

    ## unwrap the response window
    explode_response_window(trials,inplace=True)