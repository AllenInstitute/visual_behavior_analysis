import pandas as pd
import numpy as np
from dro.utilities import load_from_folder
from .cohorts import basepath, cohort_assignment, mouse_info

mouse_df = cohort_assignment['mouse'].apply(lambda mouse: pd.Series(mouse_info(mouse)))
cohort_assignment = cohort_assignment.merge(
    mouse_df,
    how='left',
    left_on='mouse',
    right_on='mouse_id'
)

   
def get_training_day(df_in):
    '''adds a column to the dataframe with the number of unique training days up to that point
         '''
    day_zero = {r['mouse']:r['day_zero'] for _,r in cohort_assignment.iterrows()}
    coh = cohort_assignment.set_index('mouse')['cohort']
#     print coh

    training_day_lookup = {}
    for mouse, group in df_in.groupby('mouse_id'):
        dates = np.sort(group['date'].unique())
        try:
            dz = day_zero[coh[mouse]]
            offset = np.argwhere(dates==dz)[0][0]
        except KeyError:
            print 'day zero not found for {}'.format(mouse)
            offset = 0
        except IndexError:
            print 'day zero ({}) not found in dates'.format(dz)
            offset = 0
        print offset
        training_day_lookup[mouse] = {date:training_day-offset for training_day,date in enumerate(dates)}
        print mouse,dates
    return df_in.apply(lambda row: training_day_lookup[row['mouse_id']][row['date']],axis=1)


def annotate_trials(trials):

    trials = trials.merge(
        cohort_assignment,
        how='left',
        on='mouse_id',
    )

    trials['training_day'] = get_training_day(trials)

    #df['day_of_week'] = df.startdatetime.dt.weekday_name

    ## build arrays for change detection

    # df['change'] = ~df['change_frame'].isnull()
    trials['change'] = trials['trial_type']=='go'
    trials['detect'] = trials['response']==1.0

    ## calculate reaction times

    trials['reaction_time'] = trials['lick_times'].map(lambda x: x[0] if len(x)>0 else np.nan)
    trials['reaction_time'] = trials.apply(lambda row: row['reaction_time'] - row['change_time'],axis=1)

    ## unwrap the response window
    trials['response_window_lower'] = trials['response_window'].map(lambda x: x[0])
    trials['response_window_upper'] = trials['response_window'].map(lambda x: x[1])

    return trials