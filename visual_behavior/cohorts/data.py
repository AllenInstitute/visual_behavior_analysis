import numpy as np
import pandas as pd
from visual_behavior.data import inplace
from visual_behavior.cohorts import load_cohort_assignment, mouse_info


@inplace
def annotate_cohort_info(trials):
    """ adds cohort metadata to the dataframe

    Parameters
    ----------
    trials : pandas DataFrame
        dataframe of trials
    inplace : bool, optional
        modify `trials` in place. if False, returns a copy. default: True

    See Also
    --------
    io.load_trials
    """

    cohort_assignment = load_cohort_assignment()

    trials = trials.merge(
        cohort_assignment,
        how='left',
        left_on='mouse_id',
        right_on='mouse',
    )

@inplace
def annotate_mouse_info(trials):
    """ adds mouse_info metadata to the dataframe

    Parameters
    ----------
    trials : pandas DataFrame
        dataframe of trials
    inplace : bool, optional
        modify `trials` in place. if False, returns a copy. default: True

    See Also
    --------
    io.load_trials
    """

    mouse_df = pd.Series(
        trials['mouse_id'].unique()
        ).apply(lambda mouse: pd.Series(mouse_info(mouse)))

    trials = trials.merge(
        mouse_df,
        how='left',
        left_on='mouse_id',
        right_on='mouse_id'
    )


def get_training_day(df_in):
    """returns a column with the number of unique training days in the dataframe

    NOTE: training days calculated from unique dates in dataframe, so this is
    unreliable if not all sessions are loaded into the dataframe
    """

    cohort_assignment = load_cohort_assignment()
    cohort_assignment = cohort_assignment[~pd.isnull(cohort_assignment['day_zero'])]

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
    """adds a column to the dataframe with the number of unique training days

    NOTE: training days calculated from unique dates in dataframe, so this is
    unreliable if not all sessions are loaded into the dataframe

    Parameters
    ----------
    trials : pandas DataFrame
        dataframe of trials
    inplace : bool, optional
        modify `trials` in place. if False, returns a copy. default: True

    See Also
    --------
    io.load_trials
    """
    trials['training_day'] = get_training_day(trials)

@inplace
def annotate_training_info(trials):
    """ performs multiple annotatations:

    - annotate_mouse_info
    - annotate_cohort_info
    - annotate_training_day

    Parameters
    ----------
    trials : pandas DataFrame
        dataframe of trials
    inplace : bool, optional
        modify `trials` in place. if False, returns a copy. default: True

    See Also
    --------
    io.load_trials
    """

    annotate_mouse_info(trials,inplace=True)
    annotate_cohort_info(trials,inplace=True)
    annotate_training_day(trials,inplace=True)
