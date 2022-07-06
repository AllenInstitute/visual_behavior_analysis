import numpy as np  # noqa E902
import pandas as pd
import visual_behavior.utilities as vbu


def calculate_response_matrix(stimuli, aggfunc=np.mean, sort_by_column=True, engaged_only=True):
    '''
    calculates the response matrix for each individual image pair in the `stimulus` dataframe

    Parameters:
    -----------
    stimuli: Pandas.DataFrame
        From experiment.stimulus_presentations, after adding engagement state and annotating as follows:
            experiment.stimulus_presentations = loading.add_model_outputs_to_stimulus_presentations(experiment.stimulus_presentations, behavior_session_id')
            stimulus_presentations = vbu.annotate_stimuli(experiment, inplace = False)
    aggfunc: function
        function to apply to calculation. Default = np.mean
        other options include np.size (to get counts) or np.median
    sort_by_column: Boolean
        if True (default), sorts outputs by column means
    engaged_only: Boolean
        If True (default), calculates only on engaged trials

    Returns:
    --------
    Pandas.DataFrame
        matrix of response probabilities for each image combination
        index = previous image
        column = current image
        catch trials are on diagonal

    '''

    stimuli_to_analyze = stimuli[(stimuli.auto_rewarded==False) & (stimuli.could_change==True) &
                                 (stimuli.image_name!='omitted') & (stimuli.previous_image_name!='omitted')]
    if engaged_only:
        stimuli_to_analyze = stimuli_to_analyze[stimuli_to_analyze.engagement_state=='engaged']

    response_matrix = pd.pivot_table(
        stimuli_to_analyze,
        values='response_lick',
        index=['previous_image_name'],
        columns=['image_name'],
        aggfunc=aggfunc
    ).astype(float)

    if sort_by_column:
        sort_by = response_matrix.mean(axis=0).sort_values().index
        response_matrix = response_matrix.loc[sort_by][sort_by]

    response_matrix.index.name = 'previous_image_name'

    return response_matrix


def calculate_d_prime_matrix(stimuli, sort_by_column=True, engaged_only=True):
    '''
    calculates the d' matrix for each individual image pair in the `stimulus` dataframe

    Parameters:
    -----------
    stimuli: Pandas.DataFrame
        From experiment.stimulus_presentations, after adding engagement state and annotating as follows:
            experiment.stimulus_presentations = loading.add_model_outputs_to_stimulus_presentations(experiment.stimulus_presentations, behavior_session_id')
            stimulus_presentations = vbu.annotate_stimuli(experiment, inplace = False)
    sort_by_column: Boolean
        if True (default), sorts outputs by column means
    engaged_only: Boolean
        If True (default), calculates only on engaged trials

    Returns:
    --------
    Pandas.DataFrame
        matrix of d' for each image combination
        index = previous image
        column = current image
        catch trials are on diagonal

    '''
    response_matrix = calculate_response_matrix(stimuli, aggfunc=np.mean, sort_by_column=sort_by_column, engaged_only=engaged_only)

    d_prime_matrix = response_matrix.copy()
    for row in response_matrix.columns:
        for col in response_matrix.columns:
            d_prime_matrix.loc[row][col] = vbu.dprime(
                hit_rate=response_matrix.loc[row][col],
                fa_rate=response_matrix[col][col],
                limits=False
            )
            if row == col:
                d_prime_matrix.loc[row][col] = np.nan

    return d_prime_matrix
