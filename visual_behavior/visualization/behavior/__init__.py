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


def get_response_probabilities_dict(behavior_session_ids, engaged_only=False):
    """
    For the provided behavior_session_ids, load or cache response_probabilities matrix across image transitions
    for each session and collect into a dictionary

    response_probabilities: dictionary where keys are behavior_session_ids and values are response probability matrices for each session
                            axes of response probability matrices are 'prior_image_name' and 'image_name'
    """
    response_probabilities = {}
    for behavior_session_id in behavior_session_ids:
        try:  # try to load from saved file
            response_probability = vbu.get_cached_behavior_stats(behavior_session_id, engaged_only=engaged_only,
                                                                 method='response_probability')
        except:  # if it doesnt exist, create it
            response_probability = vbu.cache_response_probability(behavior_session_id, engaged_only=engaged_only)
        response_probabilities[behavior_session_id] = response_probability
    return response_probabilities


def aggregate_response_probability_across_sessions(behavior_session_ids, engaged_only=False):
    """
    Collects image transition response probabilities across the provided behavior_session_ids and reshapes into a dataframe

    response_probabilities: dataframe containing response probabilities for each image transition across multiple behavior sessions
                            columns are 'image_name', 'previous_image_name', 'response_probability', and 'behavior_session_id'
    """
    response_probabilities_dict = get_response_probabilities_dict(behavior_session_ids, engaged_only=engaged_only)
    response_probabilities = pd.DataFrame()
    for behavior_session_id in behavior_session_ids:
        # get the response matrix for this session
        response_probability = response_probabilities_dict[behavior_session_id]
        # unstack it and turn it into a tidy df
        if len(response_probability.keys()) == 8: # dont include anything that doesnt have 8 columns i.e. images
            tmp = pd.DataFrame(response_probability.unstack(), columns=['response_probability'])
            tmp = tmp.reset_index()
            # add behavior session id and concatenate with other sessions
            tmp['behavior_session_id'] = behavior_session_id
            response_probabilities = pd.concat([response_probabilities, tmp])
    return response_probabilities


def average_response_probability_across_sessions(response_probabilities, sort=True):
    """
    Takes dataframe of response probabilities for image transitions across multiple sessions,
    and pivots to a dataframe with prior_image_name as index, image_name as columns, and averaged response probabilities as values
    Sorts columns by mean column value by default (i.e. orders axes / images from lowest to highest mean response probability).

    response_probabilities dataframe can be created using response_probabilities = aggregate_response_probability_across_sessions(behavior_session_ids)

    response_probabilities: dataframe containing response probabilities for each image transition across multiple behavior sessions
                            columns are 'image_name', 'previous_image_name', 'response_probability', and 'behavior_session_id'
    sort: Boolean, if True, will sort columns by the average column value

    returns: response_matrix: dataframe with prior_image_name as index, image_name as columns, mean response_probability as values
    """

    response_matrix = pd.pivot_table(response_probabilities, values='response_probability',
                                     index=['previous_image_name'], columns=['image_name'], aggfunc=np.mean).astype(
        float)

    # sort by average column value
    if sort:
        sort_by = response_matrix.mean(axis=0).sort_values().index.values
        response_matrix = response_matrix.loc[sort_by][sort_by]

    return response_matrix
