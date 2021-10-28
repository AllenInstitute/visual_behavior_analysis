import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import visual_behavior.data_access.loading as loading

from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache


def get_pref_image_for_group(group, image_column_name='image_name'):
    """
    must be a group with index cell_specimen_id with a 'mean_response' column indicating
    the average response across all presentations of each image
    image_column_name: can be 'image_name', 'prior_image_name', or 'change_image_name'
    returns the value of image_column_name corresponding to the max response across images for each cell_specimen_id
    """
    pref_image = group[(group.mean_response == np.max(group.mean_response.values))][image_column_name].values[0]
    return pd.Series({'pref_image': pref_image})


def get_pref_image_for_cell_specimen_ids(stimulus_response_df):
    """
    takes a stimulus_response_df, with multiple rows per image per cell, averages over all presentations of each image,
    then determines which image evoked the largest mean response.
    returns a dataframe with index cell_specimen_id and column pref_image indicating
    which image_name evoked the largest response for each cell
    """
    pref_images = stimulus_response_df.groupby(['cell_specimen_id', 'image_name']).mean().reset_index().groupby(
        'cell_specimen_id').apply(get_pref_image_for_group)
    return pref_images


def add_pref_image(stimulus_response_df, pref_images):
    """
    takes a stimulus_response_df with index of cell_specimen_id and a column 'image_name',
    creates a new column called 'pref_image' and sets it to True for all values of 'image_name'
    where 'image_name' is in the pref_images dataframe for each cell_specimen_id

    stimulus_response_df: dataframe with index of cell_specimen_id and a column for 'image_name', can have many rows per image_name
    pref_images: dataframe with index of cell_specimen_id and a column 'pref_image' where the value
    is the preferred image for that cell_specimen_id (the image that evoked the largest mean response).
    There should only be one value per cell_specimen_id for 'pref_image' in the pref_images dataframe
    """
    stimulus_response_df['pref_image'] = False
    for cell_specimen_id in stimulus_response_df.cell_specimen_id.unique():
        pref_image_name = pref_images.loc[cell_specimen_id].pref_image
        indices = stimulus_response_df[(stimulus_response_df.cell_specimen_id == cell_specimen_id) & (
            stimulus_response_df.image_name == pref_image_name)].index
        stimulus_response_df.at[indices, 'pref_image'] = True
    return stimulus_response_df


def get_non_pref_image_for_group(group, image_column_name='image_name'):
    """
    must be a group with index cell_specimen_id with a 'mean_response' column indicating
    the average response across all presentations of each image
    image_column_name: can be 'image_name', 'prior_image_name', or 'change_image_name'
    returns the value of image_column_name corresponding to the min response across images for each cell_specimen_id
    """
    non_pref_image = group[(group.mean_response == np.min(group.mean_response.values))][image_column_name].values[0]
    return pd.Series({'non_pref_image': non_pref_image})


def get_non_pref_image_for_cell_specimen_ids(stimulus_response_df):
    """
    takes a stimulus_response_df, with multiple rows per image per cell, averages over all presentations of each image,
    then determines which image evoked the smallest mean response.
    returns a dataframe with index cell_specimen_id and column non_pref_image indicating
    which image_name evoked the smallest response for each cell
    """
    non_pref_images = stimulus_response_df.groupby(['cell_specimen_id', 'image_name']).mean().reset_index().groupby(
        'cell_specimen_id').apply(get_non_pref_image_for_group)
    return non_pref_images


def add_non_pref_image(stimulus_response_df, non_pref_images):
    """
    takes a stimulus_response_df with index of cell_specimen_id and a column 'image_name',
    creates a new column called 'non_pref_image' and sets it to True for all values of 'image_name'
    where 'image_name' is in the pref_images dataframe for each cell_specimen_id

    stimulus_response_df: dataframe with index of cell_specimen_id and a column for 'image_name', can have many rows per image_name
    pref_images: dataframe with index of cell_specimen_id and a column 'non_pref_image' where the value
    is the preferred image for that cell_specimen_id (the image that evoked the smallest mean response).
    There should only be one value per cell_specimen_id for 'pref_image' in the pref_images dataframe
    """
    stimulus_response_df['non_pref_image'] = False
    for cell_specimen_id in stimulus_response_df.cell_specimen_id.unique():
        non_pref_image_name = non_pref_images.loc[cell_specimen_id].non_pref_image
        indices = stimulus_response_df[(stimulus_response_df.cell_specimen_id == cell_specimen_id) & (
            stimulus_response_df.image_name == non_pref_image_name)].index
        stimulus_response_df.at[indices, 'non_pref_image'] = True
    return stimulus_response_df


def add_pref_and_non_pref_stim_columns_to_df(stimulus_response_df):
    """
    adds columns for 'pref_image' and 'non_pref_image' to stimulus response dataframe
    based on max and min average image evoked responses for each cell_specimen_id
    """
    pref_images = get_pref_image_for_cell_specimen_ids(stimulus_response_df)
    stimulus_response_df = add_pref_image(stimulus_response_df, pref_images)

    non_pref_images = get_non_pref_image_for_cell_specimen_ids(stimulus_response_df)
    stimulus_response_df = add_non_pref_image(stimulus_response_df, non_pref_images)

    return stimulus_response_df


def get_image_selectivity_index_pref_non_pref(stimulus_response_df):
    """
    computes the difference over the sum of each cells preferred and non-preferred images.
    takes stimulus_response_df as input, must have columns 'pref_image' and 'non_pref_image'
    returns a dataframe with index cell_specimen_id and column 'image_selectivity_index'
    """
    pref_image_df = stimulus_response_df[stimulus_response_df.pref_image == True]
    mean_response_pref_image = pref_image_df.groupby(['cell_specimen_id']).mean()[['mean_response']]

    non_pref_image_df = stimulus_response_df[stimulus_response_df.non_pref_image == True]
    mean_response_non_pref_image = non_pref_image_df.groupby(['cell_specimen_id']).mean()[['mean_response']]

    image_selectivity_index = (mean_response_pref_image - mean_response_non_pref_image) / (
        mean_response_pref_image + mean_response_non_pref_image)
    image_selectivity_index = image_selectivity_index.rename(columns={'mean_response': 'image_selectivity_index'})
    return image_selectivity_index


def get_image_selectivity_index_one_vs_all(stimulus_response_df):
    """
    computes the difference over the sum of each cells preferred image vs. all other images.
    takes stimulus_response_df as input, must have column 'pref_image'
    returns a dataframe with index cell_specimen_id and column 'image_selectivity_index_one_vs_all'
    """
    pref_image_df = stimulus_response_df[stimulus_response_df.pref_image == True]
    mean_response_pref_image = pref_image_df.groupby(['cell_specimen_id']).mean()[['mean_response']]

    non_pref_images_df = stimulus_response_df[(stimulus_response_df.pref_image == False)]
    mean_response_non_pref_images = non_pref_images_df.groupby(['cell_specimen_id']).mean()[['mean_response']]

    image_selectivity_index = (mean_response_pref_image - mean_response_non_pref_images) / (
        mean_response_pref_image + mean_response_non_pref_images)
    image_selectivity_index = image_selectivity_index.rename(
        columns={'mean_response': 'image_selectivity_index_one_vs_all'})
    return image_selectivity_index


def get_image_selectivity_index(stimulus_response_df):
    image_selectivity = get_image_selectivity_index_pref_non_pref(stimulus_response_df)
    tmp = get_image_selectivity_index_one_vs_all(stimulus_response_df)
    image_selectivity = image_selectivity.merge(tmp, on='cell_specimen_id')
    return image_selectivity


def compute_lifetime_sparseness(image_responses):
    # image responses should be a list or array of the trial averaged responses to each image, for some condition (ex: go trials only, engaged/disengaged etc)
    # sparseness = 1-(sum of trial averaged responses to images / N)squared / (sum of (squared mean responses / n)) / (1-(1/N))
    # N = number of images
    N = float(len(image_responses))
    # modeled after Vinje & Gallant, 2000
    # ls = (1 - (((np.sum(image_responses) / N) ** 2) / (np.sum(image_responses ** 2 / N)))) / (1 - (1 / N))
    # emulated from https://github.com/AllenInstitute/visual_coding_2p_analysis/blob/master/visual_coding_2p_analysis/natural_scenes_events.py
    # formulated similar to Froudarakis et al., 2014
    ls = ((1 - (1 / N) * ((np.power(image_responses.sum(axis=0), 2)) / (np.power(image_responses, 2).sum(axis=0)))) / (
        1 - (1 / N)))
    return ls


def get_lifetime_sparseness_for_group(group):
    """
    group should have index cell_specimen_id and include the average response (column='mean_response')
    to each image (column='image_name') for each cell_specimen_id.
    computes lifetime sparseness across mean image responses for each cell_specimen_id.
    returns a pandas series with index cell_specimen_id and column lifetime_sparseness
    """
    image_responses = group.mean_response.values
    lifetime_sparseness = compute_lifetime_sparseness(image_responses)
    return pd.Series({'lifetime_sparseness': lifetime_sparseness})


def get_lifetime_sparseness_for_cell_specimen_ids(stimulus_response_df):
    """
    computes lifetime sparseness across images for each cell_specimen_id in the stimulus_response_df
    returns a dataframe with index cell_specimen_id and column lifetime_sparseness for each cell
    """
    lifetime_sparseness_df = stimulus_response_df.groupby(['cell_specimen_id', 'image_name']).mean().reset_index().groupby('cell_specimen_id').apply(get_lifetime_sparseness_for_group)
    return lifetime_sparseness_df


def get_mean_response_cell_specimen_ids(stimulus_response_df):
    """
    takes stimulus_response_df (with any filtering criteria applied, such as limited to pref image),
    and computes the average response across all stimulus_presentations for each cell_specimen_id
    """
    mean_responses = stimulus_response_df.groupby(['cell_specimen_id']).mean()[['mean_response']]
    return mean_responses


def get_fraction_significant_p_value_gray_screen(group):
    """
    takes stimulus_response_df grouped by cell_specimen_id (with desired filtering criteria previously applied)
    and computes the fraction of all stimulus_presentations for each cell_specimen_id where the
    p_value comparing the mean response for each stimulus presentation to a shuffled distribution
    of dFF or events values from the 5 mins gray screen before and after the behavior session
    """
    fraction_significant_p_value_gray_screen = len(group[group.p_value_gray_screen < 0.05]) / float(len(group))
    return pd.Series({'fraction_significant_p_value_gray_screen': fraction_significant_p_value_gray_screen})


def get_fano_factor(group):
    """
    takes stimulus_response_df grouped by cell_specimen_id (with desired filtering criteria previously applied)
    and computes the fano_factor each cell_specimen_id
    """
    mean_responses = group.mean_response.values
    sd = np.nanstd(mean_responses)
    mean_response = np.nanmean(mean_responses)
    fano_factor = np.abs((sd * 2) / mean_response)
    return pd.Series({'fano_factor': fano_factor})


def compute_reliability_vectorized(traces):
    '''
    Compute average pearson correlation between pairs of rows of the input matrix.
    Args:
        traces(np.ndarray): trace array with shape m*n, with m traces and n trace timepoints
    Returns:
        reliability (float): Average correlation between pairs of rows
    '''
    # Compute m*m pearson product moment correlation matrix between rows of input.
    # This matrix is 1 on the diagonal (correlation with self) and mirrored across
    # the diagonal (corr(A, B) = corr(B, A))
    corrmat = np.corrcoef(traces)
    # We want the inds of the lower triangle, without the diagonal, to average
    m = traces.shape[0]
    lower_tri_inds = np.where(np.tril(np.ones([m, m]), k=-1))
    # Take the lower triangle values from the corrmat and averge them
    correlation_values = list(corrmat[lower_tri_inds[0], lower_tri_inds[1]])
    reliability = np.nanmean(correlation_values)
    return reliability, correlation_values


def compute_reliability(group, params, frame_rate):
    # computes trial to trial correlation across input traces in group,
    # only for portion of the trace after the change time or flash onset time

    onset = int(np.abs(params['window_around_timepoint_seconds'][0]) * frame_rate)
    response_window = [onset, onset + (int(params['response_window_duration_seconds'] * frame_rate))]
    traces = group['trace'].values
    traces = np.vstack(traces)
    if traces.shape[0] > 5:
        traces = traces[:, response_window[0]:response_window[1]]  # limit to response window
        reliability, _ = compute_reliability_vectorized(traces)
    else:
        reliability = np.nan
    return pd.Series({'reliability': reliability})


def get_reliability_for_cell_specimen_ids(stimulus_response_df, frame_rate):
    """
    computes the average trial to trial correlation of the dF/F or events trace in the stimulus window
    returns a dataframe with index cell_specimen_id and columns for the reliability value,
    and the individual pairwise correlation values
    """
    import visual_behavior.ophys.response_analysis.response_processing as rp
    params = rp.get_default_stimulus_response_params()

    reliability = stimulus_response_df.groupby(['cell_specimen_id']).apply(compute_reliability, params, frame_rate)
    # reliability = reliability.drop(columns=['correlation_values'])
    return reliability


def add_running_to_df(stimulus_df):
    """
    takes stimulus_presentations table or stimulus_response_df, with a column for mean_running_speed,
    and returns the same df with a column 'running' which is a Boolean that is True if the mean_running_speed
    for the given row is greater than 2 cm/s and False if not
    """
    stimulus_df['running'] = [True if mean_running_speed > 2 else False for mean_running_speed in stimulus_df.mean_running_speed.values]
    return stimulus_df


def get_running_modulation_index_for_group(group):
    """
    takes group with index cell_specimen_id, column 'running' that is a Boolean, and 'mean_response' column
    computes the difference over sum of running vs not running mean response values
    """
    running = group[group.running == True].mean_response.values
    not_running = group[group.running == False].mean_response.values
    running_modulation_index = (running - not_running) / (running + not_running)
    return pd.Series({'running_modulation_index': running_modulation_index})


def get_running_modulation_index_for_cell_specimen_ids(stimulus_response_df):
    """
    computes diff over sum of running vs. not running values each cell in stimulus_response_df.
    stimulus_response_df must have a column 'mean_running_speed', which is converted to a Boolean column
    called 'running' that is True if mean_running_speed is >2 cm/s otherwise False
    """
    stimulus_response_df = add_running_to_df(stimulus_response_df)
    running_modulation_index = stimulus_response_df.groupby(['cell_specimen_id', 'running']).mean()[['mean_response']].reset_index().groupby('cell_specimen_id').apply(get_running_modulation_index_for_group)
    running_modulation_index['running_modulation_index'] = [np.nan if len(index) == 0 else index[0] for index in running_modulation_index.running_modulation_index.values]
    return running_modulation_index


def get_hit_miss_modulation_index_for_group(group):
    """
    """
    hit = group[group.hit == True].mean_response.values
    miss = group[group.hit == False].mean_response.values
    hit_miss_index = (hit - miss) / (hit + miss)
    return pd.Series({'hit_miss_index': hit_miss_index})


def get_hit_miss_modulation_index(stimulus_response_df):
    """
    computes diff over sum of hit trials vs. miss trials for each cell in stimulus_response_df.
    stimulus_response_df must have a columns 'is_change', and 'licked'
    returns a dataframe with column 'hit_miss_index' with computed value for each cell_specimen_id in stimulus_response_df
    """
    stimulus_response_df['hit'] = [True if (stimulus_response_df.iloc[row].is_change == True and stimulus_response_df.iloc[row].licked == True) else False for row in range(len(stimulus_response_df))]
    hit_miss_index = stimulus_response_df.groupby(['cell_specimen_id', 'hit']).mean()[['mean_response']].reset_index().groupby('cell_specimen_id').apply(get_hit_miss_modulation_index_for_group)
    hit_miss_index['hit_miss_index'] = [np.nan if len(index) == 0 else index[0] for index in hit_miss_index.hit_miss_index.values]
    return hit_miss_index


def get_population_coupling_for_cell_specimen_ids(traces):
    """
    input is dataframe of dff_traces or events where index is cell_specimen_id
    computes population coupling value for each cell_specimen_id, where population coupling is the
    pearson correlation of each cell's trace with the population average trace of all other cells
    returns a dataframe with index cell_specimen_id and columns population_coupling_r_value and population_coupling_p_value
    """
    from scipy.stats import pearsonr

    if 'dff' in traces.keys():
        trace_column = 'dff'
    elif 'filtered_events' in traces.keys():
        trace_column = 'filtered_events'

    cell_specimen_ids = traces.index.values
    pc_list = []
    for cell_specimen_id in cell_specimen_ids:
        cell_trace = traces.loc[cell_specimen_id][trace_column]
        population_trace = traces.loc[cell_specimen_ids[cell_specimen_ids != cell_specimen_id]][trace_column].mean()
        r, p_value = pearsonr(cell_trace, population_trace)

        pc_list.append([cell_specimen_id, r, p_value])
    columns = ['cell_specimen_id', 'population_coupling_r_value', 'population_coupling_p_value']
    population_coupling = pd.DataFrame(pc_list, columns=columns)
    population_coupling = population_coupling.set_index('cell_specimen_id')

    return population_coupling


def compute_trace_metrics(traces, ophys_frame_rate):
    if 'dff' in traces.columns:
        column = 'dff'
    elif 'filtered_events' in traces.columns:
        column = 'filtered_events'
    traces["trace_mean"] = traces[column].apply(lambda x: np.nanmean(x))
    traces["trace_max"] = traces[column].apply(lambda x: np.nanmax(x))
    traces["trace_var"] = traces[column].apply(lambda x: np.nanvar(x))
    traces["trace_std"] = traces[column].apply(lambda x: np.nanstd(x))
    traces["trace_max_over_std"] = traces["trace_max"] / traces["trace_std"]
    traces["trace_mean_over_std"] = traces["trace_mean"] / traces["trace_std"]

    traces["noise_level"] = traces[column].apply(lambda x:  np.median(np.abs(np.diff(x))) / np.sqrt(ophys_frame_rate))
    traces["mean_over_noise_level"] = traces["trace_mean"] / traces["noise_level"]

    return traces


def compute_robust_snr_on_dataframe(dataframe, use_events=False, filter_events=False):
    """takes a dataframe with cell_specimen_id as index and dff, events, or filtered_events as columns
       and computes robust signal, noise and snr metrics
       These metrics are borrowed from event detection code
    Returns:
        dataframe -- input dataframe but with the following columns added: "robust_noise", "robust_signal", "robust_snr"
    """
    import visual_behavior.data_access.processing as processing
    if use_events:
        if filter_events:
            column = 'filtered_events'
        else:
            column = 'events'
    else:
        column = 'dff'
    dataframe["robust_noise"] = dataframe[column].apply(lambda x: processing.dff_robust_noise(x))
    dataframe["robust_signal"] = dataframe.apply(lambda x: processing.dff_robust_signal(x[column], x["robust_noise"]), axis=1 )
    dataframe["robust_snr"] = dataframe["robust_signal"] / dataframe["robust_noise"]
    return dataframe


def get_trace_metrics(traces, ophys_frame_rate, use_events=False, filter_events=False):
    """
    compute metrics on a set of cell traces, including SNR, mean etc
    traces input must be a dataframe with cell_specimen_id as index and 'dff', 'events' or 'filtered_events' as column
    Note that 'compute_robust_snr_on_dataframe' does not appear to work on events or filtered events, potentially due to so many zeros?
    :param traces:
    :return:
    """
    import visual_behavior.data_access.processing as processing
    traces = compute_robust_snr_on_dataframe(traces, use_events, filter_events)
    traces = compute_trace_metrics(traces, ophys_frame_rate)
    # reorder
    trace_metrics = traces[['robust_signal', 'robust_noise', 'robust_snr', 'trace_max', 'trace_mean',
                            'trace_var', 'trace_std', 'trace_max_over_std', 'trace_mean_over_std',
                            'noise_level', 'mean_over_noise_level']]
    return trace_metrics


def generate_trace_metrics_table(ophys_experiment_id, use_events=False, filter_events=False, save=False):
    """
    Gets metrics computed over entire cell traces, such as trace SNR, population coupling,
    :param ophys_experiment_id: experiment identifier to compute metrics for
    :param use_events: Boolean
    :param filter_events: Boolean
    :return:
    """
    print('generating trace metrics for', ophys_experiment_id)

    dataset = loading.get_ophys_dataset(ophys_experiment_id)

    if use_events:
        traces = dataset.events.copy()
    else:
        traces = dataset.dff_traces.copy()

    # get standard trace metrics
    ophys_frame_rate = dataset.metadata['ophys_frame_rate']
    trace_metrics = get_trace_metrics(traces, ophys_frame_rate, use_events, filter_events)

    # # get population coupling across all cells full dff traces
    # population_coupling = get_population_coupling_for_cell_specimen_ids(traces)
    #
    # trace_metrics = trace_metrics.merge(population_coupling, on='cell_specimen_id')

    trace_metrics['ophys_experiment_id'] = ophys_experiment_id

    trace_metrics = trace_metrics.reset_index()
    # trace_metrics = trace_metrics.melt(id_vars=['cell_specimen_id', 'ophys_experiment_id', 'ophys_session_id'])
    trace_metrics['condition'] = 'full_trace'
    trace_metrics['session_subset'] = 'full_session'
    trace_metrics['stimuli'] = 'full_session'
    trace_metrics['use_events'] = use_events
    trace_metrics['filter_events'] = filter_events

    if save:
        filepath = get_metrics_df_filepath(ophys_experiment_id, condition='full_trace',
                                           stimuli='full_session', session_subset='full_session',
                                           use_events=use_events, filter_events=filter_events)
        if os.path.exists(filepath):
            os.remove(filepath)
            print('h5 file exists for', ophys_experiment_id, ' - overwriting')
        trace_metrics.to_hdf(filepath, key='df')
        print('trace metrics saved for', ophys_experiment_id)

    return trace_metrics


def generate_cell_metrics_table(ophys_experiment_id, use_events=False, filter_events=False,
                           condition='changes', session_subset='full_session', stimuli='pref_image', save=False):
    """
    Creates cell metrics table based on stimulus locked activity
    Metrics include selectivity indices, mean image response, fano factor, fraction significant trials, etc
    NOTE: session_subset = 'engaged' or 'disengaged' currently uses reward rate as engagement metric
    The engagement_state from the behavior model will need to be added to the extended_stimulus_presentations table
    in loading.get_ophys_dataset() to be able to be used here.
    :param ophys_experiment_id: unique identifier for experiment
    :param condition: 'changes', 'omissions', 'images'
    :param stimuli: 'all_images', 'pref_image'
    :param session_subset: 'engaged', 'disengaged', or 'full_session'
    :param use_events: Boolean
    :param filter_events: Boolean
    :return:
    """

    print('generating cell metrics for', ophys_experiment_id)

    dataset = loading.get_ophys_dataset(ophys_experiment_id)

    analysis = ResponseAnalysis(dataset, use_extended_stimulus_presentations=True, use_events=use_events, filter_events=filter_events)
    sdf = analysis.get_response_df(df_name='stimulus_response_df')

    if condition == 'changes':
        df = sdf[sdf.is_change == True]
    elif condition == 'omissions':
        df = sdf[sdf.omitted == True]
        # use next image name for computing pref image, selectivity, etc.
        df['image_name'] = [df.iloc[row].image_name_next_flash for row in range(len(df))]
    elif condition == 'images':
        df = sdf[sdf.omitted == False]
    else:
        print('condition name provided does not meet requirements:', condition)
        print('please check documentation for cell_metrics.generate_cell_metrics_table()')

    if 'passive' in dataset.metadata['session_type']:
        df['engaged'] = False
    if session_subset == 'engaged':
        df['engaged'] = [True if reward_rate >= 2 else False for reward_rate in df.reward_rate.values]
        df = df[df['engaged']==True]
    elif session_subset == 'disengaged':
        df['engaged'] = [True if reward_rate >= 2 else False for reward_rate in df.reward_rate.values]
        df = df[df['engaged']==False]


    df = df.reset_index(drop=True)

    # get pref and non-pref images for each cell
    pref_image = get_pref_image_for_cell_specimen_ids(df)
    non_pref_image = get_non_pref_image_for_cell_specimen_ids(df)

    # add pref & non pref images
    df = add_pref_and_non_pref_stim_columns_to_df(df)
    # compute image selectivity index (diff over sum of pref and non pref images)
    image_selectivity_index = get_image_selectivity_index(df)

    # get lifetime sparseness value for each cell
    lifetime_sparseness = get_lifetime_sparseness_for_cell_specimen_ids(df)

    if stimuli == 'pref_image':
        # restrict further analysis to pref stim condition
        df = df[df.pref_image == True]

    # get mean response across all images
    mean_response = get_mean_response_cell_specimen_ids(df)

    # get fraction responsive trials across all images
    fraction_responsive_trials = df.groupby(['cell_specimen_id']).apply(get_fraction_significant_p_value_gray_screen)

    # get fano factor across trials
    fano_factor = df.groupby(['cell_specimen_id']).apply(get_fano_factor)

    # get average trial to trial reliability across all images
    frame_rate = dataset.metadata['ophys_frame_rate']
    reliability = get_reliability_for_cell_specimen_ids(df, frame_rate)

    # get running modulation - diff over sum of mean image response for running vs. not running trials
    running_modulation_index = get_running_modulation_index_for_cell_specimen_ids(df)

    if condition == 'changes':
        # get choice modulation - diff over sum of mean image response for hit vs. miss trials
        hit_miss_modulation_index = get_hit_miss_modulation_index(df)

    metrics_table = pref_image.copy()
    metrics_table = metrics_table.merge(non_pref_image, on='cell_specimen_id')
    metrics_table = metrics_table.merge(mean_response, on='cell_specimen_id')
    metrics_table = metrics_table.merge(image_selectivity_index, on='cell_specimen_id')
    metrics_table = metrics_table.merge(lifetime_sparseness, on='cell_specimen_id')
    metrics_table = metrics_table.merge(fraction_responsive_trials, on='cell_specimen_id')
    metrics_table = metrics_table.merge(fano_factor, on='cell_specimen_id')
    metrics_table = metrics_table.merge(reliability, on='cell_specimen_id')
    metrics_table = metrics_table.merge(running_modulation_index, on='cell_specimen_id')
    if condition == 'changes':
        metrics_table = metrics_table.merge(hit_miss_modulation_index, on='cell_specimen_id')
    metrics_table['ophys_experiment_id'] = ophys_experiment_id

    metrics_table = metrics_table.reset_index()
    # metrics_table = metrics_table.melt(id_vars=['cell_specimen_id', 'ophys_experiment_id', 'ophys_session_id'])
    metrics_table['condition'] = condition
    metrics_table['session_subset'] = session_subset
    metrics_table['stimuli'] = stimuli
    metrics_table['use_events'] = use_events
    metrics_table['filter_events'] = filter_events

    metrics_table = metrics_table.rename(columns={'variable': 'metric'})

    if save:
        filepath = get_metrics_df_filepath(ophys_experiment_id, condition=condition,
                                           stimuli=stimuli, session_subset=session_subset,
                                           use_events=use_events, filter_events=filter_events)
        if os.path.exists(filepath):
            os.remove(filepath)
            print('h5 file exists for', ophys_experiment_id, ' - overwriting')
        metrics_table.to_hdf(filepath, key='df')
        print('cell metrics saved for', ophys_experiment_id, 'at', filepath)

    return metrics_table


def get_metrics_df_filename(ophys_experiment_id, condition, stimuli, session_subset, use_events, filter_events):
    """
    gets standard filename given a set of conditions, stimuli, etc.
    :param ophys_experiment_id: unique identifier for experiment, or 'all_experiments' to get all experiments from a single file
    :param condition: 'changes', 'omissions', 'images', or 'traces'
    :param stimulus: 'all_images', 'pref_image',
    :param session_subset: 'full_session', 'engaged', 'disengaged'
    :param use_events: Boolean
    :param filter_events: Boolean
    :return:
    """
    if use_events:
        if filter_events:
            trace_type = 'filtered_events'
        else:
            trace_type = 'events'
    else:
        trace_type = 'dFF'
    filename = str(ophys_experiment_id) + '_' + condition + '_' + stimuli + '_' + session_subset + '_' + trace_type
    return filename


def get_metrics_df_filepath(ophys_experiment_id, condition, stimuli, session_subset, use_events, filter_events):
    """
    gets standard filepath given a set of conditions, stimuli, etc.
    :param ophys_experiment_id: unique identifier for experiment, or 'all_experiments' to get all experiments from a single file
    :param condition: 'changes', 'omissions', 'images', or 'traces'
    :param stimulus: 'all_images', 'pref_image',
    :param session_subset: 'full_session', 'engaged', 'disengaged'
    :param use_events: Boolean
    :param filter_events: Boolean
    :return:
    """
    cache_dir = loading.get_platform_analysis_cache_dir()
    save_dir = os.path.join(cache_dir, 'cell_metrics')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    filename = get_metrics_df_filename(ophys_experiment_id, condition, stimuli, session_subset, use_events, filter_events)
    filepath = os.path.join(save_dir, filename + '.h5')
    return filepath


def generate_and_save_all_metrics_tables_for_experiment(ophys_experiment_id):
    """
    For a single ophys_experiment_id, creates trace and cell (stimulus locked) metrics
    for all possible combinations of scenarios one might want to analyze, including:
    condition: 'changes', 'omissions', 'images', or 'traces'
    stimulus: 'all_images', 'pref_image',
    session_subset: 'full_session', 'engaged', 'disengaged'
    conditions include events, filtered_events and dff
    use_events: [True, False], when use_events=True, also go through filter_events=[True, False]

    :param ophys_experiment_id: unique experiment identifier
    :return:
    """

    i = 0
    problem_expts = pd.DataFrame()

    ### trace metrics ###
    for use_events in [False, True]:
        for filter_events in [False, True]:
            try:
                trace_metrics = generate_trace_metrics_table(ophys_experiment_id,
                                                                     use_events=use_events, filter_events=filter_events)
                filepath = get_metrics_df_filepath(ophys_experiment_id, condition='traces',
                                                                stimuli='full_session', session_subset='full_session',
                                                                use_events=use_events, filter_events=filter_events)
                if os.path.exists(filepath):
                    os.remove(filepath)
                    print('h5 file exists for', ophys_experiment_id, ' - overwriting')
                trace_metrics.to_hdf(filepath, key='df')
                print('trace metrics saved for', ophys_experiment_id)
            except Exception as e:
                print('metrics not generated for trace_metrics for experiment', ophys_experiment_id)
                print(e)
                problem_expts.loc[i, 'ophys_experiment_id'] = ophys_experiment_id
                problem_expts.loc[i, 'condition'] = condition
                problem_expts.loc[i, 'stimuli'] = stimuli
                problem_expts.loc[i, 'session_subset'] = session_subset
                problem_expts.loc[i, 'use_events'] = use_events
                problem_expts.loc[i, 'filter_events'] = filter_events
                problem_expts.loc[i, 'exception'] = e
                i += 1


    ### event locked response metrics ###
    conditions = ['changes', 'omissions', 'images']
    stimuli = ['all_images', 'pref_image']
    session_subsets = ['full_session', 'engaged', 'disengaged']

    metrics_df = pd.DataFrame()
    for condition in conditions:
        for stimulus in stimuli:
            for session_subset in session_subsets:
                for use_events in [True, False]:
                    for filter_events in [True, False]:
                        # need try except because code will not always run, such as in the case of passive sessions (no trials that are 'engaged')
                        # or loops where use_events = False and filter_events = True
                        try:
                            metrics_df = generate_cell_metrics_table(ophys_experiment_id,
                                                                             use_events=use_events, filter_events=filter_events,
                                                                             condition=condition, session_subset=session_subset,
                                                                             stimuli=stimulus)

                            filepath = get_metrics_df_filepath(ophys_experiment_id, condition=condition,
                                                               stimuli=stimulus, session_subset=session_subset,
                                                               use_events=use_events, filter_events=filter_events)
                            if os.path.exists(filepath):
                                os.remove(filepath)
                                print('h5 file exists for', ophys_experiment_id, ' - overwriting')
                            metrics_df.to_hdf(filepath, key='df')
                            print('metrics generated for', condition, stimulus, session_subset,
                                  'use_events', use_events, 'filter_events', filter_events, )

                        except Exception as e:
                            print('metrics not generated for experiment_id', ophys_experiment_id,
                                  'condition', condition, 'stimulus', stimulus,
                                  'session_subset', session_subset, 'use_events', use_events, 'filter_events', filter_events, )
                            print(e)
                            problem_expts.loc[i, 'ophys_experiment_id'] = ophys_experiment_id
                            problem_expts.loc[i, 'condition'] = condition
                            problem_expts.loc[i, 'stimuli'] = stimulus
                            problem_expts.loc[i, 'session_subset'] = session_subset
                            problem_expts.loc[i, 'use_events'] = use_events
                            problem_expts.loc[i, 'filter_events'] = filter_events
                            problem_expts.loc[i, 'exception'] = e
                            i += 1

    save_metrics_generation_exceptions_log_file(problem_expts)


def save_metrics_generation_exceptions_log_file(problem_expts):
    """
    takes dataframe with ophys_experiment_id, conditions, and exceptions thrown when trying to generate metrics
    and saves to an hdf file. if the file exists, will load the file and concatenate new exceptions to it.
    """
    save_dir = os.path.join(loading.get_platform_analysis_cache_dir(), 'cell_metrics', 'exceptions')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, 'metric_generation_exceptions.h5')
    if os.path.exists(save_path):
        df = pd.read_hdf(save_path, key='df')
        df = pd.concat([df, problem_expts])
        df.to_hdf(save_path, key='df')
    else:
        problem_expts.to_hdf(save_path, key='df')


def load_metrics_generation_exceptions_log_file():
    """
    loads dataframe with ophys_experiment_id, conditions, and exceptions thrown when trying to generate metrics
    """
    save_dir = os.path.join(loading.get_platform_analysis_cache_dir(), 'cell_metrics', 'exceptions')
    save_path = os.path.join(save_dir, 'metric_generation_exceptions.h5')
    if os.path.exists(save_path):
        df = pd.read_hdf(save_path, key='df')
    else:
        print('problem loading metric_generation_exceptions.h5')
    return df


# def generate_and_save_all_metrics_tables_for_all_experiments(ophys_experiment_table):
#     """
#     creates full trace and cell (stimulus locked) metrics dataframes for all possible scenarios (dff, events, pref stim, all stim etc)
#     for all ophys_experiment_ids in the provided ophys_experiment_table
#     CAUTION: this is very slow and should only be done on the cluster via the script create_cell_metrics_table_all_experiments.py
#     :param ophys_experiment_table:
#     :return:
#     """
#
#
#     ## trace metrics ###
#     stimulus = 'full_session'
#     session_subset = 'full_session'
#     condition = 'traces'
#     for use_events in [True, False]:
#         if use_events:
#             for filter_events in [True, False]:
#                 metrics_df = pd.DataFrame()
#                 for ophys_experiment_id in ophys_experiment_table.index:
#                     try:
#                         trace_metrics = generate_trace_metrics_table(ophys_experiment_id,
#                                                                      use_events=use_events, filter_events=filter_events)
#                         metrics_df = pd.concat([metrics_df, trace_metrics])
#
#                     except Exception as e:
#                         print('metrics not generated for trace_metrics for experiment', ophys_experiment_id)
#                         print(e)
#
#                 filepath = get_metrics_df_filepath('all_experiments', condition=condition,
#                                                    stimuli=stimulus, session_subset=session_subset,
#                                                    use_events=use_events, filter_events=filter_events)
#                 if os.path.exists(filepath):
#                     os.remove(filepath)
#                     print('h5 file exists for all experiments  - overwriting')
#                 metrics_df.to_hdf(filepath, key='df')
#                 print('trace metrics saved for all experiments')
#
#     ### event locked response metrics ###
#     conditions = ['changes', 'omissions', 'images']
#     stimuli = ['all_images', 'pref_image']
#     session_subsets = ['engaged', 'disengaged', 'full_session']
#
#     metrics_df = pd.DataFrame()
#     for condition in conditions:
#         for stimulus in stimuli:
#             for session_subset in session_subsets:
#                 for use_events in [True, False]:
#                     if use_events:
#                         for filter_events in [True, False]:
#                             metrics_df = pd.DataFrame()
#                             for ophys_experiment_id in ophys_experiment_table.index:
#                                 try:  # code will not always run, such as in the case of passive sessions (no trials that are 'engaged')
#                                     cell_metrics = generate_cell_metrics_table(ophys_experiment_id,
#                                                                                           use_events=use_events,
#                                                                                           filter_events=filter_events,
#                                                                                           condition=condition,
#                                                                                           session_subset=session_subset,
#                                                                                           stimuli=stimulus)
#
#                                     metrics_df = pd.concat([metrics_df, cell_metrics])
#                                 except Exception as e:
#                                     print('metrics not generated for', condition, stimulus, session_subset, use_events,
#                                           filter_events, ophys_experiment_id)
#                                     print(e)
#
#                                 filepath = get_metrics_df_filepath('all_experiments', condition=condition,
#                                                                    stimuli=stimulus, session_subset=session_subset,
#                                                                    use_events=use_events, filter_events=filter_events)
#                                 if os.path.exists(filepath):
#                                     os.remove(filepath)
#                                     print('h5 file exists for cell metrics all experiments  - overwriting')
#                                 metrics_df.to_hdf(filepath, key='df')
#                                 print('cell metrics saved for all experiments')



def load_metrics_table_for_experiment(ophys_experiment_id, condition, stimuli, session_subset, use_events, filter_events):
    """
    Loads metrics table from file, either cell metrics (stimulus locked metrics) or full trace metrics, depending on provided conditions
    :param ophys_experiment_id: unique identifier for experiment, or 'all_experiments' to get all experiments from a single file
    :param condition: 'changes', 'omissions', 'images', or 'traces'
    :param stimuli: 'all_images', 'pref_image', or 'full_session' (for 'traces')
    :param session_subset: 'engaged', 'disengaged', or 'full_session' (for traces or cell metrics)
    :param use_events: Boolean
    :param filter_events: Boolean
    :return: metrics table
    """
    filepath = get_metrics_df_filepath(ophys_experiment_id, condition, stimuli,
                                       session_subset, use_events, filter_events)
    metrics_table = pd.read_hdf(filepath, key='df')
    return metrics_table


def load_metrics_table_for_experiments(ophys_experiment_ids, condition, stimuli, session_subset, use_events, filter_events):
    '''
    Loads a metrics table, either cell metrics (stimulus locked) or full trace metrics for multiple experiments
    ophys_experiment_ids: unique identifier for experiment or 'all_experiments' to load table from single file for all expts
    :param condition: 'changes', 'omissions', 'images', or 'traces'
    :param stimuli: 'all_images', 'pref_image', or 'full_session' (for 'traces')
    :param session_subset: 'engaged', 'disengaged', or 'full_session' (for traces or cell metrics)
    use_events (bool)
    filter_events (bool)
    '''

    i = 0
    problem_expts = pd.DataFrame()
    metrics_table = pd.DataFrame()
    if (type(ophys_experiment_ids) is str) and (ophys_experiment_ids == 'all_experiments'):
        try:
            metrics_table = load_metrics_table_for_experiment(ophys_experiment_ids, condition, stimuli, session_subset,
                                                                  use_events, filter_events)
        except BaseException:
            print('problem loading all experiments metrics table')
    else:
        for ophys_experiment_id in tqdm(ophys_experiment_ids):
            try:
                tmp = load_metrics_table_for_experiment(ophys_experiment_id, condition, stimuli, session_subset,
                                                            use_events, filter_events)
                metrics_table = pd.concat([metrics_table, tmp])
            except Exception as e:
                print('problem for experiment', ophys_experiment_id)
                problem_expts.loc[i, 'ophys_experiment_id'] = ophys_experiment_id
                problem_expts.loc[i, 'condition'] = condition
                problem_expts.loc[i, 'stimuli'] = stimuli
                problem_expts.loc[i, 'session_subset'] = session_subset
                problem_expts.loc[i, 'use_events'] = use_events
                problem_expts.loc[i, 'filter_events'] = filter_events
                problem_expts.loc[i, 'exception'] = e
                i += 1

    save_metrics_loading_exceptions_log_file(problem_expts)

    return metrics_table


def save_metrics_loading_exceptions_log_file(problem_expts):
    """
    takes dataframe with ophys_experiment_id, conditions, and exceptions thrown when trying to load metrics
    and saves to an hdf file. if the file exists, will load the file and concatenate new exceptions to it.
    """
    save_dir = os.path.join(loading.get_platform_analysis_cache_dir(), 'cell_metrics', 'exceptions')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, 'metric_loading_exceptions.h5')
    if os.path.exists(save_path):
        df = pd.read_hdf(save_path, key='df')
        df = pd.concat([df, problem_expts])
        df.to_hdf(save_path, key='df')
    else:
        problem_expts.to_hdf(save_path, key='df')


def load_metrics_loading_exceptions_log_file():
    """
    loads dataframe with ophys_experiment_id, conditions, and exceptions thrown when trying to load metrics
    """
    save_dir = os.path.join(loading.get_platform_analysis_cache_dir(), 'cell_metrics', 'exceptions')
    save_path = os.path.join(save_dir, 'metric_loading_exceptions.h5')
    if os.path.exists(save_path):
        df = pd.read_hdf(save_path, key='df')
    else:
        print('problem loading metric_loading_exceptions.h5')
    return df


def load_and_save_all_metrics_tables_for_all_experiments(ophys_experiment_table):
    """
    loads full trace and cell (stimulus locked) metrics dataframes for all possible scenarios (dff, events, pref stim, all stim etc)
    for all ophys_experiment_ids in the provided ophys_experiment_table then saves to a single file
    :param ophys_experiment_table:
    :return:
    """

    ophys_experiment_ids = ophys_experiment_table.index.values

    ### trace metrics ###
    condition = 'traces'
    stimuli = 'full_session'
    session_subset = 'full_session'

    for use_events in [True, False]:
        if use_events:
            for filter_events in [True, False]:
                try:
                    metrics_table = load_metrics_table_for_experiments(ophys_experiment_ids,
                                                                                    condition=condition,
                                                                                    stimuli=stimuli,
                                                                                    session_subset=session_subset,
                                                                                    use_events=use_events,
                                                                                    filter_events=filter_events)

                    # save
                    filepath = get_metrics_df_filepath('all_experiments', condition, stimuli,
                                                                    session_subset, use_events, filter_events)
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        print('h5 file exists for all experiments  - overwriting')
                    metrics_table.to_hdf(filepath, key='df')
                    print('trace metrics saved for all experiments')

                except Exception as e:
                    print('metrics not loaded for trace_metrics for all experiments', condition, stimuli,
                          session_subset, use_events, filter_events)
                    print(e)

    ### event locked response metrics ###
    conditions = ['changes', 'omissions', 'images']
    stimuli = ['all_images', 'pref_image']
    session_subsets = ['engaged', 'disengaged', 'full_session']

    metrics_df = pd.DataFrame()
    for condition in conditions:
        for stimulus in stimuli:
            for session_subset in session_subsets:
                for use_events in [True, False]:
                    if use_events:
                        for filter_events in [True, False]:
                            try:  # code will not always run, such as in the case of passive sessions (no trials that are 'engaged')
                                metrics_table = load_metrics_table_for_experiments(ophys_experiment_ids,
                                                                                                condition=condition,
                                                                                                stimuli=stimulus,
                                                                                                session_subset=session_subset,
                                                                                                use_events=use_events,
                                                                                                filter_events=filter_events)

                                # save
                                filepath = get_metrics_df_filepath('all_experiments', condition, stimulus,
                                                                                session_subset, use_events,
                                                                                filter_events)
                                if os.path.exists(filepath):
                                    os.remove(filepath)
                                    print('h5 file exists for all experiments  - overwriting')
                                metrics_table.to_hdf(filepath, key='df')
                                print('cell metrics saved for all experiments')

                            except Exception as e:
                                print('all_experiments metrics not loaded for', condition, stimulus, session_subset,
                                      use_events,
                                      filter_events, ophys_experiment_id)
                                print(e)



if __name__ == '__main__':

    ophys_experiment_table = loading.get_filtered_ophys_experiment_table(release_data_only=True)

    ophys_experiment_id = ophys_experiment_table.index[50]
    dataset = loading.get_ophys_dataset(ophys_experiment_id)

    condition = 'changes'
    session_subset = 'full_session'
    stimuli = 'all_images'


    # to get metrics table for a single experiment and set of conditions (does not save anything, just returns tables)
    cell_metrics_table = generate_cell_metrics_table(ophys_experiment_id, ophys_experiment_table, use_events=True, filter_events=False,
                                           condition=condition, session_subset=session_subset, stimuli=stimuli)

    trace_metrics = generate_trace_metrics_table(ophys_experiment_id, use_events=False, filter_events=False)

    metrics_table = pd.concat([metrics_table, trace_metrics])

    # to generate and save metrics tables for all possible conditions for a single experiment
    generate_and_save_all_metrics_tables_for_experiment(ophys_experiment_id)

    # to generate and save metrics tables for all experiments, all conditions
