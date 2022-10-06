import os
import numpy as np
import pandas as pd
import seaborn as sns

import logging

logger = logging.getLogger(__name__)


def filter_by_responsiveness(data, use_events=False):
    df = data[data.pref_stim == True].copy()
    if use_events:
        filtered_df = df[(df.mean_response > 0.005) & (df.fraction_nonzero_trials > 0.25)]
    else:
        filtered_df = df[(df.mean_response > 0.05) & (df.fraction_significant_trials > 0.25)]
    return filtered_df


# def get_nearest_frame(timepoint, timestamps):
#     return int(np.nanargmin(abs(timestamps - timepoint)))

# modified 181212, again 190127
def get_nearest_frame(timepoint, timestamps):
    nearest_frame = int(np.nanargmin(abs(timestamps - timepoint)))
    nearest_timepoint = timestamps[nearest_frame]
    if nearest_timepoint < timepoint:  # nearest frame time (2P) must be greater than provided timepoint (stim)
        # nearest_frame = nearest_frame - 1  # 181212
        nearest_frame = nearest_frame + 1  # use next frame to ensure nearest (2P) follows input timepoint (stim), 190127
    return nearest_frame


def get_successive_frame_list(timepoints_array, timestanps):
    # This is a modification of get_nearest_frame for speedup
    #  This implementation looks for the first 2p frame consecutive to the stim
    successive_frames = np.searchsorted(timestanps, timepoints_array)

    return successive_frames


def get_trace_around_timepoint(timepoint, trace, timestamps, window, frame_rate):
    #   frame_for_timepoint = get_nearest_frame(timepoint, timestamps)
    frame_for_timepoint = get_successive_frame_list(timepoint, timestamps)
    lower_frame = frame_for_timepoint + (window[0] * frame_rate)
    upper_frame = frame_for_timepoint + (window[1] * frame_rate)
    trace = trace[int(lower_frame):int(upper_frame)]
    timepoints = timestamps[int(lower_frame):int(upper_frame)]
    return trace, timepoints


def get_responses_around_event_times(trace, timestamps, event_times, frame_rate, window=[-2, 3]):
    responses = []
    for event_time in event_times:
        response, times = get_trace_around_timepoint(event_time, trace, timestamps,
                                                     frame_rate=frame_rate, window=window)
        responses.append(response)
    responses = np.asarray(responses)
    return responses


def get_mean_in_window(trace, window, frame_rate, use_events=False):
    mean = np.nanmean(trace[int(window[0] * frame_rate): int(window[1] * frame_rate)])
    return mean


def get_sd_in_window(trace, window, frame_rate):
    std = np.std(
        trace[int(window[0] * frame_rate): int(window[1] * frame_rate)])
    return std


def get_n_nonzero_in_window(trace, window, frame_rate):
    datapoints = trace[int(np.round(window[0] * frame_rate)): int(np.round(window[1] * frame_rate))]
    n_nonzero = len(np.where(datapoints > 0)[0])
    return n_nonzero


def get_sd_over_baseline(trace, response_window, baseline_window, frame_rate):
    baseline_std = get_sd_in_window(trace, baseline_window, frame_rate)
    response_mean = get_mean_in_window(trace, response_window, frame_rate)
    return response_mean / (baseline_std)


def get_p_val(trace, response_window, frame_rate):
    from scipy import stats
    response_window_duration = response_window[1] - response_window[0]
    baseline_end = int(response_window[0] * frame_rate)
    baseline_start = int((response_window[0] - response_window_duration) * frame_rate)
    stim_start = int(response_window[0] * frame_rate)
    stim_end = int((response_window[0] + response_window_duration) * frame_rate)
    (_, p) = stats.f_oneway(trace[baseline_start:baseline_end], trace[stim_start:stim_end])
    return p


def ptest(x, num_conditions):
    ptest = len(np.where(x < (0.05 / num_conditions))[0])
    return ptest


# changed variable names and used longer response window
def get_p_values_from_shuffle_synthetic(analysis, stimulus_table, flash_response_df):
    # get data
    dataset = analysis.dataset
    fdf = flash_response_df.copy()
    stim_table = stimulus_table.copy()
    included_flashes = fdf.flash_number.unique()
    stim_table = stim_table[stim_table.flash_number.isin(included_flashes)]
    omitted_flashes = stim_table[stim_table.omitted == True]
    omitted_flashes['start_frame'] = [get_nearest_frame(start_time, dataset.timestamps_ophys) for start_time in
                                      omitted_flashes.start_time.values]
    # set params
    n_shuffles = 10000
    response_window = analysis.response_window_duration
    frame_rate = analysis.ophys_frame_rate
    n_mean_window_frames = int(np.round(response_window * frame_rate, 0))  # 500ms response window * 31Hz = 16 frames
    cell_indices = dataset.get_cell_indices()
    n_cells = len(cell_indices)
    # get omitted flash frames across full response window
    omitted_start_times = omitted_flashes.start_frame.values
    omitted_start_times = omitted_start_times[:-1]  # exclude last omission for cases where it occured at the end of the recording
    # shuffle omitted flash frames
    shuffled_omitted_start_frames = np.random.choice(omitted_start_times, n_shuffles)
    # create synthetic traces by drawing from shuffled omitted frames
    shuffled_omitted_responses = np.empty((n_cells, n_shuffles, n_mean_window_frames))
    for mean_window_frame in range(n_mean_window_frames):
        shuffled_omitted_responses[:, :, mean_window_frame] = dataset.dff_traces[:, shuffled_omitted_start_frames + mean_window_frame]
    # average across mean response window
    shuffled_means = shuffled_omitted_responses.mean(axis=2)
    # compare flash responses to shuffled mean response values and make a dataframe of p_value for cell by sweep
    flash_p_values = pd.DataFrame(index=stim_table.index.values, columns=np.array(range(n_cells)).astype(str))
    for i, cell_index in enumerate(cell_indices):
        flash_responses = fdf[fdf.cell == cell_index].mean_response.values
        null_distribution_matrix = np.tile(shuffled_means[i, :], reps=(len(flash_responses), 1))
        # compute the number of times out of the 10000 shuffles where the flash response is less than the null
        flash_response_is_less = flash_responses.reshape(len(flash_responses), 1) <= null_distribution_matrix
        p_values = np.mean(flash_response_is_less, axis=1)
        flash_p_values[str(cell_index)] = p_values
    return flash_p_values


# modified code to compute null distribution from shuffled mean responses rather than creating a synthetic trace
def get_p_values_from_shuffle(analysis, flash_response_df):
    # test similar to: https: // genomicsclass.github.io / book / pages / permutation_tests.html
    # get data
    dataset = analysis.dataset
    fdf = flash_response_df.copy()
    odf = fdf[fdf.omitted == True].copy()
    # set params
    n_shuffles = 10000
    cell_indices = dataset.get_cell_indices()
    # create p-values by comparing flash responses to shuffled omitted responses
    flash_p_values = pd.DataFrame(index=fdf.flash_number.unique(), columns=cell_indices.astype(str))
    for i, cell_index in enumerate(cell_indices):
        cell_omitted_responses = odf[odf.cell == cell_index].mean_response.values
        # shuffle omitted flash responses
        shuffled_omitted_responses = np.random.choice(cell_omitted_responses, n_shuffles)
        # compare flash responses to shuffled mean response values
        flash_responses = fdf[fdf.cell == cell_index].mean_response.values
        # create null distribution of shuffled values to compare each flash to
        null_distribution_matrix = np.tile(shuffled_omitted_responses, reps=(len(flash_responses), 1))
        # compare all flash responses for this cell to the null distribution
        flash_response_is_less = flash_responses.reshape(len(flash_responses), 1) <= null_distribution_matrix
        # p_value is the average number of times where the flash response is less than the null
        p_values = np.mean(flash_response_is_less, axis=1)
        flash_p_values[str(cell_index)] = p_values
    return flash_p_values


def get_p_values_from_shuffle_omitted(analysis, stimulus_table, omitted_flash_response_df):
    dataset = analysis.dataset
    # data munging
    odf = omitted_flash_response_df.copy()
    ost = stimulus_table.copy()
    included_flashes = odf.flash_number.unique()
    ost = ost[ost.flash_number.isin(included_flashes)]
    stimulus_flashes = dataset.stimulus_table[dataset.stimulus_table.omitted == False]
    stimulus_flashes['start_frame'] = [get_nearest_frame(start_time, dataset.timestamps_ophys) for start_time in
                                       stimulus_flashes.start_time.values]
    stimulus_flashes['end_frame'] = [get_nearest_frame(end_time, dataset.timestamps_ophys) for end_time in
                                     stimulus_flashes.end_time.values]
    # set params
    response_window = analysis.response_window_duration
    frame_rate = analysis.ophys_frame_rate
    stim_frames = int(np.round(response_window * frame_rate, 0))  # stimulus window = 0.25ms*31Hz = 7.75 frames
    cell_indices = dataset.get_cell_indices()
    n_cells = len(cell_indices)
    # get shuffled values from stimulus flashes
    shuffled_stimulus_responses = np.empty((n_cells, 10000, stim_frames))
    stimulus_idx = np.random.choice(stimulus_flashes.start_frame.values, 10000)  # get random stimulus start times
    for i in range(stim_frames):  # make a fake trace of len(stim_duration) to take a shuffled mean
        shuffled_stimulus_responses[:, :, i] = dataset.dff_traces[:, stimulus_idx + i]
    shuffled_mean = shuffled_stimulus_responses.mean(axis=2)  # take mean of shuffled trace
    # compare omitted responses to shuffled stimulus values and make a dataframe of p_value for cell by sweep
    omitted_flash_p_values = pd.DataFrame(index=ost.index.values, columns=np.array(range(n_cells)).astype(str))
    for i, cell_index in enumerate(cell_indices):
        responses = odf[odf.cell == cell_index].mean_response.values
        null_dist_mat = np.tile(shuffled_mean[i, :], reps=(len(responses), 1))
        actual_is_less = responses.reshape(len(responses), 1) <= null_dist_mat
        p_values = np.mean(actual_is_less, axis=1)
        omitted_flash_p_values[str(cell_index)] = p_values
    return omitted_flash_p_values


def get_spontaneous_frames(dataset):
    st = dataset.stimulus_table.copy()

    # dont use full 5 mins to avoid fingerprint and countdown
    # spont_duration_frames = 4 * 60 * 60  # 4 mins * * 60s/min * 60Hz
    spont_duration = 4 * 60  # 4mins * 60sec

    # for spontaneous at beginning of session
    behavior_start_time = st.iloc[0].start_time
    spontaneous_start_time_pre = behavior_start_time - spont_duration
    spontaneous_end_time_pre = behavior_start_time
    spontaneous_start_frame_pre = get_successive_frame_list(spontaneous_start_time_pre, dataset.timestamps_ophys)
    spontaneous_end_frame_pre = get_successive_frame_list(spontaneous_end_time_pre, dataset.timestamps_ophys)
    spontaneous_frames_pre = np.arange(spontaneous_start_frame_pre, spontaneous_end_frame_pre, 1)

    # for spontaneous epoch at end of session
    behavior_end_time = st.iloc[-1].end_time
    spontaneous_start_time_post = behavior_end_time
    spontaneous_end_time_post = behavior_end_time + spont_duration
    spontaneous_start_frame_post = get_successive_frame_list(spontaneous_start_time_post, dataset.timestamps_ophys)
    spontaneous_end_frame_post = get_successive_frame_list(spontaneous_end_time_post, dataset.timestamps_ophys)
    spontaneous_frames_post = np.arange(spontaneous_start_frame_post, spontaneous_end_frame_post, 1)

    # add them together
    spontaneous_frames = list(spontaneous_frames_pre) + (list(spontaneous_frames_post))
    return spontaneous_frames


def get_p_values_from_shuffle_spontaneous(dataset, flash_response_df, response_window_duration=0.5):
    # data munging
    fdf = flash_response_df.copy()
    st = dataset.stimulus_table.copy()
    included_flashes = fdf.flash_number.unique()
    st = st[st.flash_number.isin(included_flashes)]

    #  params
    ophys_frame_rate = 31
    n_mean_response_window_frames = int(
        np.round(response_window_duration * ophys_frame_rate, 0))  # stimulus window = 0.25ms*31Hz = 7.75 frames

    spontaneous_frames = np.array(get_spontaneous_frames(dataset))
    spontaneous_frames = spontaneous_frames[spontaneous_frames < max(
        spontaneous_frames) - n_mean_response_window_frames]  # avoid overruning the end of the vector

    cell_indices = dataset.get_cell_indices()
    n_cells = len(cell_indices)
    # get shuffled values from spontaneous periods
    shuffled_responses = np.empty((n_cells, 10000, n_mean_response_window_frames))
    idx = np.random.choice(spontaneous_frames, 10000)
    # get mean response for 10000 shuffles of the spontaneous activity frames
    # in a window the same size as the stim response window duration
    for i in range(n_mean_response_window_frames):
        shuffled_responses[:, :, i] = dataset.dff_traces[:, idx + i]
    shuffled_mean = shuffled_responses.mean(axis=2)
    # compare flash responses to shuffled values and make a dataframe of p_value for cell by sweep
    flash_p_values = pd.DataFrame(index=st.index.values, columns=np.array(range(n_cells)).astype(str))
    for i, cell_index in enumerate(cell_indices):
        responses = fdf[fdf.cell == cell_index].mean_response.values
        null_dist_mat = np.tile(shuffled_mean[i, :], reps=(len(responses), 1))
        actual_is_less = responses.reshape(len(responses), 1) <= null_dist_mat
        p_values = np.mean(actual_is_less, axis=1)
        flash_p_values[str(cell_index)] = p_values
    return flash_p_values


def get_mean_sem_trace(group):
    mean_response = np.mean(group['mean_response'])
    mean_baseline = np.mean(group['baseline_response'])
    mean_responses = group['mean_response'].values
    sem_response = np.std(group['mean_response'].values) / np.sqrt(len(group['mean_response'].values))
    sem_baseline = np.std(group['baseline_response'].values) / np.sqrt(len(group['baseline_response'].values))
    mean_trace = np.mean(group['trace'], axis=0)
    sem_trace = np.std(group['trace'].values) / np.sqrt(len(group['trace'].values))
    trace_timestamps = np.mean(group['trace_timestamps'], axis=0)
    return pd.Series({'mean_response': mean_response, 'sem_response': sem_response,
                      'mean_baseline': mean_baseline, 'sem_baseline': sem_baseline,
                      'mean_trace': mean_trace, 'sem_trace': sem_trace,
                      'trace_timestamps': trace_timestamps,
                      'mean_responses': mean_responses})


def get_mean_sem(group):
    mean_response = np.mean(group['mean_response'])
    sem_response = np.std(group['mean_response'].values) / np.sqrt(len(group['mean_response'].values))
    return pd.Series({'mean_response': mean_response, 'sem_response': sem_response})


def get_fraction_significant_trials(group):
    fraction_significant_trials = len(group[group.p_value < 0.05]) / float(len(group))
    return pd.Series({'fraction_significant_trials': fraction_significant_trials})


def get_fraction_significant_p_value_gray_screen(group):
    fraction_significant_p_value_gray_screen = len(group[group.p_value_gray_screen < 0.05]) / float(len(group))
    return pd.Series({'fraction_significant_p_value_gray_screen': fraction_significant_p_value_gray_screen})


def get_fraction_significant_p_value_omission(group):
    fraction_significant_p_value_omission = len(group[group.p_value_omission < 0.05]) / float(len(group))
    return pd.Series({'fraction_significant_p_value_omission': fraction_significant_p_value_omission})


def get_fraction_significant_p_value_stimulus(group):
    fraction_significant_p_value_stimulus = len(group[group.p_value_stimulus < 0.05]) / float(len(group))
    return pd.Series({'fraction_significant_p_value_stimulus': fraction_significant_p_value_stimulus})


def get_fraction_active_trials(group):
    fraction_active_trials = len(group[group.mean_response > 0.05]) / float(len(group))
    return pd.Series({'fraction_active_trials': fraction_active_trials})


def get_fraction_responsive_trials(group):
    fraction_responsive_trials = len(group[(group.p_value_baseline < 0.05)]) / float(len(group))
    return pd.Series({'fraction_responsive_trials': fraction_responsive_trials})


def get_fraction_nonzero_trials(group):
    fraction_nonzero_trials = len(group[group.n_events > 0]) / float(len(group))
    return pd.Series({'fraction_nonzero_trials': fraction_nonzero_trials})


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


def compute_reliability(group, window=[-3, 3], response_window_duration=0.5, frame_rate=30.):
    # computes trial to trial correlation across input traces in group,
    # only for portion of the trace after the change time or flash onset time

    onset = int(np.abs(window[0]) * frame_rate)
    response_window = [onset, onset + (int(response_window_duration * frame_rate))]
    traces = group['trace'].values
    traces = np.vstack(traces)
    if traces.shape[0] > 5:
        traces = traces[:, response_window[0]:response_window[1]]  # limit to response window
        reliability, correlation_values = compute_reliability_vectorized(traces)
    else:
        reliability = np.nan
        correlation_values = []
    return pd.Series({'reliability': reliability, 'correlation_values': correlation_values})


def get_window(analysis=None, flashes=False, omitted=False):
    if analysis and omitted:
        window = analysis.omitted_flash_window
    elif analysis and flashes and not omitted:
        window = analysis.flash_window
    elif analysis and not flashes and not omitted:
        window = analysis.trial_window
    elif not analysis and flashes and not omitted:
        window = [-0.5, 0.75]
    elif not analysis and omitted:
        window = [-3, 3]
    else:
        window = [-4, 8]
    return window


def get_mean_df(response_df, conditions=['cell', 'change_image_name'], frame_rate=30.,
                time_window=[-3, 3.1], response_window_duration=0.5,
                get_pref_stim=True, exclude_omitted_from_pref_stim=True):
    """

    Takes stimulus_response_df as input (dataframe with trace for each cell for each stimulus presentation in a session,
        and averages over the conditions provided to produce an average timeseries, mean response value,
        and computes other metrics across the trials for each condition (such as trial to trial reliability).

    :param response_df: stimulus_response_df from mindscope_utilities.visual_behavior_ophys.data_formatting.get_stimulus_response_df()
    :param conditions: columns in response_df to groupby before averaging. Columns must be Bool or categorical.
                            ex: ['cell_specimen_id', 'image_name', 'change'] gives the average response for each cell, for each image name for changes and non-changes
    :param frame_rate: Frame rate of timeseries. Either the native frame rate of the physio, or if interpolation was used in
                            get_stimulus_response_df(), provide the output_sampling_rate used for interpolation
    :param time_window: window of time around each stimulus_presentation_id that was used to extract stimulus aligned traces to create stimulus_response_df
    :param response_window_duration: duration of time, in seconds, following the start_time for each stimulus_presentations_id,
                            over which to average to get a mean response for each stimulus
    :param get_pref_stim: Boolean, whether or not to annotate dataframe with a 'pref_stim' column,
                            which is also a Boolean indicating which image gave the maximal response for each condition for each cell.
                            Only works if one of the conditions is 'image_name', 'change_image_name' or 'prior_image_name'
    :param exclude_omitted_from_pref_stim: Boolean, whether or not to exclude image_name='omitted' from calculation of preferred stimulus
                            If False, stimulus omissions can be the 'pref_stim' for a given cell

    :return: response df averaged over the provided conditions, including the average timeseries, the mean response given by response_window_duration,
                            and several other metrics characterizing trial to trial variability (i.e. fano factor, reliability, etc)
    """

    import visual_behavior.ophys.response_analysis.response_processing as rp

    window = time_window.copy()

    rdf = response_df.copy()

    mdf = rdf.groupby(conditions).apply(get_mean_sem_trace)
    mdf = mdf[['mean_response', 'sem_response', 'mean_trace', 'sem_trace', 'trace_timestamps', 'mean_responses', 'mean_baseline', 'sem_baseline']]
    mdf = mdf.reset_index()
    # save response window duration as a column for reference
    mdf['response_window_duration'] = response_window_duration

    if get_pref_stim:
        if ('image_name' in conditions) or ('change_image_name' in conditions) or ('prior_image_name' in conditions):
            mdf = annotate_mean_df_with_pref_stim(mdf, exclude_omitted_from_pref_stim)

    try:
        mdf = annotate_mean_df_with_fano_factor(mdf)
        mdf = annotate_mean_df_with_time_to_peak(mdf, time_window, frame_rate)
        mdf = annotate_mean_df_with_p_value(mdf, window, response_window_duration, frame_rate)
        mdf = annotate_mean_df_with_sd_over_baseline(mdf, time_window, response_window_duration, frame_rate)
    except Exception as e:  # NOQA E722
        print(e)
        pass

    if 'p_value_gray_screen' in rdf.keys():
        fraction_significant_p_value_gray_screen = rdf.groupby(conditions).apply(
            get_fraction_significant_p_value_gray_screen)
        fraction_significant_p_value_gray_screen = fraction_significant_p_value_gray_screen.reset_index()
        mdf['fraction_significant_p_value_gray_screen'] = fraction_significant_p_value_gray_screen.fraction_significant_p_value_gray_screen

    try:
        reliability = rdf.groupby(conditions).apply(compute_reliability, time_window, response_window_duration, frame_rate)
        reliability = reliability.reset_index()
        mdf['reliability'] = reliability.reliability
        mdf['correlation_values'] = reliability.correlation_values
        # print('done computing reliability')
    except Exception as e:
        print('failed to compute reliability')
        print(e)
        pass

    if 'index' in mdf.keys():
        mdf = mdf.drop(columns=['index'])
    return mdf


def get_cre_lines(mean_df):
    cre_lines = np.sort(mean_df.cre_line.unique())
    return cre_lines


def get_cell_classes(mean_df):
    cell_classes = np.sort(mean_df.cell_class.unique())
    return cell_classes


def get_colors_for_cre_lines():
    colors = [sns.color_palette()[2], sns.color_palette()[4]]
    return colors


def get_image_sets(mean_df):
    image_sets = np.sort(mean_df.image_set.unique())
    return image_sets


def get_visual_areas(mean_df):
    areas = np.sort(mean_df.area.unique())[::-1]
    return areas


def get_image_names(mean_df):
    if 'change_image_name' in mean_df.keys():
        image_names = np.sort(mean_df.change_image_name.unique())
    else:
        image_names = np.sort(mean_df.image_name.unique())
        # image_names = image_names[image_names!='omitted']
    return image_names


def get_color_for_image_name(image_names, image_name):
    image_names = np.sort(image_names)
    if 'omitted' in image_names:
        if image_name == 'omitted':
            color = 'gray'
        else:
            colors = sns.color_palette("hls", len(image_names) - 1)
            image_index = np.where(image_names == image_name)[0][0]
            color = colors[image_index]
    else:
        colors = sns.color_palette("hls", len(image_names))
        image_index = np.where(image_names == image_name)[0][0]
        color = colors[image_index]
    return color


def get_color_for_area(area):
    colors = sns.color_palette()
    if area == 'VISp':
        color = colors[4]
    elif area == 'VISal':
        color = colors[9]
    return color


def get_colors_for_areas(df):
    if 'area' in df:
        area = 'area'
    else:
        area = 'targeted_structure'
    colors = []
    for area in np.sort(df[area].unique()):
        colors.append(get_color_for_area(area))
    return colors


def get_colors_for_image_sets():
    colors = [sns.color_palette()[3]] + sns.color_palette('Blues_r', 5)[:3]
    # colors = sns.color_palette()
    # colors = [colors[3],colors[0],colors[2],colors[4]]
    return colors


def get_colors_for_repeats():
    greens = sns.light_palette("green", 5)[::-1]
    purple = sns.dark_palette("purple")[-1]
    colors = [list(purple), list(greens[0]), list(greens[1]), list(greens[2])]
    return colors


def get_colors_for_changes():
    colors = [sns.color_palette()[0],
              sns.color_palette()[3]]
    return colors


def get_colors_for_trained_untrained():
    colors = [sns.color_palette()[0],
              sns.color_palette()[3]]
    return colors


def get_colors_for_behavioral_response_types():
    colors = sns.color_palette()
    colors = [colors[2], colors[8], colors[0], colors[3]]
    return colors


def add_metadata_to_mean_df(mdf, metadata):
    import visual_behavior.data_access.reformat as reformat
    metadata = reformat.convert_metadata_to_dataframe(metadata)
    metadata = metadata.drop(columns=['excitation_lambda', 'emission_lambda', 'indicator',
                                      'field_of_view_width', 'field_of_view_height'])
    metadata['image_set'] = metadata['session_type'].values[0][-1]
    metadata['session_number'] = metadata['session_type'].values[0][6]
    mdf = mdf.merge(metadata, how='outer', on='ophys_experiment_id')
    return mdf


def get_time_to_peak(trace, window=[-4, 8], frame_rate=30.):
    response_window_duration = 0.75
    response_window = [np.abs(window[0]), np.abs(window[0]) + response_window_duration]
    response_window_trace = trace[int(response_window[0] * frame_rate):(int(response_window[1] * frame_rate))]
    peak_response = np.amax(response_window_trace)
    peak_frames_from_response_window_start = np.where(response_window_trace == np.amax(response_window_trace))[0][0]
    time_to_peak = peak_frames_from_response_window_start / float(frame_rate)
    return peak_response, time_to_peak


def annotate_mean_df_with_time_to_peak(mean_df, window=[-4, 8], frame_rate=30.):
    ttp_list = []
    peak_list = []
    for idx in mean_df.index:
        mean_trace = mean_df.iloc[idx].mean_trace
        peak_response, time_to_peak = get_time_to_peak(mean_trace, window=window, frame_rate=frame_rate)
        ttp_list.append(time_to_peak)
        peak_list.append(peak_response)
    mean_df['peak_response'] = peak_list
    mean_df['time_to_peak'] = ttp_list
    return mean_df


def annotate_mean_df_with_fano_factor(mean_df):
    ff_list = []
    for idx in mean_df.index:
        mean_responses = mean_df.iloc[idx].mean_responses
        sd = np.nanstd(mean_responses)
        mean_response = np.nanmean(mean_responses)
        fano_factor = np.abs((sd * 2) / mean_response)  # take abs value to account for negative mean_response
        ff_list.append(fano_factor)
    mean_df['fano_factor'] = ff_list
    return mean_df


def annotate_mean_df_with_p_value(mean_df, window=[-4, 8], response_window_duration=0.5, frame_rate=30.):
    response_window = [np.abs(window[0]), np.abs(window[0]) + response_window_duration]
    p_val_list = []
    for idx in mean_df.index:
        mean_trace = mean_df.iloc[idx].mean_trace
        p_value = get_p_val(mean_trace, response_window, frame_rate)
        p_val_list.append(p_value)
    mean_df['p_value'] = p_val_list
    return mean_df


def annotate_mean_df_with_sd_over_baseline(mean_df, window=[-4, 8], response_window_duration=0.5, frame_rate=30.):
    response_window = [np.abs(window[0]), np.abs(window[0]) + response_window_duration]
    baseline_window = [np.abs(window[0]) - response_window_duration, (np.abs(window[0]))]
    sd_list = []
    for idx in mean_df.index:
        mean_trace = mean_df.iloc[idx].mean_trace
        sd = get_sd_over_baseline(mean_trace, response_window, baseline_window, frame_rate)
        sd_list.append(sd)
    mean_df['sd_over_baseline'] = sd_list
    return mean_df


def annotate_mean_df_with_pref_stim(mean_df, exclude_omitted_from_pref_stim=True):
    if 'prior_image_name' in mean_df.keys():
        image_name = 'prior_image_name'
    elif 'image_name' in mean_df.keys():
        image_name = 'image_name'
    else:
        image_name = 'change_image_name'
    mdf = mean_df.reset_index()
    mdf['pref_stim'] = False
    if 'cell_specimen_id' in mdf.keys():
        cell_key = 'cell_specimen_id'
    else:
        cell_key = 'cell'
    for cell in mdf[cell_key].unique():
        mc = mdf[(mdf[cell_key] == cell)]
        if exclude_omitted_from_pref_stim:
            if 'omitted' in mdf[image_name].unique():
                mc = mc[mc[image_name] != 'omitted']
        pref_image = mc[(mc.mean_response == np.max(mc.mean_response.values))][image_name].values[0]
        row = mdf[(mdf[cell_key] == cell) & (mdf[image_name] == pref_image)].index
        mdf.loc[row, 'pref_stim'] = True
    return mdf


def annotate_trial_response_df_with_pref_stim(trial_response_df):
    rdf = trial_response_df.copy()
    rdf['pref_stim'] = False
    if 'cell_specimen_id' in rdf.keys():
        cell_key = 'cell_specimen_id'
    else:
        cell_key = 'cell'
    mean_response = rdf.groupby([cell_key, 'change_image_name']).apply(get_mean_sem_trace)
    m = mean_response.unstack()
    for cell in m.index:
        image_index = np.where(m.loc[cell]['mean_response'].values == np.max(m.loc[cell]['mean_response'].values))[0][0]
        pref_image = m.loc[cell]['mean_response'].index[image_index]
        trials = rdf[(rdf[cell_key] == cell) & (rdf.change_image_name == pref_image)].index
        for trial in trials:
            rdf.loc[trial, 'pref_stim'] = True
    return rdf


def annotate_flash_response_df_with_pref_stim(fdf):
    fdf = fdf.reset_index()
    if 'cell_specimen_id' in fdf.keys():
        cell_key = 'cell_specimen_id'
    else:
        cell_key = 'cell'
    fdf['pref_stim'] = False
    mean_response = fdf.groupby([cell_key, 'image_name']).apply(get_mean_sem)
    m = mean_response.unstack()
    for cell in m.index:
        image_index = \
            np.where(m.loc[cell]['mean_response'].values == np.nanmax(m.loc[cell]['mean_response'].values))[0][0]
        pref_image = m.loc[cell]['mean_response'].index[image_index]
        trials = fdf[(fdf[cell_key] == cell) & (fdf.image_name == pref_image)].index
        for trial in trials:
            fdf.loc[trial, 'pref_stim'] = True
    return fdf


def annotate_flashes_with_reward_rate(dataset):
    last_time = 0
    reward_rate_by_frame = []
    trials = dataset.trials[dataset.trials.trial_type != 'aborted']
    flashes = dataset.stimulus_table.copy()
    for change_time in trials.change_time.values:
        reward_rate = trials[trials.change_time == change_time].reward_rate.values[0]
        for start_time in flashes.start_time:
            if (start_time < change_time) and (start_time > last_time):
                reward_rate_by_frame.append(reward_rate)
                last_time = start_time
    # fill the last flashes with last value
    for i in range(len(flashes) - len(reward_rate_by_frame)):
        reward_rate_by_frame.append(reward_rate_by_frame[-1])
    flashes['reward_rate'] = reward_rate_by_frame
    return flashes


def get_gray_response_df(dataset, window=0.5):
    window = 0.5
    row = []
    flashes = dataset.stimulus_table.copy()
    # stim_duration = dataset.task_parameters.stimulus_duration.values[0]
    for cell in range(dataset.dff_traces.shape[0]):
        for x, gray_start_time in enumerate(
                flashes.end_time[:-5]):  # exclude the last 5 frames to prevent truncation of traces
            ophys_start_frame = int(np.nanargmin(abs(dataset.timestamps_ophys - gray_start_time)))
            ophys_end_time = gray_start_time + int(dataset.metadata.ophys_frame_rate.values[0] * 0.5)
            gray_end_time = gray_start_time + window
            ophys_end_frame = int(np.nanargmin(abs(dataset.timestamps_ophys - ophys_end_time)))
            mean_response = np.mean(dataset.dff_traces[cell][ophys_start_frame:ophys_end_frame])
            row.append([cell, x, gray_start_time, gray_end_time, mean_response])
    gray_response_df = pd.DataFrame(data=row, columns=['cell', 'gray_number', 'gray_start_time', 'gray_end_time',
                                                       'mean_response'])
    return gray_response_df


def add_repeat_to_stimulus_table(stimulus_table):
    repeat = []
    post_omitted = []
    n = 0
    # for i, image in enumerate(stimulus_table.image_name.values):
    #     if image != stimulus_table.image_name.values[i-1]:
    #         n = 1
    #         repeat.append(n)
    #     else:
    #         n += 1
    #         repeat.append(n)
    for i, current_image in enumerate(stimulus_table.image_name.values):
        if current_image == 'omitted':
            repeat.append(1)
            post_omitted.append(False)
        else:
            last_image = stimulus_table.image_name.values[i - 1]
            if last_image == 'omitted':
                n += 1
                repeat.append(n)
                post_omitted.append(True)
            else:
                if current_image != last_image:
                    n = 1
                    repeat.append(n)
                    post_omitted.append(False)
                else:
                    n += 1
                    repeat.append(n)
                    post_omitted.append(False)
    stimulus_table['repeat'] = repeat
    stimulus_table['repeat'] = [int(r) for r in stimulus_table.repeat.values]
    stimulus_table['post_omitted'] = post_omitted
    return stimulus_table


def add_repeat_number_to_flash_response_df(flash_response_df, stimulus_table):
    stimulus_table = add_repeat_to_stimulus_table(stimulus_table)
    flash_response_df = flash_response_df.merge(stimulus_table[['flash_number', 'repeat', 'post_omitted']],
                                                on='flash_number')
    return flash_response_df


def add_image_block_to_stimulus_table(stimulus_table):
    stimulus_table['image_block'] = np.nan
    for image_name in stimulus_table.image_name.unique():
        block = 0
        for index in stimulus_table[stimulus_table.image_name == image_name].index.values:
            if stimulus_table.iloc[index]['repeat'] == 1:
                block += 1
            stimulus_table.loc[index, 'image_block'] = int(block)
    stimulus_table['image_block'] = [int(image_block) for image_block in stimulus_table.image_block.values]
    return stimulus_table


def add_image_block_to_flash_response_df(flash_response_df, stimulus_table):
    stimulus_table = add_image_block_to_stimulus_table(stimulus_table)
    flash_response_df = flash_response_df.merge(stimulus_table[['flash_number', 'image_block']], on='flash_number')
    return flash_response_df


def annotate_flash_response_df_with_block_set(flash_response_df):
    fdf = flash_response_df.copy()
    fdf['block_set'] = np.nan
    block_sets = np.arange(0, np.amax(fdf.image_block.unique()), 10)
    for i, block_set in enumerate(block_sets):
        if block_set != np.amax(block_sets):
            indices = fdf[(fdf.image_block >= block_sets[i]) & (fdf.image_block < block_sets[i + 1])].index.values
        else:
            indices = fdf[(fdf.image_block >= block_sets[i])].index.values
        for index in indices:
            fdf.loc[index, 'block_set'] = i
    return fdf


def add_early_late_block_ratio_for_fdf(fdf, repeat=1, pref_stim=True):
    data = fdf[(fdf.repeat == repeat) & (fdf.pref_stim == pref_stim)]

    data['early_late_block_ratio'] = np.nan
    for cell in data.cell.unique():
        first_blocks = data[(data.cell == cell) & (data.block_set.isin([0, 1]))].mean_response.mean()
        last_blocks = data[(data.cell == cell) & (data.block_set.isin([2, 3]))].mean_response.mean()
        index = (last_blocks - first_blocks) / (last_blocks + first_blocks)
        ratio = first_blocks / last_blocks
        indices = data[data.cell == cell].index
        data.loc[indices, 'early_late_block_index'] = index
        data.loc[indices, 'early_late_block_ratio'] = ratio
    return data


def add_ophys_times_to_behavior_df(behavior_df, timestamps_ophys):
    """
    behavior_df can be dataset.running, dataset.licks or dataset.rewards
    """
    # ophys_frames = [get_nearest_frame(timepoint, timestamps_ophys) for timepoint in behavior_df.time.values]
    ophys_frames = get_successive_frame_list(behavior_df.time.values, timestamps_ophys)

    ophys_times = [timestamps_ophys[frame] for frame in ophys_frames]
    behavior_df['ophys_frame'] = ophys_frames
    behavior_df['ophys_time'] = ophys_times
    return behavior_df


def add_ophys_times_to_stimulus_table(stimulus_table, timestamps_ophys):
    # ophys_start_frames = [get_nearest_frame(timepoint, timestamps_ophys) for timepoint in
    #                      stimulus_table.start_time.values]
    ophys_start_frames = get_successive_frame_list(stimulus_table.start_time.values, timestamps_ophys)

    ophys_start_times = [timestamps_ophys[frame] for frame in ophys_start_frames]
    stimulus_table['ophys_start_frame'] = ophys_start_frames
    stimulus_table['ophys_start_time'] = ophys_start_times

    ophys_end_frames = get_successive_frame_list(stimulus_table.end_time.values, timestamps_ophys)
    #    ophys_end_frames = [get_nearest_frame(timepoint, timestamps_ophys) for timepoint in stimulus_table.end_time.values]
    ophys_end_times = [timestamps_ophys[frame] for frame in ophys_end_frames]
    stimulus_table['ophys_end_frame'] = ophys_end_frames
    stimulus_table['ophys_end_time'] = ophys_end_times
    return stimulus_table


def get_running_speed_ophys_time(running_speed, timestamps_ophys):
    """
    running_speed dataframe must have column 'ophys_times'
    """
    if 'ophys_time' not in running_speed.keys():
        logger.info('ophys_times not in running_speed dataframe')
    running_speed_ophys_time = np.empty(timestamps_ophys.shape)
    for i, ophys_time in enumerate(timestamps_ophys):
        run_df = running_speed[running_speed.ophys_time == ophys_time]
        if len(run_df) > 0:
            run_speed = run_df.running_speed.mean()
        else:
            run_speed = np.nan
        running_speed_ophys_time[i] = run_speed
    return running_speed_ophys_time


def get_binary_mask_for_behavior_events(behavior_df, timestamps_ophys):
    """
    behavior_df must have column 'ophys_times' from add_ophys_times_to_behavior_df
    """
    binary_mask = [1 if time in behavior_df.ophys_time.values else 0 for time in timestamps_ophys]
    binary_mask = np.asarray(binary_mask)
    return binary_mask


def get_image_for_ophys_time(ophys_timestamp, stimulus_table):
    flash_number = np.searchsorted(stimulus_table['ophys_start_time'], ophys_timestamp) - 1
    flash_number = flash_number[0]
    end_flash = np.searchsorted(stimulus_table['ophys_end_time'], ophys_timestamp)
    end_flash = end_flash[0]
    if flash_number == end_flash:
        return stimulus_table.loc[flash_number]['image_name']
    else:
        return None


def get_stimulus_df_for_ophys_times(stimulus_table, timestamps_ophys):
    timestamps_df = pd.DataFrame(timestamps_ophys, columns=['ophys_timestamp'])
    timestamps_df['image'] = timestamps_df['ophys_timestamp'].map(lambda x: get_image_for_ophys_time(x, stimulus_table))
    stimulus_df = pd.get_dummies(timestamps_df, columns=['image'])
    stimulus_df.insert(loc=1, column='image', value=timestamps_df['image'])
    stimulus_df.insert(loc=0, column='ophys_frame', value=np.arange(0, len(timestamps_ophys), 1))
    return stimulus_df


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


def get_active_cell_indices(dff_traces):
    snr_values = []
    for i, trace in enumerate(dff_traces):
        mean = np.mean(trace, axis=0)
        std = np.std(trace, axis=0)
        snr = mean / std
        snr_values.append(snr)
    active_cell_indices = np.argsort(snr_values)[-10:]
    return active_cell_indices


def save_active_cell_indices(dataset, active_cell_indices, save_dir):
    experiment_id = dataset.experiment_id
    cell_specimen_ids = [dataset.get_cell_specimen_id_for_cell_index(cell) for cell in active_cell_indices]
    cells = pd.DataFrame(active_cell_indices, columns=['cell'])
    cells['cell_specimen_id'] = cell_specimen_ids
    cells.to_csv(os.path.join(save_dir, str(experiment_id) + '_active_cell_indices.csv'))


def get_saved_active_cell_indices(experiment_id, data_dir):
    data_file = [file for file in os.listdir(data_dir) if str(experiment_id) + '_active_cell_indices' in file]
    if len(data_file) > 0:
        df = pd.read_csv(os.path.join(data_dir, data_file[0]))
        active_cell_indices = df['cell'].values
        active_cell_specimen_ids = df['cell_specimen_id'].values
    else:
        print('cant find active cell indices')
        active_cell_indices = None
        active_cell_specimen_ids = None
    return active_cell_indices, active_cell_specimen_ids


def get_unrewarded_first_lick_times(dataset):
    trials = dataset.trials.copy()
    all_licks = dataset.licks.time.values
    #     rewarded_lick_times = trials[trials.trial_type=='go'].lick_times.values
    rewarded_lick_times = trials[trials.response_type.isin(['HIT'])].lick_times.values
    rewarded_lick_times = np.hstack(list(rewarded_lick_times))
    unrewarded_lick_times = [lick for lick in all_licks if lick not in rewarded_lick_times]
    unrewarded_lick_times = np.asarray(unrewarded_lick_times)

    median_inter_lick_interval = np.median(np.diff(unrewarded_lick_times))
    #     print median_inter_lick_interval

    first_licks = []
    for i, lick in enumerate(unrewarded_lick_times):
        #         print lick, unrewarded_lick_times[i-1], lick - unrewarded_lick_times[i-1]
        if lick - unrewarded_lick_times[i - 1] > median_inter_lick_interval * 3:
            #             print 'first lick '
            first_licks.append(lick)
    first_lick_times = np.asarray(first_licks)
    return first_lick_times


def get_experiments_with_omissions(df):
    omitted = []
    for experiment_id in df.experiment_id.unique():
        if 'omitted' in df[df.experiment_id == experiment_id].image_name.unique():
            omitted.append(experiment_id)
    return omitted


def get_experiments_without_omissions(df):
    omission_expts = get_experiments_with_omissions(df)
    expts_without_omissions = [expt_id for expt_id in df.experiment_id.unique() if expt_id not in omission_expts]
    return expts_without_omissions


def filter_omission_experiments(df):
    df = df[df.experiment_id.isin(get_experiments_with_omissions(df))]
    return df


def get_ramp_index(mean_trace, ophys_frame_rate=31.):
    fr = ophys_frame_rate
    flash_window = [-0.5, 0.75]
    onset = int(fr * np.abs(flash_window[0]))

    # pre stim window is 12 frames prior to stim onset ~400ms
    early = np.nanmean(mean_trace[onset - 12:onset - 8])
    late = np.nanmean(mean_trace[onset - 4:onset])
    pre_stim_ramp_index = np.log2(late / early)

    # stim window is 4 frames after stim onset ~125ms
    early = np.nanmean(mean_trace[onset:onset + 2])
    late = np.nanmean(mean_trace[onset + 2:onset + 4])
    stim_ramp_index = np.log2(late / early)

    # post stim window is 4 frames after stim offset
    early = np.nanmean(mean_trace[onset + 8:onset + 10])
    late = np.nanmean(mean_trace[onset + 10:onset + 12])
    post_stim_ramp_index = np.log2(late / early)

    return pre_stim_ramp_index, stim_ramp_index, post_stim_ramp_index


def get_omission_ramp_index(mean_trace, ophys_frame_rate=31., window=[-3, 3]):
    fr = ophys_frame_rate
    onset = int(fr * np.abs(window[0]))

    # omission window is -4 from omission onset to 22 frames after omission onset ~700ms
    early = np.nanmean(mean_trace[onset - 4:onset])
    late = np.nanmean(mean_trace[onset + 19:onset + 23])
    omission_ramp_index = np.log2(late / early)

    return omission_ramp_index


def remove_inf(df, col_name):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[df[col_name].isnull() == False]
    return df
