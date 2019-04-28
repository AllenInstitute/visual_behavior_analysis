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


def get_successive_frame_list(timepoints_array, timestamps):
    # This is a modification of get_nearest_frame for speedup
    #  This implementation looks for the first 2p frame consecutive to the stim
    successive_frames = np.searchsorted(timestamps, timepoints_array)

    return successive_frames


def get_trace_around_timepoint(timepoint, trace, timestamps, window, frame_rate):
    #   frame_for_timepoint = get_nearest_frame(timepoint, timestamps)
    frame_for_timepoint = get_successive_frame_list(timepoint, timestamps)
    lower_frame = frame_for_timepoint + (window[0] * frame_rate)
    upper_frame = frame_for_timepoint + (window[1] * frame_rate)
    trace = trace[int(lower_frame):int(upper_frame)]
    timepoints = timestamps[int(lower_frame):int(upper_frame)]
    return trace, timepoints


def get_mean_in_window(trace, window, frame_rate, use_events=False):
    # if use_events:
    #     trace[trace==0] = np.nan
    # mean = np.nanmean(trace[int(np.round(window[0] * frame_rate)): int(np.round(window[1] * frame_rate))]) # 181212
    mean = np.nanmean(trace[int(window[0] * frame_rate): int(window[1] * frame_rate)])  # modified 190127
    # if np.isnan(mean):
    #     mean = 0
    return mean


def get_sd_in_window(trace, window, frame_rate):
    # std = np.std(
    #     trace[int(np.round(window[0] * frame_rate)): int(np.round(window[1] * frame_rate))])  # modified 181212
    std = np.std(
        trace[int(window[0] * frame_rate): int(window[1] * frame_rate)])  # modified 190127
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


def get_p_values_from_shuffle_omitted(dataset, stimulus_table, flash_response_df, response_window_duration):
    # data munging
    fdf = flash_response_df.copy()
    st = stimulus_table.copy()
    included_flashes = fdf.flash_number.unique()
    st = st[st.flash_number.isin(included_flashes)]
    ost = dataset.stimulus_table[dataset.stimulus_table.omitted == True]

    ost.loc[:, 'start_frame'] = get_successive_frame_list(ost.start_time.values, dataset.timestamps_ophys)
    ost.loc[:, 'end_frame'] = get_successive_frame_list(ost.end_time.values, dataset.timestamps_ophys)

    # set params
    frame_rate = 31
    stim_frames = int(
        np.round(response_window_duration * frame_rate, 0))  # mean_response_window = 0.5s*31Hz = 16 frames
    cell_indices = dataset.get_cell_indices()
    n_cells = len(cell_indices)
    # get shuffled values from omitted flash sweeps
    shuffled_responses = np.empty((n_cells, 10000, stim_frames))
    idx = np.random.choice(ost.start_frame.values, 10000)
    for i in range(stim_frames):
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

    spontaneous_frames = get_spontaneous_frames(dataset)

    #  params
    ophys_frame_rate = 31
    n_mean_response_window_frames = int(
        np.round(response_window_duration * ophys_frame_rate, 0))  # stimulus window = 0.25ms*31Hz = 7.75 frames
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
    mean_responses = group['mean_response'].values
    sem_response = np.std(group['mean_response'].values) / np.sqrt(len(group['mean_response'].values))
    mean_trace = np.mean(group['trace'])
    sem_trace = np.std(group['trace'].values) / np.sqrt(len(group['trace'].values))
    return pd.Series({'mean_response': mean_response, 'sem_response': sem_response,
                      'mean_trace': mean_trace, 'sem_trace': sem_trace,
                      'mean_responses': mean_responses})


def get_mean_sem(group):
    mean_response = np.mean(group['mean_response'])
    sem_response = np.std(group['mean_response'].values) / np.sqrt(len(group['mean_response'].values))
    return pd.Series({'mean_response': mean_response, 'sem_response': sem_response})


def get_fraction_significant_trials(group):
    fraction_significant_trials = len(group[group.p_value < 0.05]) / float(len(group))
    return pd.Series({'fraction_significant_trials': fraction_significant_trials})


def get_fraction_active_trials(group):
    fraction_active_trials = len(group[group.mean_response > 0.05]) / float(len(group))
    return pd.Series({'fraction_active_trials': fraction_active_trials})


def get_fraction_responsive_trials(group):
    fraction_responsive_trials = len(group[group.p_value_baseline < 0.05]) / float(len(group))
    return pd.Series({'fraction_responsive_trials': fraction_responsive_trials})


def get_fraction_nonzero_trials(group):
    fraction_nonzero_trials = len(group[group.n_events > 0]) / float(len(group))
    return pd.Series({'fraction_nonzero_trials': fraction_nonzero_trials})


# def get_reliability(group):
#     import scipy as sp
#     if 'trial' in group.keys():
#         trials = group['trial'].values
#     elif 'flash_number' in group.keys():
#         trials = group['flash_number'].values
#     corr_values = []
#     traces = group['trace'].values
#     for i, t1 in enumerate(trials[:-1]):
#         for j, t2 in enumerate(trials[:-1]):
#             trial1 = traces[i]
#             trial2 = traces[j]
#             corr = sp.stats.pearsonr(trial1, trial2)[0]
#             corr_values.append(corr)
#     corr_values = np.asarray(corr_values)
#     reliability = np.mean(corr_values)
#     return pd.Series({'reliability': reliability})


def compute_reliability(group, analysis=None, flashes=True, omitted=False):
    from itertools import combinations
    import scipy as sp
    if analysis and omitted:
        response_window = [int(np.abs(analysis.omitted_flash_window[0]) * 31),
                           int(analysis.omitted_flash_window[1] * 31)]
    elif analysis and flashes and not omitted:
        response_window = [int(np.abs(analysis.flash_window[0]) * 31), int(analysis.flash_window[1] * 31)]
    elif analysis and not flashes and not omitted:
        response_window = [int(np.abs(analysis.trial_window[0]) * 31), int(analysis.trial_window[1] * 31)]
    elif not analysis and flashes and not omitted:
        response_window = [int(0.5 * 31), int(1.25 * 31)]
    elif not analysis and omitted:
        response_window = [int(3 * 31), int(6 * 31)]
    else:
        response_window = [int(4 * 31), int(8 * 31)]
    corr_values = []
    traces = group['trace'].values
    traces = np.vstack(traces)
    traces = traces[:, response_window[0]:response_window[1]]  # limit to post change window
    combos = combinations(traces, 2)
    corr_values = []
    for combo in combos:
        corr = sp.stats.pearsonr(combo[0], combo[1])[0]
        corr_values.append(corr)
    corr_values = np.asarray(corr_values)
    reliability = np.mean(corr_values)
    return pd.Series({'reliability': reliability})


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


def get_mean_df(response_df, analysis=None, conditions=['cell', 'change_image_name'], flashes=False, omitted=False,
                get_reliability=False):
    window = get_window(analysis, flashes, omitted)

    rdf = response_df.copy()

    mdf = rdf.groupby(conditions).apply(get_mean_sem_trace)
    mdf = mdf[['mean_response', 'sem_response', 'mean_trace', 'sem_trace', 'mean_responses']]
    mdf = mdf.reset_index()
    mdf = annotate_mean_df_with_pref_stim(mdf)
    if analysis is not None:
        mdf = annotate_mean_df_with_p_value(analysis, mdf, window)
        mdf = annotate_mean_df_with_sd_over_baseline(analysis, mdf, window=window)
        mdf = annotate_mean_df_with_time_to_peak(analysis, mdf, window=window)
        mdf = annotate_mean_df_with_fano_factor(analysis, mdf)

    fraction_significant_trials = rdf.groupby(conditions).apply(get_fraction_significant_trials)
    fraction_significant_trials = fraction_significant_trials.reset_index()
    mdf['fraction_significant_trials'] = fraction_significant_trials.fraction_significant_trials

    fraction_active_trials = rdf.groupby(conditions).apply(get_fraction_active_trials)
    fraction_active_trials = fraction_active_trials.reset_index()
    mdf['fraction_active_trials'] = fraction_active_trials.fraction_active_trials

    fraction_responsive_trials = rdf.groupby(conditions).apply(get_fraction_responsive_trials)
    fraction_responsive_trials = fraction_responsive_trials.reset_index()
    mdf['fraction_responsive_trials'] = fraction_responsive_trials.fraction_responsive_trials

    fraction_nonzero_trials = rdf.groupby(conditions).apply(get_fraction_nonzero_trials)
    fraction_nonzero_trials = fraction_nonzero_trials.reset_index()
    mdf['fraction_nonzero_trials'] = fraction_nonzero_trials.fraction_nonzero_trials

    if get_reliability:
        reliability = rdf.groupby(conditions).apply(compute_reliability, analysis, flashes, omitted)
        reliability = reliability.reset_index()
        mdf['reliability'] = reliability.reliability

    return mdf


def get_cre_lines(mean_df):
    cre_lines = np.sort(mean_df.cre_line.unique())
    return cre_lines


def get_colors_for_cre_lines():
    colors = [sns.color_palette()[2], sns.color_palette()[4]]
    return colors


def get_image_sets(mean_df):
    image_sets = np.sort(mean_df.image_set.unique())
    return image_sets


def get_image_names(mean_df):
    if 'change_image_name' in mean_df.keys():
        image_names = np.sort(mean_df.change_image_name.unique())
    else:
        image_names = np.sort(mean_df.image_name.unique())
        # image_names = image_names[image_names!='omitted']
    return image_names


def get_color_for_image_name(image_names, image_name):
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
        color = colors[7]
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


def add_metadata_to_mean_df(mdf, metadata):
    metadata = metadata.reset_index()
    metadata = metadata.rename(columns={'ophys_experiment_id': 'experiment_id'})
    metadata = metadata.drop(columns=['ophys_frame_rate', 'stimulus_frame_rate', 'index'])
    if metadata['experiment_container_id'].values[0] == None:
        metadata['experiment_container_id'] = 0
    else:
        metadata['experiment_container_id'] = float(metadata['experiment_container_id'].values[0])
    metadata['experiment_id'] = float(metadata.experiment_id.values[0])
    metadata['image_set'] = metadata.session_type.values[0][-1]
    metadata['training_state'] = ['trained' if image_set == 'A' else 'untrained' for image_set in metadata.image_set.values]
    metadata['stage'] = str(metadata.stage.values[0])
    mdf = mdf.merge(metadata, how='outer', on='experiment_id')
    return mdf


def get_time_to_peak(analysis, trace, window=[-4, 8]):
    response_window_duration = analysis.response_window_duration
    response_window = [np.abs(window[0]), np.abs(window[0]) + response_window_duration]
    frame_rate = analysis.ophys_frame_rate
    response_window_trace = trace[int(response_window[0] * frame_rate):(int(response_window[1] * frame_rate))]
    peak_response = np.amax(response_window_trace)
    peak_frames_from_response_window_start = np.where(response_window_trace == np.amax(response_window_trace))[0][0]
    time_to_peak = peak_frames_from_response_window_start / float(frame_rate)
    return peak_response, time_to_peak


def annotate_mean_df_with_time_to_peak(analysis, mean_df, window=[-4, 8]):
    ttp_list = []
    peak_list = []
    for idx in mean_df.index:
        mean_trace = mean_df.iloc[idx].mean_trace
        peak_response, time_to_peak = get_time_to_peak(analysis, mean_trace, window=window)
        ttp_list.append(time_to_peak)
        peak_list.append(peak_response)
    mean_df['peak_response'] = peak_list
    mean_df['time_to_peak'] = ttp_list
    return mean_df


def annotate_mean_df_with_fano_factor(analysis, mean_df):
    ff_list = []
    for idx in mean_df.index:
        mean_responses = mean_df.iloc[idx].mean_responses
        sd = np.std(mean_responses)
        mean_response = np.mean(mean_responses)
        fano_factor = (sd * 2) / mean_response
        ff_list.append(fano_factor)
    mean_df['fano_factor'] = ff_list
    return mean_df


def annotate_mean_df_with_p_value(analysis, mean_df, window=[-4, 8]):
    response_window_duration = analysis.response_window_duration
    response_window = [np.abs(window[0]), np.abs(window[0]) + response_window_duration]
    frame_rate = analysis.ophys_frame_rate
    p_val_list = []
    for idx in mean_df.index:
        mean_trace = mean_df.iloc[idx].mean_trace
        p_value = get_p_val(mean_trace, response_window, frame_rate)
        p_val_list.append(p_value)
    mean_df['p_value'] = p_val_list
    return mean_df


def annotate_mean_df_with_sd_over_baseline(analysis, mean_df, window=[-4, 8]):
    response_window_duration = analysis.response_window_duration
    response_window = [np.abs(window[0]), np.abs(window[0]) + response_window_duration]
    baseline_window = [np.abs(window[0]) - response_window_duration, (np.abs(window[0]))]
    frame_rate = analysis.ophys_frame_rate
    sd_list = []
    for idx in mean_df.index:
        mean_trace = mean_df.iloc[idx].mean_trace
        sd = get_sd_over_baseline(mean_trace, response_window, baseline_window, frame_rate)
        sd_list.append(sd)
    mean_df['sd_over_baseline'] = sd_list
    return mean_df


def annotate_mean_df_with_pref_stim(mean_df):
    if 'image_name' in mean_df.keys():
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
        image_index = np.where(m.loc[cell]['mean_response'].values == np.nanmax(m.loc[cell]['mean_response'].values))[0][0]
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
