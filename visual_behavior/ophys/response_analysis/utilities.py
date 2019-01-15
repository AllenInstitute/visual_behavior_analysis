import numpy as np
import pandas as pd

import logging

logger = logging.getLogger(__name__)


# def get_nearest_frame(timepoint, timestamps):
#     return int(np.nanargmin(abs(timestamps - timepoint)))

# modified 181212
def get_nearest_frame(timepoint, timestamps, timepoint_must_be_before=True):
    nearest_frame = int(np.nanargmin(abs(timestamps - timepoint)))
    nearest_timepoint = timestamps[nearest_frame]
    if nearest_timepoint > timepoint == False:  # nearest frame time must be greater than provided timepoint
        nearest_frame = nearest_frame - 1  # use previous frame to ensure nearest follows input timepoint
    return nearest_frame


def get_trace_around_timepoint(timepoint, trace, timestamps, window, frame_rate):
    frame_for_timepoint = get_nearest_frame(timepoint, timestamps)
    lower_frame = frame_for_timepoint + (window[0] * frame_rate)
    upper_frame = frame_for_timepoint + (window[1] * frame_rate)
    trace = trace[int(lower_frame):int(upper_frame)]
    timepoints = timestamps[int(lower_frame):int(upper_frame)]
    return trace, timepoints


def get_mean_in_window(trace, window, frame_rate, use_events=False):
    # if use_events:
    #     trace[trace==0] = np.nan
    mean = np.nanmean(trace[int(np.round(window[0] * frame_rate)): int(np.round(window[1] * frame_rate))])
    # if np.isnan(mean):
    #     mean = 0
    return mean


def get_sd_in_window(trace, window, frame_rate):
    return np.std(
        trace[int(np.round(window[0] * frame_rate)): int(np.round(window[1] * frame_rate))])  # modified 181212


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


def get_fraction_responsive_trials(group):
    fraction_responsive_trials = len(group[group.mean_response > 0.05]) / float(len(group))
    return pd.Series({'fraction_responsive_trials': fraction_responsive_trials})


def get_fraction_nonzero_trials(group):
    fraction_nonzero_trials = len(group[group.n_events > 0]) / float(len(group))
    return pd.Series({'fraction_nonzero_trials': fraction_nonzero_trials})


def get_mean_df(response_df, analysis=None, conditions=['cell', 'change_image_name'], flashes=False):
    rdf = response_df.copy()

    mdf = rdf.groupby(conditions).apply(get_mean_sem_trace)
    mdf = mdf[['mean_response', 'sem_response', 'mean_trace', 'sem_trace', 'mean_responses']]
    mdf = mdf.reset_index()
    mdf = annotate_mean_df_with_pref_stim(mdf, flashes=flashes)
    if analysis is not None:
        mdf = annotate_mean_df_with_p_value(analysis, mdf, flashes=flashes)
        mdf = annotate_mean_df_with_sd_over_baseline(analysis, mdf, flashes=flashes)
        mdf = annotate_mean_df_with_time_to_peak(analysis, mdf, flashes=flashes)
        mdf = annotate_mean_df_with_fano_factor(analysis, mdf)

    fraction_significant_trials = rdf.groupby(conditions).apply(get_fraction_significant_trials)
    fraction_significant_trials = fraction_significant_trials.reset_index()
    mdf['fraction_significant_trials'] = fraction_significant_trials.fraction_significant_trials

    fraction_responsive_trials = rdf.groupby(conditions).apply(get_fraction_responsive_trials)
    fraction_responsive_trials = fraction_responsive_trials.reset_index()
    mdf['fraction_responsive_trials'] = fraction_responsive_trials.fraction_responsive_trials

    fraction_nonzero_trials = rdf.groupby(conditions).apply(get_fraction_nonzero_trials)
    fraction_nonzero_trials = fraction_nonzero_trials.reset_index()
    mdf['fraction_nonzero_trials'] = fraction_nonzero_trials.fraction_nonzero_trials

    return mdf


def get_cre_lines(mean_df):
    cre_lines = np.sort(mean_df.cre_line.unique())
    return cre_lines


def get_image_names(mean_df):
    if 'change_image_name' in mean_df.keys():
        image_names = np.sort(mean_df.change_image_name.unique())
    else:
        image_names = np.sort(mean_df.image_name.unique())
    return image_names


def add_metadata_to_mean_df(mdf, metadata):
    metadata = metadata.reset_index()
    metadata = metadata.rename(columns={'ophys_experiment_id': 'experiment_id'})
    metadata = metadata.drop(columns=['ophys_frame_rate', 'stimulus_frame_rate', 'index'])
    metadata['experiment_id'] = [int(experiment_id) for experiment_id in metadata.experiment_id]
    metadata['image_set'] = metadata.session_type.values[0][-1]
    metadata['training_state'] = ['trained' if image_set == 'A' else 'untrained' for image_set in
                                  metadata.image_set.values]
    # metadata['session_type'] = ['image_set_' + image_set for image_set in metadata.image_set.values]
    mdf = mdf.merge(metadata, how='outer', on='experiment_id')
    return mdf


def get_time_to_peak(analysis, trace, flashes=False):
    if flashes:
        response_window_duration = analysis.response_window_duration
        flash_window = [-response_window_duration, response_window_duration]
        response_window = [flash_window[0] + response_window_duration, flash_window[1]]
    else:
        response_window = analysis.response_window
    frame_rate = analysis.ophys_frame_rate
    response_window_trace = trace[int(response_window[0] * frame_rate):(int(response_window[1] * frame_rate))]
    peak_response = np.amax(response_window_trace)
    peak_frames_from_response_window_start = np.where(response_window_trace == np.amax(response_window_trace))[0][0]
    time_to_peak = peak_frames_from_response_window_start / float(frame_rate)
    return peak_response, time_to_peak


def annotate_mean_df_with_time_to_peak(analysis, mean_df, flashes=False):
    ttp_list = []
    peak_list = []
    for idx in mean_df.index:
        mean_trace = mean_df.iloc[idx].mean_trace
        peak_response, time_to_peak = get_time_to_peak(analysis, mean_trace, flashes=flashes)
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


def annotate_mean_df_with_p_value(analysis, mean_df, flashes=False):
    if flashes:
        response_window_duration = analysis.response_window_duration
        flash_window = [-response_window_duration, response_window_duration]
        response_window = [np.abs(flash_window[0]), np.abs(flash_window[0]) + response_window_duration]
    else:
        response_window = analysis.response_window
    frame_rate = analysis.ophys_frame_rate
    p_val_list = []
    for idx in mean_df.index:
        mean_trace = mean_df.iloc[idx].mean_trace
        p_value = get_p_val(mean_trace, response_window, frame_rate)
        p_val_list.append(p_value)
    mean_df['p_value'] = p_val_list
    return mean_df


def annotate_mean_df_with_sd_over_baseline(analysis, mean_df, flashes=False):
    if flashes:
        response_window_duration = analysis.response_window_duration
        flash_window = [-response_window_duration, response_window_duration]
        response_window = [np.abs(flash_window[0]), np.abs(flash_window[0]) + response_window_duration]
        baseline_window = [np.abs(flash_window[0]) - response_window_duration, (np.abs(flash_window[0]))]
    else:
        response_window = analysis.response_window
        baseline_window = analysis.baseline_window
    frame_rate = analysis.ophys_frame_rate
    sd_list = []
    for idx in mean_df.index:
        mean_trace = mean_df.iloc[idx].mean_trace
        sd = get_sd_over_baseline(mean_trace, response_window, baseline_window, frame_rate)
        sd_list.append(sd)
    mean_df['sd_over_baseline'] = sd_list
    return mean_df


def annotate_mean_df_with_pref_stim(mean_df, flashes=False):
    if flashes:
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
    if 'cell_specimen_id' in fdf.keys():
        cell_key = 'cell_specimen_id'
    else:
        cell_key = 'cell'
    fdf['pref_stim'] = False
    mean_response = fdf.groupby([cell_key, 'image_name']).apply(get_mean_sem)
    m = mean_response.unstack()
    for cell in m.index:
        image_index = np.where(m.loc[cell]['mean_response'].values == np.max(m.loc[cell]['mean_response'].values))[0][0]
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
    n = 0
    for i, image in enumerate(stimulus_table.image_name.values):
        if image != stimulus_table.image_name.values[i - 1]:
            n = 1
            repeat.append(n)
        else:
            n += 1
            repeat.append(n)
    stimulus_table['repeat'] = repeat
    stimulus_table['repeat'] = [int(r) for r in stimulus_table.repeat.values]
    return stimulus_table


def add_repeat_number_to_flash_response_df(flash_response_df, stimulus_table):
    stimulus_table = add_repeat_to_stimulus_table(stimulus_table)
    flash_response_df = flash_response_df.merge(stimulus_table[['flash_number', 'repeat']], on='flash_number')
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
    ophys_frames = [get_nearest_frame(timepoint, timestamps_ophys) for timepoint in behavior_df.time.values]
    ophys_times = [timestamps_ophys[frame] for frame in ophys_frames]
    behavior_df['ophys_frame'] = ophys_frames
    behavior_df['ophys_time'] = ophys_times
    return behavior_df


def add_ophys_times_to_stimulus_table(stimulus_table, timestamps_ophys):
    ophys_start_frames = [get_nearest_frame(timepoint, timestamps_ophys) for timepoint in
                          stimulus_table.start_time.values]
    ophys_start_times = [timestamps_ophys[frame] for frame in ophys_start_frames]
    stimulus_table['ophys_start_frame'] = ophys_start_frames
    stimulus_table['ophys_start_time'] = ophys_start_times

    ophys_end_frames = [get_nearest_frame(timepoint, timestamps_ophys) for timepoint in stimulus_table.end_time.values]
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
