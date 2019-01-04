"""
Created on Saturday July 14 2018

@author: marinag
"""

import numpy as np
from scipy import stats
import pandas as pd


import logging
logger = logging.getLogger(__name__)


# def get_nearest_frame(timepoint, timestamps):
#     return int(np.nanargmin(abs(timestamps - timepoint)))

#modified 181212
def get_nearest_frame(timepoint, timestamps, timepoint_must_be_before=True):
    nearest_frame = int(np.nanargmin(abs(timestamps - timepoint)))
    nearest_timepoint = timestamps[nearest_frame]
    if nearest_timepoint>timepoint == False: #nearest frame time must be greater than provided timepoint
        nearest_frame = nearest_frame-1 #use previous frame to ensure nearest follows input timepoint
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
    return np.std(trace[int(np.round(window[0] * frame_rate)): int(np.round(window[1] * frame_rate))]) #modified 181212


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


def get_mean_df(analysis, response_df, conditions=['cell', 'change_image_name'], flashes=False):
    rdf = response_df.copy()

    mdf = rdf.groupby(conditions).apply(get_mean_sem_trace)
    mdf = mdf[['mean_response', 'sem_response', 'mean_trace', 'sem_trace', 'mean_responses']]
    mdf = mdf.reset_index()
    mdf = annotate_mean_df_with_pref_stim(mdf, flashes=flashes)
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


def get_colors_for_cre_lines():
    colors = [sns.color_palette()[2],sns.color_palette()[4]]
    return colors


def get_image_names(mean_df):
    image_names = np.sort(mean_df.change_image_name.unique())
    return image_names


def add_metadata_to_mean_df(mdf, metadata):
    metadata = metadata.reset_index()
    metadata = metadata.rename(columns={'ophys_experiment_id': 'experiment_id'})
    metadata = metadata.drop(columns=['ophys_frame_rate', 'stimulus_frame_rate', 'index'])
    metadata['experiment_id'] = [int(experiment_id) for experiment_id in metadata.experiment_id]
    metadata['image_set'] = metadata.session_type.values[0][-1]
    metadata['training_state'] = ['trained' if image_set == 'A' else 'untrained' for image_set in metadata.image_set.values]
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
    response_window_trace = trace[int(response_window[0]*frame_rate):(int(response_window[1]*frame_rate))]
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
        fano_factor = (sd*2)/mean_response
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
    ls = ((1-(1/N) * ((np.power(image_responses.sum(axis=0),2)) / (np.power(image_responses,2).sum(axis=0)))) / (1-(1/N)))

    return ls


def plot_ranked_image_tuning_curve_trial_types(analysis, cell, ax=None, save=False, use_events=False):
    from scipy.stats import sem as compute_sem
    from visual_behavior.ophys.response_analysis import utilities as ut
    c = sns.color_palette()
    colors = [c[3],c[0],c[2]]
    ylabel, suffix = get_ylabel_and_suffix(use_events)
    cell_specimen_id = analysis.dataset.get_cell_specimen_id_for_cell_index(cell)
    tdf = ut.get_mean_df(analysis, analysis.trial_response_df,
                         conditions = ['cell_specimen_id','change_image_name','trial_type'])
    if ax is None:
        figsize = (6,4)
        fig,ax = plt.subplots(figsize=figsize)
    ls_list = []
    for t,trial_type in enumerate(['go','catch']):
        tmp = tdf[(tdf.cell_specimen_id==cell_specimen_id)&(tdf.trial_type==trial_type)]
        responses = tmp.mean_response.values
        ls = ut.compute_lifetime_sparseness(responses)
        ls_list.append(ls)
        order = np.argsort(responses)[::-1]
        images = tmp.change_image_name.unique()[order]
        for i,image in enumerate(images):
            means = tmp[tmp.change_image_name==image].mean_response.values[0]
            sem = compute_sem(means)
            ax.errorbar(i,np.mean(means),yerr=sem,color=colors[t])
            ax.plot(i,np.mean(means),'o',color=colors[t])
        ax.plot(i,np.mean(means),'o',color=colors[t],label=trial_type)
#     ax.set_ylim(ymin=0)
    ax.set_ylabel('mean '+ylabel)
    ax.set_xticks(np.arange(0,len(responses),1))
    ax.set_xticklabels(images,rotation=90);
    ax.legend()
    ax.set_title('lifetime sparseness go: '+str(np.round(ls_list[0],3))+'\nlifetime sparseness catch: '+str(np.round(ls_list[1],3)));
    if save:
        save_figure(fig,figsize,analysis.dataset.analysis_dir,'lifetime_sparseness'+suffix,'trial_types_tc_'+str(cell_specimen_id))
    return ax

def plot_ranked_image_tuning_curve_all_flashes(analysis, cell, ax=None, save=None, use_events=False):
    from scipy.stats import sem as compute_sem
    from visual_behavior.ophys.response_analysis import utilities as ut
    c = sns.color_palette()
    colors = [c[3],c[0],c[2]]
    ylabel, suffix = get_ylabel_and_suffix(use_events)
    cell_specimen_id = analysis.dataset.get_cell_specimen_id_for_cell_index(cell)
    fdf = analysis.flash_response_df.copy()
    fmdf = ut.get_mean_df(analysis, fdf, conditions = ['cell_specimen_id','image_name'], flashes=True)
    if ax is None:
        figsize = (6,4)
        fig,ax = plt.subplots(figsize=figsize)
    tmp = fdf[(fdf.cell_specimen_id==cell_specimen_id)]
    responses = fmdf[(fmdf.cell_specimen_id==cell_specimen_id)].mean_response.values
    ls = ut.compute_lifetime_sparseness(responses)
    order = np.argsort(responses)[::-1]
    images = fmdf[(fmdf.cell_specimen_id==cell_specimen_id)].image_name.values
    images = images[order]
    for i,image in enumerate(images):
        means = tmp[tmp.image_name==image].mean_response.values
        sem = compute_sem(means)
        ax.errorbar(i,np.mean(means),yerr=sem,color=colors[1])
        ax.plot(i,np.mean(means),'o',color=colors[1])
        ax.plot(i,np.mean(means),'o',color=colors[1])
#     ax.set_ylim(ymin=0)
    ax.set_ylabel('mean dF/F')
    ax.set_xticks(np.arange(0,len(responses),1))
    ax.set_xticklabels(images,rotation=90);
    ax.set_title('lifetime sparseness all flashes: '+str(np.round(ls,3)));
    ax.legend()
    if save:
        save_figure(fig,figsize,save_dir,'lifetime_sparseness_flashes','roi_'+str(cell))
    return ax

def plot_ranked_image_tuning_curve_flashes(analysis, cell, ax=None, save=None, use_events=False):
    from scipy.stats import sem as compute_sem
    from visual_behavior.ophys.response_analysis import utilities as ut
    c = sns.color_palette()
    colors = [c[3],c[0],c[2]]
    ylabel, suffix = get_ylabel_and_suffix(use_events)
    cell_specimen_id = analysis.dataset.get_cell_specimen_id_for_cell_index(cell)
    fdf = analysis.flash_response_df.copy()
    repeats = [1,5,10]
    fdf = fdf[fdf.repeat.isin(repeats)]
    fmdf = ut.get_mean_df(analysis, fdf, conditions = ['cell_specimen_id','image_name','repeat'], flashes=True)
    if ax is None:
        figsize = (6,4)
        fig,ax = plt.subplots(figsize=figsize)
    ls_list = []
    for r, repeat in enumerate(fmdf.repeat.unique()):
        tmp = fdf[(fdf.cell_specimen_id==cell_specimen_id)&(fdf.repeat==repeat)]
        responses = fmdf[(fmdf.cell_specimen_id==cell_specimen_id)&(fmdf.repeat==repeat)].mean_response.values
        ls = ut.compute_lifetime_sparseness(responses)
        ls_list.append(ls)
        if r == 0:
            order = np.argsort(responses)[::-1]
            images = fmdf[(fmdf.cell_specimen_id==cell_specimen_id)&(fmdf.repeat==repeat)].image_name.values
            images = images[order]
        for i,image in enumerate(images):
            means = tmp[tmp.image_name==image].mean_response.values
            sem = compute_sem(means)
            ax.errorbar(i,np.mean(means),yerr=sem,color=colors[r])
            ax.plot(i,np.mean(means),'o',color=colors[r])
            ax.plot(i,np.mean(means),'o',color=colors[r])
    #     ax.set_ylim(ymin=0)
    ax.set_ylabel('mean '+ylabel)
    ax.set_xticks(np.arange(0,len(responses),1))
    ax.set_xticklabels(images,rotation=90);
    ax.set_title('lifetime sparseness repeat '+str(repeats[0])+': '+str(np.round(ls_list[0],3))+
                 '\nlifetime sparseness repeat '+str(repeats[1])+': '+str(np.round(ls_list[1],3))+
                 '\nlifetime sparseness repeat '+str(repeats[2])+': '+str(np.round(ls_list[2],3)))
    ax.legend()
    if save:
        fig.tight_layout()
        save_figure(fig,figsize,analysis.dataset.analysis_dir,'lifetime_sparseness_flashes'+suffix,str(cell_specimen_id))
    return ax


def plot_mean_trace_from_mean_df(mean_df, ophys_frame_rate, label=None, color='k', interval_sec=1, xlims=(2,6), ax=None, use_events=False):
    ylabel, suffix = get_ylabel_and_suffix(use_events)
    if ax is None:
        fig,ax = plt.subplots()
    mean_trace = mean_df.mean_trace.values[0]
    sem = mean_df.sem_trace.values[0]
    times = np.arange(0, len(mean_trace), 1)
    ax.plot(mean_trace, label=label, linewidth=3, color=color)
    ax.fill_between(times, mean_trace + sem, mean_trace - sem, alpha=0.5, color=color)
    xticks, xticklabels = get_xticks_xticklabels(mean_trace, analysis.ophys_frame_rate, interval_sec=1)
    ax.set_xticks(xticks);
    ax.set_xticklabels(xticklabels);
    ax.set_xlim([xlims[0]*ophys_frame_rate, xlims[1]*ophys_frame_rate])
    ax.set_xlabel('time after change (s)')
    ax.set_ylabel(ylabel)
    sns.despine(ax=ax)
    return ax

def plot_mean_trace_with_variability(traces, frame_rate, ylabel='dF/F', label=None, color='k', interval_sec=1, xlims=[-4, 4],
                    ax=None):
#     xlims = [xlims[0] + np.abs(xlims[1]), xlims[1] + xlims[1]]
    if ax is None:
        fig, ax = plt.subplots()
    if len(traces) > 0:
        mean_trace = np.mean(traces)
        times = np.arange(0, len(mean_trace), 1)
        sem = (traces.std()) / np.sqrt(float(len(traces)))
        for trace in traces:
            ax.plot(trace, linewidth=1, color='gray')
        ax.plot(mean_trace, label=label, linewidth=3, color=color, zorder=100)
        xticks, xticklabels = get_xticks_xticklabels(mean_trace, frame_rate, interval_sec)
        ax.set_xticks([int(x) for x in xticks])
        ax.set_xticklabels([int(x) for x in xticklabels])
        ax.set_xlim(xlims[0] * int(frame_rate), xlims[1] * int(frame_rate))
        ax.set_xlabel('time (sec)')
        ax.set_ylabel(ylabel)
    sns.despine(ax=ax)
    return ax



def plot_mean_response_pref_stim_metrics(analysis, cell, ax=None, save=None, use_events=False):
    import visual_behavior.ophys.response_analysis.utilities as ut
    cell_specimen_id = analysis.dataset.get_cell_specimen_id_for_cell_index(cell)
    tdf = analysis.trial_response_df.copy()
    tdf = tdf[tdf.cell_specimen_id==cell_specimen_id]
    mdf = ut.get_mean_df(analysis, analysis.trial_response_df, conditions = ['cell_specimen_id','change_image_name','trial_type'])
    mdf = mdf[mdf.cell_specimen_id==cell_specimen_id]
    if ax is None:
        figsize=(12,6)
        fig,ax = plt.subplots(1,2,figsize=figsize,sharey=True)
        ax = ax.ravel()
    pref_image = tdf[tdf.pref_stim==True].change_image_name.values[0]
    images = np.sort(tdf.change_image_name.unique())
    stim_code = np.where(images==pref_image)[0][0]
    color = get_color_for_image_name(analysis.dataset, pref_image)
    for i,trial_type in enumerate(['go', 'catch']):
        tmp = tdf[(tdf.trial_type==trial_type)&(tdf.change_image_name==pref_image)]
        mean_df = mdf[(mdf.trial_type==trial_type)&(mdf.change_image_name==pref_image)]
        ax[i] = plot_mean_trace_with_variability(tmp.trace.values, analysis.ophys_frame_rate, label=None, color=color, interval_sec=1,xlims=(2,6),ax=ax[i])
        ax[i] = plot_flashes_on_trace(ax[i], analysis, trial_type=trial_type, omitted=False, alpha=.05*8)
        mean = np.round(mean_df.mean_response.values[0],3)
        p_val = np.round(mean_df.p_value.values[0],4)
        sd = np.round(mean_df.sd_over_baseline.values[0],2)
        time_to_peak = np.round(mean_df.time_to_peak.values[0],3)
        fano_factor = np.round(mean_df.fano_factor.values[0],3)
        ax[i].set_title(trial_type+' - mean: '+str(mean)+'\np_val: '+str(p_val)+', sd: '+str(sd)+
                        '\ntime_to_peak: '+str(time_to_peak)+
                       '\nfano_factor: '+str(fano_factor));
    ax[1].set_ylabel('')
    if save:
        fig.tight_layout()
        plt.gcf().subplots_adjust(top=0.7)
        save_figure(fig, figsize, analysis.dataset.analysis_dir, 'mean_response_pref_stim_metrics', str(cell_specimen_id))
        plt.close()
    return ax


def format_table_data(dataset):
    table_data = dataset.metadata.copy()
    table_data = table_data[['specimen_id','donor_id','targeted_structure','imaging_depth',
                             'experiment_date','cre_line','reporter_line','session_type']]
    table_data['experiment_date'] = str(table_data['experiment_date'].values[0])[:10]
    table_data = table_data.transpose()
    return table_data

def get_color_for_image_name(dataset, image_name):
    images = np.sort(dataset.stimulus_table.image_name.unique())
    colors = sns.color_palette("hls", len(images))
    image_index = np.where(images==image_name)[0][0]
    color = colors[image_index]
    return color

def plot_images(dataset, orientation='row', color_box=True, save=False, ax=None):
    orientation = 'row'
    if orientation == 'row':
        figsize = (20, 5)
        cols = len(dataset.stimulus_metadata)
        rows = 1
        if rows == 2:
            cols = cols/2
            figsize = (10,4)
    elif orientation == 'column':
        figsize = (5, 20)
        cols = 1
        rows = len(dataset.stim_codes.stim_code.unique())
    if ax is None:
        fig, ax = plt.subplots(rows, cols, figsize=figsize)
        ax = ax.ravel();

    stimuli = dataset.stimulus_metadata
    image_names = np.sort(dataset.stimulus_table.image_name.unique())
    colors = sns.color_palette("hls", len(image_names))
    for i, image_name in enumerate(image_names):
        image_index = stimuli[stimuli.image_name==image_name].image_index.values[0]
        image = dataset.stimulus_template[image_index]
        ax[i].imshow(image, cmap='gray', vmin=0, vmax=np.amax(image));
        ax[i].grid('off')
        ax[i].axis('off')
        ax[i].set_title(image_name, color='k');
        if color_box:
            linewidth = 6
            ax[i].axhline(y=-20, xmin=0.04, xmax=0.95, linewidth=linewidth, color=colors[i]);
            ax[i].axhline(y=image.shape[0] - 20, xmin=0.04, xmax=0.95, linewidth=linewidth, color=colors[i]);
            ax[i].axvline(x=-30, ymin=0.05, ymax=0.95, linewidth=linewidth, color=colors[i]);
            ax[i].axvline(x=image.shape[1], ymin=0, ymax=0.95, linewidth=linewidth, color=colors[i]);
            # ax[i].set_title(str(stim_code), color=colors[i])
    if save:
        title ='images_'+str(rows)
        if color_box:
            title = title+'_c'
        save_figure(fig, figsize, dataset.analysis_dir, 'images', title, formats=['.png'])
    return ax


def plot_cell_summary_figure(analysis, cell_index, save=False, show=False, cache_dir=None, use_events=False):
    cell_specimen_id = dataset.get_cell_specimen_id_for_cell_index(cell_index)
    rdf = analysis.trial_response_df.copy()
    fdf = analysis.flash_response_df.copy()
    ylabel, suffix = get_ylabel_and_suffix(use_events)

    figsize = [2 * 11, 2 * 8.5]
    fig = plt.figure(figsize=figsize, facecolor='white')

    ax = esf.placeAxesOnGrid(fig, dim=(1, 1), xspan=(.2, .7), yspan=(0, .2))
    ax = plot_behavior_events_trace(dataset, [cell_index], xmin=600, length=2, ax=ax, save=False, use_events=use_events)
    ax.set_title(dataset.analysis_folder)

    ax = esf.placeAxesOnGrid(fig, dim=(1, 1), xspan=(.0, .20), yspan=(0, .22))
    ax = sf.plot_cell_zoom(dataset.roi_mask_dict, dataset.max_projection, cell_specimen_id, spacex=25, spacey=25,
                           show_mask=True, ax=ax)
    ax.set_title('cell ' + str(cell_index) + ' - ' + str(cell_specimen_id))

    ax = esf.placeAxesOnGrid(fig, dim=(1, 1), xspan=(.0, .7), yspan=(.16, .35))
    ax = plot_trace(dataset.timestamps_ophys, dataset.dff_traces[cell_index, :], ax,
                    title='cell_specimen_id: ' + str(cell_specimen_id), ylabel=ylabel)
    ax = plot_behavior_events(dataset, ax)
    ax.set_title('')

    ax = esf.placeAxesOnGrid(fig, dim=(1, len(rdf.change_image_name.unique())), xspan=(.0, .7), yspan=(.33, .55),
                             wspace=0.35)
    vmax = np.percentile(dataset.dff_traces[cell_index, :], 99.9)
    ax = plot_transition_type_heatmap(analysis, [cell_index], vmax=vmax, ax=ax, cmap='magma', colorbar=False)

    ax = esf.placeAxesOnGrid(fig, dim=(1, 2), xspan=(.0, .5), yspan=(.53, .75), wspace=0.25, sharex=True, sharey=True)
    ax = plot_image_response_for_trial_types(analysis, cell_index, legend=False, save_dir=None, use_events=use_events,
                                             ax=ax)

    if 'omitted' in analysis.flash_response_df.keys():
        try:
            ax = esf.placeAxesOnGrid(fig, dim=(1, 1), xspan=(.46, .66), yspan=(.57, .77))
            ax = plot_omitted_flash_response_all_stim(analysis.omitted_flash_response_df, cell, ax=ax)
            ax.legend(bbox_to_anchor=(1.4, 2))
        except:
            'cant plot omitted flashes'

    fig.tight_layout()

    # ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.7, 0.88), yspan=(.0, .2))
    # ax = plot_mean_trace_behavioral_response_types_pref_image(rdf, sdf, cell, behavioral_response_types=['HIT', 'MISS'],
    #                                                              ax=ax)
    #
    # ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.7, 0.88), yspan=(.2, .4))
    # ax = plot_mean_trace_behavioral_response_types_pref_image(rdf, sdf, cell, behavioral_response_types=['FA', 'CR'],
    #                                                              ax=ax)
    #
    # ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.7, 0.88), yspan=(.39, .59))
    # ax = plot_running_not_running(rdf, sdf, cell, trial_type='go', ax=ax)
    #
    # ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.7, 0.88), yspan=(.58, 0.78))
    # ax = plot_engaged_disengaged(rdf, sdf, cell, code='change_image', trial_type='go', ax=ax)

    ax = esf.placeAxesOnGrid(fig, dim=(8, 1), xspan=(.68, .86), yspan=(.2, .99), wspace=0.25, hspace=0.25)
    try:
        ax = plot_images(dataset, orientation='column', color_box=True, save=False, ax=ax);
    except:
        pass

    # ax = esf.placeAxesOnGrid(fig, dim=(1, 1), xspan=(0.0, 0.2), yspan=(.79, 1))
    # ax = plot_mean_cell_response_heatmap(analysis, cell, values='mean_response', index='initial_image_name',
    #                                     columns='change_image_name', save=False, ax=ax, use_events=use_events)

    ax = esf.placeAxesOnGrid(fig, dim=(1, 1), xspan=(0.5, 0.7), yspan=(0.55, 0.75))
    ax = plot_ranked_image_tuning_curve_trial_types(analysis, cell, ax=ax, save=False, use_events=use_events)

    ax = esf.placeAxesOnGrid(fig, dim=(1, 1), xspan=(0.5, 0.7), yspan=(0.78, 0.99))
    # ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(0.2, 0.44), yspan=(.79, 1))
    ax = plot_ranked_image_tuning_curve_flashes(analysis, cell, ax=ax, save=False, use_events=use_events)

    ax = esf.placeAxesOnGrid(fig, dim=(1, 2), xspan=(0.0, 0.5), yspan=(.78, .99), wspace=0.25, sharex=True, sharey=True)
    ax = plot_mean_response_pref_stim_metrics(analysis, cell, ax=ax, save=False, use_events=use_events)

    # # ax = placeAxesOnGrid(fig, dim=(1, 1), xspan=(.83, 1), yspan=(.78, 1))
    # ax = esf.placeAxesOnGrid(fig, dim=(1, 1), xspan=(.7, .99), yspan=(0.05, .16))
    # table_data = format_table_data(dataset)
    # xtable = ax.table(cellText=table_data.values, cellLoc='left', rowLoc='left', loc='center', fontsize=12)
    # xtable.scale(1, 3)
    # ax.axis('off');
    fig.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.05)
    if save:
        save_figure(fig, figsize, analysis.dataset.analysis_dir, 'cell_summary_plots', str(cell_specimen_id))
        if cache_dir:
            save_figure(fig, figsize, cache_dir, 'cell_summary',
                        str(dataset.experiment_id) + '_' + str(cell_specimen_id))
        if not show:
            plt.close()