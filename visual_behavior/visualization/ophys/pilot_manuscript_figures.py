import os
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})
sns.set_style('white',
              {'axes.spines.right': False, 'axes.spines.top': False, 'xtick.bottom': True, 'ytick.left': True, })
sns.set_palette('deep')

import visual_behavior.visualization.ophys.population_summary_figures as psf
import visual_behavior.visualization.ophys.experiment_summary_figures as esf
import visual_behavior.visualization.ophys.summary_figures as sf

import visual_behavior.ophys.response_analysis.utilities as ut
from visual_behavior.visualization.utils import save_figure
import visual_behavior.utilities as vbut


### functions to support figure generation for pilot study manuscript ###


### statistics for figures

def get_stats_for_conditions(df, metric, sample_type, group, group_names, condition, condition_names):
    import scipy.stats as stats
    import itertools
    df[metric] = pd.to_numeric(df[metric])
    df_list = []
    for group_name in group_names:
        data = df[(df[group] == group_name)]
        data = data[data[condition].isnull() == False]
        # get rid of NaNs
        condition_values = []
        for condition_name in condition_names:
            t = data[(data[condition] == condition_name)][metric].values
            t = [val for val in t if np.isnan(val) == False]
            condition_values.append(t)
        # ANOVA to test for overall effect
        if len(condition_names) == 2:
            group_f_stat, group_p_value = stats.f_oneway(condition_values[0], condition_values[1])
        elif len(condition_names) == 4:
            group_f_stat, group_p_value = stats.f_oneway(condition_values[0], condition_values[1], condition_values[2],
                                                         condition_values[3])
        # Welch's t-test for all combinations within a group (does not assume equal variance or sample size)
        pairs = itertools.combinations(condition_names, 2)
        n_pairs = sum(1 for p in itertools.combinations(condition_names, 2))
        significance_level = 0.05 / n_pairs  # Bonferroni correction for multiple comparisons
        for pair in pairs:
            cond1_values = data[(data[condition] == pair[0]) & (data[metric].isnull() == False)][metric].values
            cond2_values = data[(data[condition] == pair[1]) & (data[metric].isnull() == False)][metric].values
            pair_t_stat, pair_p_value = stats.ttest_ind(cond1_values, cond2_values, equal_var=False)
            rounded_p_value = np.round(pair_p_value, 6)
            significant = [True if (group_p_value < 0.05) & (pair_p_value <= significance_level) else False]
            df_list.append([metric, sample_type, group_name, group_f_stat, group_p_value, significance_level, condition,
                            pair[0], pair[1], pair_t_stat, pair_p_value, rounded_p_value, significant[0]])
    columns = ['metric', 'sample_type', 'group', 'ANOVA_f_stat', 'ANOVA_p_value', 'corrected_significance_level',
               'condition',
               'condition1', 'condition2', 'ttest_statistic', 'ttest_p_value', 'rounded_p_value', 'significant']
    stats_df = pd.DataFrame(data=df_list, columns=columns)
    return stats_df


def get_label_df(ax, stats_df, condition, condition_values, hue, hue_names):
    import matplotlib
    label_list = []
    for condition_name in condition_values:
        for hue_name in hue_names:
            label_list.append([condition_name, hue_name])
    label_df = pd.DataFrame(data=label_list, columns=['condition_name', 'hue_name'])
    label_df['points'] = None
    label_df['x_pos'] = None
    label_df['y_pos'] = None

    i = 0
    for c in ax.get_children():
        if type(c) == matplotlib.patches.PathPatch:
            bbox = c.get_extents()
            points = bbox.get_points()
            x_pos = np.mean([bbox.get_points()[0][0], bbox.get_points()[1][0]])
            y_pos = bbox.get_points()[1][1]
            label_df.loc[i, 'points'] = points
            label_df.loc[i, 'x_pos'] = x_pos
            label_df.loc[i, 'y_pos'] = y_pos
            i += 1
    return label_df


def label_ax_with_stats(ax, label_df, stats_df, sig_y_val, condition, condition_values, hue, hue_names):
    inv = ax.transData.inverted()
    increment = sig_y_val * .15
    for condition_name in condition_values:
        ypos = sig_y_val
        for hue_name in hue_names:
            tmp = stats_df[(stats_df.group == condition_name) & (stats_df.condition1 == hue_name)]
            for condition2 in tmp.condition2.unique():
                if tmp[(tmp.condition2 == condition2)].significant.values[0] == True:
                    x1 = label_df[
                        (label_df.condition_name == condition_name) & (label_df.hue_name == hue_name)].x_pos.values[0]
                    x2 = label_df[
                        (label_df.condition_name == condition_name) & (label_df.hue_name == condition2)].x_pos.values[0]
                    x1, y1 = inv.transform((x1, ypos))
                    x2, y2 = inv.transform((x2, ypos))
                    y_val = ypos + increment
                    ax.plot(np.mean([x1, x2]), y_val + increment / 3., marker=(6, 2, 0), color='gray')
                    h = ax.hlines(y_val, x1, x2, colors='gray')
                    ypos = ypos + increment
    return ax


def sig_subplot(ax, stats, condition_values):
    import itertools
    pairs = itertools.combinations(condition_values, 2)
    n_pairs = sum(1 for p in itertools.combinations(condition_values, 2))
    for i, pair in enumerate(pairs):
        sig_data = stats[(stats.condition1 == pair[0]) & (stats.condition2 == pair[1])]
        if len(sig_data) == 0:
            sig_data = stats[(stats.condition1 == pair[1]) & (stats.condition2 == pair[0])]
        sig = sig_data.significant.values[0]
        if sig == True:
            x = np.where(condition_values == pair[0])[0][0]
            y = np.where(condition_values[::-1] == pair[1])[0][0]
            ax.plot(x, y, marker=(6, 2, 0), color='gray', markersize=6)

    ax.set_xlim(-0.5, len(condition_values) - 1 + 0.5)
    ax.set_ylim(-0.5, len(condition_values) - 1 + 0.5)
    ax.set_xticks(np.arange(0, len(condition_values), 1))
    ax.set_yticks(np.arange(0, len(condition_values), 1)[::-1])
    ax.set_xticklabels(condition_values)
    ax.set_yticklabels(condition_values)
    return ax


def get_stats_for_session_metric_image_sets(session_summary_df, metric):
    df = session_summary_df.copy()
    df['cre line'] = [cre_line.split('-')[0] for cre_line in df.cre_line.values]

    condition = 'cre line'
    condition_values = np.sort(df[condition].unique())
    hue = 'image_set'
    hue_names = np.sort(df[hue].unique())
    sample_type = 'sessions'

    stats_df = get_stats_for_conditions(df, metric, sample_type, condition, condition_values, hue, hue_names)
    return stats_df


def get_stats_for_session_metric_areas(session_summary_df, metric):
    df = session_summary_df.copy()

    condition = 'image_set'
    condition_values = np.sort(df[condition].unique())
    hue = 'area'
    hue_names = np.sort(df[hue].unique())
    sample_type = 'sessions'

    stats_df = pd.DataFrame()
    for cre_line in df.cre_line.unique():
        tmp = get_stats_for_conditions(df[df.cre_line == cre_line], metric, sample_type, condition, condition_values,
                                       hue, hue_names)
        stats_df = pd.concat([stats_df, tmp])
    return stats_df


def get_stats_for_cell_summary_image_sets(cell_summary_df, metric):
    df = cell_summary_df.copy()
    df['cre line'] = [cre_line.split('-')[0] for cre_line in df.cre_line.values]

    condition = 'cre line'
    condition_values = np.sort(df[condition].unique())
    hue = 'image_set'
    hue_names = np.sort(df[hue].unique())
    sample_type = 'cells'

    stats_df = get_stats_for_conditions(df, metric, sample_type, condition, condition_values, hue, hue_names)
    return stats_df


def get_stats_for_cell_summary_areas(cell_summary_df, metric):
    df = cell_summary_df.copy()
    df['cre line'] = [cre_line.split('-')[0] for cre_line in df.cre_line.values]
    sample_type = 'cells'

    stats_df = pd.DataFrame()
    for cre_line in df.cre_line.unique():
        cdf = df[df.cre_line == cre_line].copy()
        condition = 'image_set'
        condition_values = np.sort(cdf[condition].unique())
        hue = 'area'
        hue_names = np.sort(cdf[hue].unique())

        tmp = get_stats_for_conditions(cdf, metric, sample_type, condition, condition_values, hue, hue_names)
        stats_df = pd.concat([stats_df, tmp])

        condition = 'cre line'
        condition_values = np.sort(cdf[condition].unique())
        hue = 'area'
        hue_names = np.sort(cdf[hue].unique())
        tmp = get_stats_for_conditions(cdf, metric, sample_type, condition, condition_values, hue, hue_names)
        stats_df = pd.concat([stats_df, tmp])

    return stats_df


def get_stats_for_cell_summary_session_numbers(cell_summary_df, metric):
    df = cell_summary_df.copy()
    df['cre line'] = [cre_line.split('-')[0] for cre_line in df.cre_line.values]

    condition = 'cre line'
    condition_values = np.sort(df[condition].unique())
    hue = 'session_number'
    hue_names = np.sort(df[hue].unique())
    sample_type = 'cells'

    stats_df = get_stats_for_conditions(df, metric, sample_type, condition, condition_values, hue, hue_names)
    return stats_df


## behavior - figure 1


def plot_behavior_metrics(rdf, x='trial_type', y='response_rate', ylabel='response rate', hue=None, show_stats=True,
                          colors=None, gray_sessions=False, legend=False, title=None, save_dir=None, suffix=''):
    figsize = (3.5, 3.5)
    fig, ax = plt.subplots(figsize=figsize)
    if colors:
        if not legend:
            if gray_sessions:
                ax = sns.swarmplot(data=rdf, x=x, y=y, color='gray', ax=ax, size=3, zorder=1)
            else:
                ax = sns.swarmplot(data=rdf, x=x, y=y, hue=hue, palette=colors, ax=ax, size=3, zorder=1)
        ax = sns.pointplot(data=rdf, x=x, y=y, hue=hue, palette=colors, ax=ax, zorder=100)
    else:
        if not legend:
            ax = sns.swarmplot(data=rdf, x=x, y=y, color='gray', ax=ax, size=3, zorder=1)
        ax = sns.pointplot(data=rdf, x=x, y=y, color='gray', ax=ax, zorder=100)
    ax.set_xlim(-0.2, len(rdf[x].unique()) - 1 + 0.2)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('')
    if legend:
        suffix = suffix + '_legend'
        l = ax.legend(fontsize='large', title='image set')
        plt.setp(l.get_title(), fontsize='large')
    else:
        suffix = suffix + ''
        if colors:
            ax.get_legend().remove()
        else:
            hue = 'none'
    ax.set_ylim(ymin=0)
    ax.set_xlabel(x.split('_')[0] + ' ' + x.split('_')[1])
    if x is 'image_name':
        ax.set_xticklabels(rdf.image_name.unique(), rotation=90)
    if title:
        ax.set_title(title)

    hue_order = rdf[hue].unique()
    if show_stats:
        sample_type = 'behavior_sessions'
        condition = x
        hue_names = hue_order
        sig_y_val = rdf[y].max()
        metric = y
        condition_values = rdf[condition].unique()
        stats_df = get_stats_for_conditions(rdf, metric, sample_type, condition, condition_values, hue, hue_names)
        label_df = get_label_df(ax, stats_df, condition, condition_values, hue, hue_names)
        ax = label_ax_with_stats(ax, label_df, stats_df, sig_y_val, condition, condition_values, hue, hue_names)

    fig.tight_layout()
    if save_dir:
        sf.save_figure(fig, figsize, save_dir, 'behavior', y + '_' + x + '_' + hue + suffix)


def plot_behavior_metrics_lines(rdf, x='trial_type', y='response_rate', ylabel='response rate', hue=None,
                                show_stats=True,
                                colors=None, legend=False, save_dir=None, suffix=''):
    figsize = (3.5, 3.5)
    fig, ax = plt.subplots(figsize=figsize)
    for experiment_id in rdf.experiment_id.unique():
        t = rdf[rdf.experiment_id == experiment_id]
        y_vals = []
        for x_val in t[x].unique():
            y_vals.append(np.nanmean(t[t[x] == x_val][y].values))
        y_vals = np.asarray(y_vals)
        x_vals = np.arange(0, len(y_vals))
        ax.plot(x_vals, y_vals, '.-', color='gray', linewidth=1, alpha=0.3)
    if colors:
        ax = sns.pointplot(data=rdf, x=x, y=y, hue=hue, palette=colors, ax=ax)
    else:
        ax = sns.pointplot(data=rdf, x=x, y=y, color='gray', ax=ax)
    ax.set_xlim(-0.2, len(rdf[x].unique()) - 1 + 0.2)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('')
    if legend:
        suffix = suffix + '_legend'
        l = ax.legend(fontsize='large', title='image set')
        plt.setp(l.get_title(), fontsize='large')
    else:
        suffix = suffix + ''
        if colors:
            ax.get_legend().remove()
        else:
            hue = 'none'
    ax.set_ylim(ymin=0)
    ax.set_xlabel(x.split('_')[0] + ' ' + x.split('_')[1])
    if x is 'image_name':
        ax.set_xticklabels(rdf.image_name.unique(), rotation=90)

    hue_order = rdf[hue].unique()
    if show_stats:
        sample_type = 'behavior_sessions'
        hue_names = hue_order
        sig_y_val = rdf[y].max()
        metric = y
        condition = x
        condition_values = rdf[x].unique()
        stats_df = get_stats_for_conditions(rdf, metric, sample_type, condition, condition_values, hue, hue_names)
        label_df = get_label_df(ax, stats_df, condition, condition_values, hue, hue_names)
        ax = label_ax_with_stats(ax, label_df, stats_df, sig_y_val, condition, condition_values, hue, hue_names)
    fig.tight_layout()
    if save_dir:
        sf.save_figure(fig, figsize, save_dir, 'behavior', y + '_' + x + '_' + hue + suffix + '_lines')


def generate_image_count_summary(mouse_list):
    from visual_behavior.translator import foraging2, foraging  # NOQA: E402
    df_list = []
    for mouse_id in mouse_list:
        folder = os.path.join(r"\\allen\programs\braintv\workgroups\neuralcoding\Behavior\Data", "M{}".format(mouse_id),
                              "output")
        files = np.sort(os.listdir(folder))
        for file in files:
            if ('NaturalImages' in file) & ('Ophys' not in file):
                pkl_path = os.path.join(folder, file)
                pkl = pd.read_pickle(pkl_path)
                core_data = foraging.data_to_change_detection_core(pkl)
                st = core_data['visual_stimuli']
                tmp = [mouse_id, file]
                for image_name in np.sort(st.image_name.unique()):
                    n = len(st[st.image_name == image_name])
                    tmp = tmp + [n]
                df_list.append(tmp)

    df = pd.DataFrame(data=df_list, columns=['mouse_id', 'filename'] + list(np.sort(st.image_name.unique())))
    df.at[df[df.mouse_id == 'm329071'].index, 'mouse_id'] = 'M329071'
    df.to_hdf(os.path.join(cache_dir, 'multi_session_summary_dfs', 'image_count.h5'), key='df')


def generate_behavior_session_summary(mouse_list):
    df_list = []
    for m in mouse_list:
        mouse_id = "M{}".format(m)
        folder = os.path.join(r"\\allen\programs\braintv\workgroups\neuralcoding\Behavior\Data", mouse_id, "output")
        pkl_path = os.path.join(folder, 'extended_trials_df.pkl')
        mouse_df = pd.read_pickle(pkl_path)
        mouse_df = mouse_df[mouse_df.auto_rewarded != True]
        mouse_df = mouse_df[mouse_df.reward_rate > 2]
        mouse_df['strdate'] = mouse_df.startdatetime.values
        mouse_df['stage_number'] = [stage_name_to_numbers[stage] for stage in mouse_df.stage.values]

        for date in mouse_df.date.unique():
            tmp = mouse_df[mouse_df.date == date]
            manifest_data = manifest[(manifest.date == date) & (manifest.donor_id == m)]
            cre_line = manifest[(manifest.donor_id == m)].cre_line.values[0]
            if len(manifest_data) > 0:
                expt_id = manifest_data.experiment_id.values[0]
            else:
                expt_id = np.nan
            if mouse_id == 'm329071':
                print('changing mouse_id')
                mouse_id = 'M329071'
                print(mouse_id)
            else:
                mouse_id = mouse_id
            stage_number = tmp.stage_number.unique()[0]
            stage_name = tmp.stage.unique()[0]
            if stage_number == 4:  # include image set A exposures prior to ophys stage
                exposure_number = len(
                    mouse_df[(mouse_df.date < date) & (mouse_df.stage_number.isin([3, 4]))].date.unique())
            else:
                exposure_number = len(
                    mouse_df[(mouse_df.date < date) & (mouse_df.stage_number == stage_number)].date.unique())
            color = colors[tmp.stage.unique()[0]]
            hit_rate, catch_rate, d_prime = vbut.get_response_rates(tmp, sliding_window=100)
            mean_d_prime = np.nanmean(d_prime)
            df_list.append(
                [mouse_id, cre_line, date, expt_id, stage_number, stage_name, exposure_number, color, mean_d_prime])

    columns = ['mouse_id', 'cre_line', 'date', 'experiment_id', 'stage_number', 'stage_name', 'exposure_number',
               'color', 'mean_d_prime']
    session_summary = pd.DataFrame(df_list, columns=columns)
    session_summary.to_hdf(os.path.join(cache_dir, 'multi_session_summary_dfs', 'behavior_session_summary.h5'),
                           key='df')


def add_inter_trial_lick_diff_to_trials_df(trials):
    inter_trial_lick_diff = []
    for row in range(len(trials)):
        current_trial = trials.iloc[row]
        previous_trial = trials.iloc[row - 1]
        if len(current_trial.lick_times) > 0 and len(previous_trial.lick_times) > 0:
            inter_trial_lick_diff.append(current_trial.lick_times[0] - previous_trial.lick_times[-1])
        else:
            inter_trial_lick_diff.append(np.nan)
    trials['inter_trial_lick_diff'] = inter_trial_lick_diff
    return trials


def annotate_trials(trials, dataset, window=[-2, 3]):
    running_trace = dataset.running_speed.running_speed.values
    post_change_lick_times = []
    first_lick_time = []
    running_speed = []
    for row in range(len(trials)):
        times = list(trials.iloc[row].lick_times - trials.iloc[row].change_time)
        if len(times) > 0:
            first_lick_time.append(times[0])
        else:
            first_lick_time.append([])
        post_change_lick_times.append(times)
        trial_run_speed, times = ut.get_trace_around_timepoint(trials.iloc[row].change_time, running_trace,
                                                               dataset.timestamps_stimulus, frame_rate=60.,
                                                               window=window)
        running_speed.append(trial_run_speed)
    trials['experiment_id'] = dataset.experiment_id
    trials['post_change_lick_times'] = post_change_lick_times
    trials['first_lick_time'] = first_lick_time
    trials['running_speed'] = running_speed
    trials['trial_duration'] = trials.endtime - trials.starttime
    trials = add_inter_trial_lick_diff_to_trials_df(trials)
    return trials


def annotate_all_trials(all_trials, dataset, window=[-2, 3]):
    post_start_lick_times = []
    post_change_lick_times = []
    first_lick_time_after_start = []
    first_lick_time_after_change = []
    for row in range(len(all_trials)):
        times = list(all_trials.iloc[row].lick_times - all_trials.iloc[row].starttime)
        if len(times) > 0:
            first_lick_time_after_start.append(times[0])
        else:
            first_lick_time_after_start.append([])
        post_start_lick_times.append(times)
        if all_trials.iloc[row].trial_type is 'go' or 'catch':
            times = list(all_trials.iloc[row].lick_times - all_trials.iloc[row].change_time)
            if len(times) > 0:
                first_lick_time_after_change.append(times[0])
            else:
                first_lick_time_after_change.append([])
            post_change_lick_times.append(times)
        else:
            post_change_lick_times.append(np.nan)
    all_trials['post_start_lick_times'] = post_start_lick_times
    all_trials['post_change_lick_times'] = post_change_lick_times
    all_trials['first_lick_time_after_start'] = first_lick_time_after_start
    all_trials['first_lick_time_after_change'] = first_lick_time_after_change
    all_trials['experiment_id'] = dataset.experiment_id
    return all_trials


def generate_multi_session_all_trials_df(experiment_ids):
    from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset
    multi_session_all_trials = pd.DataFrame()
    for experiment_id in experiment_ids:
        dataset = VisualBehaviorOphysDataset(experiment_id, cache_dir=cache_dir)
        all_trials = dataset.all_trials.copy()
        all_trials = annotate_all_trials(all_trials, dataset, window=[-2, 6])
        all_trials = ut.add_metadata_to_mean_df(all_trials, dataset.metadata)
        multi_session_all_trials = pd.concat([multi_session_all_trials, all_trials])

    save_path = os.path.join(cache_dir, 'multi_session_summary_dfs', 'multi_session_all_trials.h5')
    multi_session_all_trials.to_hdf(save_path, key='df')


def generate_multi_session_trials_df(experiment_ids):
    from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset
    multi_session_trials = pd.DataFrame()
    for experiment_id in experiment_ids:
        dataset = VisualBehaviorOphysDataset(experiment_id, cache_dir=cache_dir)
        trials = dataset.trials.copy()
        trials = annotate_trials(trials, dataset, window=[-2, 6])
        trials = ut.add_metadata_to_mean_df(trials, dataset.metadata)
        multi_session_trials = pd.concat([multi_session_trials, trials])
    save_path = os.path.join(cache_dir, 'multi_session_summary_dfs', 'multi_session_trials_df.h5')
    multi_session_trials.to_hdf(save_path, key='df')


def add_trial_types_to_flashes(flashes, trials):
    flashes['change_time'] = flashes.start_time.values
    flashes = pd.merge(flashes, trials[['change_time', 'trial_type']], on='change_time', how='outer')
    return flashes


def add_lick_times_to_flashes(flashes, licks):
    flash_licks = []
    for row in range(len(flashes))[:-1]:
        start = flashes.iloc[row].start_time
        end = flashes.iloc[row + 1].start_time
        lick_times = licks[(licks.time > start) & (licks.time < end)].time.values
        if len(lick_times) > 0:
            flash_licks.append(lick_times)
        else:
            flash_licks.append([])
    flash_licks.append([])
    flashes['lick_times'] = flash_licks
    flashes['n_licks'] = [len(licks) for licks in flashes.lick_times.values]
    return flashes


def add_inter_flash_lick_diff_to_flashes(flashes):
    inter_flash_lick_diff = []
    inter_flash_lick_diff.append(np.nan)
    for row in range(len(flashes))[1:]:
        current_trial = flashes.iloc[row]
        previous_trial = flashes.iloc[row - 1]
        if (len(current_trial.lick_times) > 0) and (len(previous_trial.lick_times) > 0):
            inter_flash_lick_diff.append(current_trial.lick_times[0] - previous_trial.lick_times[-1])
        else:
            inter_flash_lick_diff.append(np.nan)
    flashes['inter_flash_lick_diff'] = inter_flash_lick_diff
    return flashes


def add_first_lick_in_bout_to_flashes(flashes):
    lick_times = flashes[flashes.lick_times.isnull() == False].lick_times.values
    median_inter_lick_interval = np.median(np.diff(np.hstack(list(lick_times))))

    first_lick_in_bout = []
    first_lick_in_bout_time = []
    for row in range(len(flashes)):
        flash_data = flashes.iloc[row]
        if len(flash_data.lick_times) > 0:  # if there is a lick
            if flash_data.inter_flash_lick_diff < median_inter_lick_interval * 3:  # if that lick is within 3x the median inter lick interval
                first_lick_in_bout.append(False)
                first_lick_in_bout_time.append(np.nan)
            else:  # if there is a lick and it is not within ~500ms of last lick, it is the first in a bout
                first_lick_in_bout.append(True)
                first_lick_in_bout_time.append(flash_data.lick_times[0])
        else:
            first_lick_in_bout.append(False)
            first_lick_in_bout_time.append(np.nan)
    flashes['first_lick_in_bout'] = first_lick_in_bout
    flashes['first_lick_in_bout_time'] = first_lick_in_bout_time
    flashes['post_flash_response_latency'] = flashes.first_lick_in_bout_time - flashes.start_time
    return flashes


def get_consumption_licks(flashes):
    lick_times = flashes[flashes.lick_times.isnull() == False].lick_times.values
    median_inter_lick_interval = np.median(np.diff(np.hstack(list(lick_times))))

    flashes['consumption_licks'] = False
    for row in range(len(flashes)):
        row_data = flashes.iloc[row]
        if (row_data.trial_type == 'go') and (row_data.first_lick_in_bout == True):
            flashes.loc[row, 'consumption_licks'] = True
        if (flashes.iloc[row - 1].consumption_licks == True) & (
                flashes.iloc[row].inter_flash_lick_diff < median_inter_lick_interval * 3):
            flashes.loc[row, 'consumption_licks'] = True
    return flashes


def annotate_flashes_with_reward_rate(flashes, trials):
    last_time = 0
    reward_rate_by_frame = []
    trials = trials[trials.trial_type != 'aborted']
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


def annotate_flashes(flashes, trials):
    #     flashes = ut.add_repeat_to_stimulus_table(flashes)
    flashes['repeat'] = flashes.index_within_block.values
    flashes['flash_reward_rate'] = flashes.reward_rate.values
    flashes = annotate_flashes_with_reward_rate(flashes, trials)
    #     flashes = add_lick_times_to_flashes(flashes, licks)
    flashes['lick_times'] = [list(licks) for licks in flashes.licks.values]
    flashes['n_licks'] = [len(licks) for licks in flashes.lick_times.values]
    flashes = add_trial_types_to_flashes(flashes, trials)
    flashes = flashes[:-5]  # get rid of last flashes, can have weirdness
    flashes = add_inter_flash_lick_diff_to_flashes(flashes)
    flashes = add_first_lick_in_bout_to_flashes(flashes)
    flashes = get_consumption_licks(flashes)

    #     flashes = add_flash_after_omitted(flashes)
    return flashes


def get_response_rates(flashes):
    trial_flashes = flashes[flashes.trial_type == 'go']
    hit_rate = len(trial_flashes[trial_flashes.n_licks > 0]) / float(len(trial_flashes))

    catch_flashes = flashes[flashes.trial_type == 'catch']
    catch_fa_rate = len(catch_flashes[catch_flashes.n_licks > 0]) / float(len(catch_flashes))

    non_trial_flashes = flashes[
        (flashes.trial_type.isin(['go', 'catch']) == False) & (flashes.consumption_licks == False)]
    abort_responses = non_trial_flashes[(non_trial_flashes.first_lick_in_bout == True)]
    abort_fa_rate = len(abort_responses) / float(len(non_trial_flashes))

    non_trial_flashes = flashes[
        (flashes.trial_type.isin(['go', 'catch']) == False) & (flashes.consumption_licks == False)]
    flashes_with_licks = non_trial_flashes[(non_trial_flashes.n_licks > 0)]
    fraction_flashes_with_licks = len(flashes_with_licks) / float(len(non_trial_flashes))

    if True in flashes.omitted.unique():
        omitted_flashes = flashes[(flashes.omitted == True) & (flashes.consumption_licks == False)]
        omitted_flash_rate = len(omitted_flashes[omitted_flashes.n_licks > 0]) / float(len(omitted_flashes))

        post_omitted_flashes = flashes[(flashes.flash_after_omitted == True) & (flashes.consumption_licks == False)]
        post_omitted_flash_rate = len(post_omitted_flashes[post_omitted_flashes.n_licks > 0]) / float(
            len(post_omitted_flashes))
    else:
        omitted_flash_rate = np.nan
        post_omitted_flash_rate = np.nan

    return hit_rate, catch_fa_rate, abort_fa_rate, fraction_flashes_with_licks, omitted_flash_rate, post_omitted_flash_rate


def get_response_latencies(flashes):
    hit_response_times = flashes[(flashes.trial_type == 'go') &
                                 (flashes.n_licks > 0)].post_flash_response_latency.values

    catch_fa_response_times = flashes[(flashes.trial_type == 'catch') &
                                      (flashes.n_licks > 0)].post_flash_response_latency.values

    abort_fa_response_times = flashes[(flashes.trial_type.isin(['go', 'catch']) == False) &
                                      (flashes.first_lick_in_bout == True) & (
                                              flashes.n_licks > 0)].post_flash_response_latency.values

    return hit_response_times, catch_fa_response_times, abort_fa_response_times


def generate_response_rate_summary_df(experiment_ids):
    from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset
    df_list = []
    for experiment_id in experiment_ids:
        print(experiment_id)
        dataset = VisualBehaviorOphysDataset(experiment_id, cache_dir=cache_dir)

        trials = dataset.all_trials.copy()
        trials['trial_duration'] = trials.endtime - trials.starttime
        n_aborted_trials = len(trials[trials.trial_type == 'aborted'])
        total_trials = len(trials)
        n_change_trials = len(trials[(trials.trial_type == 'go')])
        fraction_aborted_trials = n_aborted_trials / float(total_trials)
        fraction_time_aborted = trials[
                                    trials.trial_type == 'aborted'].trial_duration.sum() / trials.trial_duration.sum()
        fraction_time_engaged = trials[trials.reward_rate > 2].trial_duration.sum() / trials.trial_duration.sum()

        flashes = dataset.extended_stimulus_presentations.copy()
        trials = dataset.trials.copy()
        flashes = annotate_flashes(flashes, trials)
        flashes = flashes[flashes.reward_rate > 2]
        hit_rate, catch_fa_rate, abort_fa_rate, fraction_flashes_with_licks, omitted_flash_rate, post_omitted_flash_rate = get_response_rates(
            flashes)
        print(dataset.analysis_folder)

        hit_response_times, catch_fa_response_times, abort_fa_response_times = get_response_latencies(flashes)

        df_list.append([experiment_id, hit_rate, catch_fa_rate, abort_fa_rate, fraction_flashes_with_licks,
                        omitted_flash_rate, post_omitted_flash_rate,
                        hit_response_times, catch_fa_response_times, abort_fa_response_times,
                        n_aborted_trials, n_change_trials, total_trials,
                        fraction_aborted_trials, fraction_time_aborted, fraction_time_engaged])
        columns = ['experiment_id', 'hit_rate', 'catch_fa_rate', 'abort_fa_rate', 'fraction_flashes_with_licks',
                   'omitted_flash_response_rate', 'post_omitted_flash_response_rate',
                   'hit_response_times', 'catch_fa_response_times', 'abort_fa_response_times',
                   'n_aborted_trials', 'n_change_trials', 'total_trials', 'fraction_aborted_trials',
                   'fraction_time_aborted', 'fraction_time_engaged']
        response_rate_summary_df = pd.DataFrame(data=df_list, columns=columns)

    rdf = response_rate_summary_df.copy()
    annotated_rdf = pd.DataFrame()
    for experiment_id in experiment_ids:
        dataset = VisualBehaviorOphysDataset(experiment_id, cache_dir=cache_dir)
        df = rdf[rdf.experiment_id == experiment_id]
        df = ut.add_metadata_to_mean_df(df, dataset.metadata)
        annotated_rdf = pd.concat([annotated_rdf, df])
    response_rate_summary_df = annotated_rdf.copy()

    save_path = os.path.join(cache_dir, 'multi_session_summary_dfs', 'response_rate_summary_df_engaged.h5')
    response_rate_summary_df.to_hdf(save_path, key='df')


## boxplots

def adjust_box_widths(ax, fac):
    from matplotlib.patches import PathPatch
    # Adjust the withs of a seaborn-generated boxplot.
    for c in ax.get_children():
        if isinstance(c, PathPatch):
            p = c.get_path()
            verts = p.vertices
            verts_sub = verts[:-1]
            xmin = np.min(verts_sub[:, 0])
            xmax = np.max(verts_sub[:, 0])
            xmid = 0.5 * (xmin + xmax)
            xhalf = 0.5 * (xmax - xmin)
            xmin_new = xmid - fac * xhalf
            xmax_new = xmid + fac * xhalf
            verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new.copy()
            verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new.copy()
            for l in ax.lines:
                if np.all(l.get_xdata() == [xmin, xmax]):
                    l.set_xdata([xmin_new, xmax_new])


def plot_boxplot_for_condition(df, metric, condition='image_set', condition_values=['A', 'B', 'C', 'D'],
                               colors=sns.color_palette(), hue='cre_line', ylabel=None,
                               range=(0, 1), ax=None, save_figures=False, save_dir=None, folder=None):
    if ax is None:
        figsize = (4.5, 4.5)
        fig, ax = plt.subplots(figsize=figsize)
    ax = sns.boxplot(data=df, x=condition, y=metric, hue_order=np.sort(df[hue].unique()),
                     hue=hue, ax=ax, width=0.4, dodge=True, palette=colors)
    if ylabel is None:
        ax.set_ylabel('fraction of cells per session')
    else:
        ax.set_ylabel(ylabel)
    ax.set_ylim(range[0] - 0.05, range[1] + .05)
    ax.get_legend().remove()
    ax.set_title(metric)
    sns.despine(offset=10, trim=True)
    plt.gcf().subplots_adjust(top=0.85)
    plt.gcf().subplots_adjust(left=0.25)
    plt.gcf().subplots_adjust(right=0.85)
    plt.gcf().subplots_adjust(bottom=0.25)
    if save_figures:
        if show_stats:
            suffix = '_stats'
        else:
            suffix = ''
        save_figure(fig, figsize, save_dir, folder, metric + '_by_' + condition + '_box' + suffix)
    return ax


def plot_boxplot_and_swarm_for_condition(df, metric, condition='cre_line', condition_values=['A', 'B'],
                                         colors=sns.color_palette(), hue='image_set', hue_order=None, ylabel=None,
                                         xlabel=None, title=None, use_events=False, sample_type='sessions',
                                         plot_swarm=True, show_stats=True, range=(0, 1), ax=None, save_figures=False,
                                         save_dir=None, folder=None, suffix=''):
    df[metric] = pd.to_numeric(df[metric])
    if hue_order is None:
        hue_order = np.sort(df[hue].unique())
    if ax is None:
        figsize = (4.5, 4.5)
        fig, ax = plt.subplots(figsize=figsize)
    if plot_swarm:
        suffix = suffix + '_swarm'
        if show_stats:
            ax = sns.boxplot(data=df, x=condition, y=metric, hue_order=hue_order, order=condition_values,
                             hue=hue, ax=ax, dodge=True, color='white')  # palette=colors
            adjust_box_widths(ax, 0.5)
        ax = sns.swarmplot(data=df, x=condition, y=metric, order=condition_values,
                           size=4, ax=ax, hue=hue, hue_order=hue_order, palette=colors, dodge=True)  # color='.3',
    # swarm_cols = ax.collections
    #         for swarm in swarm_cols:
    #             swarm.set_facecolors([0.6,0.6,0.6])
    #         swarm.set_linewidths([0.5])
    #         swarm.set_edgecolors([0.2,0.2,0.2])
    else:
        ax = sns.boxplot(data=df, x=condition, y=metric, hue_order=hue_order, order=condition_values,
                         hue=hue, ax=ax, dodge=True, palette=colors)  # color='white'
        adjust_box_widths(ax, 0.8)
    if ylabel is None:
        ax.set_ylabel('fraction of cells per session')
    else:
        ax.set_ylabel(ylabel)
    if xlabel is None:
        if len(condition.split('_')) > 1:
            ax.set_xlabel(condition.split('_')[0] + ' ' + condition.split('_')[1])
        else:
            ax.set_xlabel(condition)
    else:
        ax.set_xlabel(xlabel)
    if title is None:
        ax.set_title(metric)
    else:
        ax.set_title(title)
    if use_events:
        ax.set_ylim(range[0] - 0.0005, range[1] + 0.0005)
    else:
        ax.set_ylim(range[0] - 0.05, range[1] + 0.05)
    ax.get_legend().remove()
    sns.despine(offset=10, trim=True)

    if show_stats:
        hue_names = hue_order
        sig_y_val = df[metric].max()

        stats_df = get_stats_for_conditions(df, metric, sample_type, condition, condition_values, hue, hue_names)
        label_df = get_label_df(ax, stats_df, condition, condition_values, hue, hue_names)
        ax = label_ax_with_stats(ax, label_df, stats_df, sig_y_val, condition, condition_values, hue, hue_names)

    plt.gcf().subplots_adjust(top=0.85)
    plt.gcf().subplots_adjust(left=0.25)
    plt.gcf().subplots_adjust(right=0.85)
    plt.gcf().subplots_adjust(bottom=0.25)
    if save_figures:
        if show_stats:
            suffix = suffix + '_stats'
        save_figure(fig, figsize, save_dir, folder, metric + '_by_' + condition + '_box' + suffix)

    return ax


### pointplots

def plot_pointplot_for_condition(df, metric, condition='cre_line', condition_values=['A', 'B'],
                                 colors=sns.color_palette(), hue='image_set', hue_order=None, ylabel=None, xlabel=None,
                                 title=None, use_events=False, sample_type='sessions',
                                 plot_swarm=True, show_stats=True, range=(0, 1), ax=None, save_figures=False,
                                 save_dir=None, folder=None, suffix=''):
    df[metric] = pd.to_numeric(df[metric])
    if hue_order is None:
        hue_order = np.sort(df[hue].unique())
    if ax is None:
        figsize = (4.5, 4.5)
        fig, ax = plt.subplots(figsize=figsize)
    if plot_swarm:
        suffix = suffix + '_swarm'
        ax = sns.swarmplot(data=df, x=condition, y=metric, dodge=0.4, order=condition_values,
                           size=3, ax=ax, hue=hue, hue_order=hue_order, color='gray', zorder=1)  # color='.3',
        if len(hue_order) < 3:
            dodge = 0.4
        else:
            dodge = 0.6
        ax = sns.pointplot(data=df, x=condition, y=metric, join=False, dodge=dodge, order=condition_values,
                           size=4, ax=ax, hue=hue, hue_order=hue_order, palette=colors, zorder=100)  # color='.3',
    else:
        ax = sns.pointplot(data=df, x=condition, y=metric, join=False, dodge=0.5, order=condition_values,
                           size=4, ax=ax, hue=hue, hue_order=hue_order, palette=colors, zorder=100)  # color='.3',
    if ylabel is None:
        ax.set_ylabel('fraction of cells per session')
    else:
        ax.set_ylabel(ylabel)
    if xlabel is None:
        if len(condition.split('_')) > 1:
            ax.set_xlabel(condition.split('_')[0] + ' ' + condition.split('_')[1])
        else:
            ax.set_xlabel(condition)
    else:
        ax.set_xlabel(xlabel)
    if title is None:
        ax.set_title(metric)
    else:
        ax.set_title(title)
    if use_events:
        ax.set_ylim(range[0] - 0.05, range[1] + .05)
    else:
        ax.set_ylim(range[0] - 0.0005, range[1] + .00005)
    ax.get_legend().remove()
    sns.despine(offset=10, trim=True)

    if show_stats:
        hue_names = hue_order
        sig_y_val = df[metric].max()

        stats_df = get_stats_for_conditions(df, metric, sample_type, condition, condition_values, hue, hue_names)
        label_df = get_label_df(ax, stats_df, condition, condition_values, hue, hue_names)
        ax = label_ax_with_stats(ax, label_df, stats_df, sig_y_val, condition, condition_values, hue, hue_names)

    plt.gcf().subplots_adjust(top=0.85)
    plt.gcf().subplots_adjust(left=0.25)
    plt.gcf().subplots_adjust(right=0.85)
    plt.gcf().subplots_adjust(bottom=0.25)
    if save_figures:
        #         l = ax.legend(title=condition, fontsize='small')
        #         plt.setp(l.get_title(),fontsize='small')
        if show_stats:
            suffix = suffix + '_stats'
        save_figure(fig, figsize, save_dir, folder, metric + '_by_' + condition + '_points' + suffix)

    return ax


def plot_points_for_image_sets_cre_lines(df, metric, points_range=(0, 1), xlabel=None, show_stats=False,
                                         label_kde=False, show_legend=False, bins=30, offset=False, ylabel='',
                                         sample_type='sessions',
                                         save_figures=False, save_dir=None, folder=None, use_events=False):
    condition = 'image_set'
    condition_values = ut.get_image_sets(df)
    hue = 'image_set'
    hue_order = condition_values
    colors = ut.get_colors_for_image_sets()
    cre_lines = ut.get_cre_lines(df)

    figsize = (4 * len(cre_lines), 4)
    fig, ax = plt.subplots(1, len(cre_lines), figsize=figsize, sharey=True)
    ax = ax.ravel()
    for i, cre_line in enumerate(cre_lines):
        tmp = df[df.cre_line == cre_line].copy()
        #         ax[i] = plot_hist_for_condition(tmp, metric, condition, condition_values, colors, cre_line, hist_range=hist_ranges[i],
        #                                         show_stats=show_stats, show_legend=show_legend, show_kde=show_kde, label_kde=label_kde, bins=bins,
        #                                         ax=ax[i], offset=offset, save_figures=False)

        ax[i] = plot_pointplot_for_condition(tmp, metric, condition, condition_values, colors, hue, hue_order=hue_order,
                                             ylabel=ylabel, xlabel='image set', title=title, range=points_range,
                                             ax=ax[i], use_events=use_events, sample_type=sample_type,
                                             plot_swarm=False, show_stats=False, save_figures=False, save_dir=save_dir,
                                             folder=folder)

        if xlabel:
            ax[i].set_xlabel(xlabel)
        else:
            ax[i].set_xlabel(metric)
        ax[i].set_ylabel('')
        ax[i].set_title(cre_line)
    ax[0].set_ylabel(ylabel)
    plt.gcf().subplots_adjust(wspace=.38)
    plt.gcf().subplots_adjust(bottom=.25)
    if save_figures:
        if show_stats:
            suffix = '_stats'
        else:
            suffix = ''
        save_figure(fig, figsize, save_dir, folder, metric + '_points' + suffix)


### histograms

def plot_histogram(values, label, color='k', hist_range=(0, 1), ax=None, offset=False, bins=30):
    results, edges = np.histogram(values, density=True, range=hist_range, bins=bins)
    binWidth = edges[1] - edges[0]
    if offset:
        ax.bar(edges[:-1] + binWidth, results * binWidth, binWidth, color=color, label=label, alpha=0.5)
    else:
        ax.bar(edges[:-1], results * binWidth, binWidth, color=color, label=label, alpha=0.5)
    return ax


def plot_hist_for_condition(df, metric, condition='image_set', condition_values=['A', 'B', 'C', 'D'],
                            colors=sns.color_palette(), cre_line=None, title=None, bins=30, hist_range=(0, 1),
                            show_stats=True, show_legend=False, show_kde=True, label_kde=False, offset=False,
                            ax=None, save_figures=False, save_dir=None, folder=None):
    if ax is None:
        figsize = (5, 5)
        fig, ax = plt.subplots(figsize=figsize)
    ax1 = ax.twinx()
    for i, condition_value in enumerate(condition_values):
        values = df[df[condition] == condition_value][metric].values
        ax = plot_histogram(values, bins=bins, offset=offset, label=condition_value, color=colors[i],
                            hist_range=hist_range, ax=ax)
        if show_kde:
            ax1 = sns.distplot(values, bins=bins, kde=True, hist=False, color=colors[i], ax=ax1,
                               kde_kws={'linewidth': 3})
        if not label_kde:
            ax1.set_yticklabels('')
            ax1.set_ylabel('')
            ax1.yaxis.set_ticks_position('none')
            sns.despine(ax=ax1, right=True)
        else:
            ax1.set_ylabel('density')
            sns.despine(right=False)
    ax.set_xlabel(metric)
    ax.set_ylabel('fraction of cells')
    if title is None:
        ax.set_title(cre_line)
    else:
        ax.set_title(title)

    if show_kde:
        ax.set_xlim(hist_range[0] - 0.1, hist_range[1] + 0.1)
    else:
        ax.set_xlim(hist_range[0], hist_range[1])

    if show_stats:
        if condition == 'image_set':
            stats_df = get_stats_for_cell_summary_image_sets(df, metric)
        elif condition == 'area':
            stats_df = get_stats_for_cell_summary_areas(df, metric)
        stats = stats_df[(stats_df.group == cre_line.split('-')[0])]
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        subax = inset_axes(ax,
                           width='{}%'.format(np.round(len(condition_values) * 0.06 * 100)),
                           height='{}%'.format(np.round(len(condition_values) * 0.06 * 100)),
                           loc=1)
        subax = sig_subplot(subax, stats, condition_values)
        #         subax.set_xticklabels([])
        subax.set_yticklabels(condition_values, fontsize=12)
        rotation = [90 if len(condition_values) < 3 else 0][0]
        subax.set_xticklabels(condition_values, rotation=rotation, fontsize=12)
        [t.set_color(colors[c]) for c, t in enumerate(subax.yaxis.get_ticklabels())]
        [t.set_color(colors[c]) for c, t in enumerate(subax.xaxis.get_ticklabels())]
        show_legend = False

    l = ax.legend(title=condition, fontsize='small')
    plt.setp(l.get_title(), fontsize='small')
    if not show_legend:
        ax.get_legend().remove()

    if save_figures:
        plt.gcf().subplots_adjust(top=0.85)
        plt.gcf().subplots_adjust(left=0.25)
        plt.gcf().subplots_adjust(right=0.85)
        plt.gcf().subplots_adjust(bottom=0.25)
        if show_stats:
            suffix = '_stats'
        else:
            suffix = ''
        save_figure(fig, figsize, save_dir, folder, metric + '_by_' + condition + '_' + cre_line + '_hist' + suffix)
    return ax


def plot_hist_for_image_sets_cre_lines(df, metric, hist_ranges=[(-1, 1), (-1, 1)], xlabel=None, show_kde=True,
                                       show_stats=True, title=None,
                                       label_kde=False, show_legend=False, bins=30, offset=False,
                                       save_figures=False, save_dir=None, folder=None):
    condition = 'image_set'
    condition_values = ut.get_image_sets(df)
    colors = ut.get_colors_for_image_sets()
    cre_lines = ut.get_cre_lines(df)

    figsize = (13, 4)
    fig, ax = plt.subplots(1, len(cre_lines), figsize=figsize, sharey=False)
    ax = ax.ravel()
    for i, cre_line in enumerate(cre_lines):
        tmp = df[df.cre_line == cre_line].copy()
        ax[i] = plot_hist_for_condition(tmp, metric, condition, condition_values, colors, cre_line,
                                        hist_range=hist_ranges[i],
                                        show_stats=show_stats, show_legend=show_legend, show_kde=show_kde,
                                        label_kde=label_kde, bins=bins,
                                        ax=ax[i], offset=offset, save_figures=False)
        if xlabel:
            ax[i].set_xlabel(xlabel)
        else:
            ax[i].set_xlabel(metric)
        ax[i].set_ylabel('')
        ax[i].set_title(cre_line)
    ax[0].set_ylabel('fraction of cells')
    if title:
        plt.suptitle(title, x=0.55, y=1.01, horizontalalignment='center', fontsize=18)
    plt.gcf().subplots_adjust(top=0.85)
    plt.gcf().subplots_adjust(right=0.85)
    plt.gcf().subplots_adjust(left=0.25)
    plt.gcf().subplots_adjust(wspace=0.3)
    if save_figures:
        if show_stats:
            suffix = '_stats'
        else:
            suffix = ''
        save_figure(fig, figsize, save_dir, folder, metric + '_hist' + suffix)


def plot_hist_for_areas(df, metric, hist_ranges=[(-1, 1), (-1, 1)], xlabel=None, show_kde=True, show_stats=True,
                        label_kde=False, show_legend=False, bins=30, offset=False,
                        save_figures=False, save_dir=None, folder=None):
    condition = 'area'
    condition_values = ut.get_visual_areas(df)
    colors = ut.get_colors_for_areas(df)
    cre_lines = ut.get_cre_lines(df)

    figsize = (4 * len(cre_lines), 4)
    fig, ax = plt.subplots(1, len(cre_lines), figsize=figsize, sharey=False)
    ax = ax.ravel()
    for i, cre_line in enumerate(cre_lines):
        tmp = df[df.cre_line == cre_line].copy()
        ax[i] = plot_hist_for_condition(tmp, metric, condition, condition_values, colors, cre_line,
                                        hist_range=hist_ranges[i],
                                        show_legend=show_legend, show_kde=show_kde, label_kde=label_kde,
                                        show_stats=show_stats, bins=bins,
                                        ax=ax[i], offset=offset, save_figures=False)
        if xlabel:
            ax[i].set_xlabel(xlabel)
        else:
            ax[i].set_xlabel(metric)
        ax[i].set_ylabel('')
        ax[i].set_title(cre_line)
    ax[0].set_ylabel('fraction of cells')
    plt.gcf().subplots_adjust(wspace=.38)
    plt.gcf().subplots_adjust(bottom=.25)
    if save_figures:
        if show_stats:
            suffix = '_stats'
        else:
            suffix = ''
        save_figure(fig, figsize, save_dir, folder, metric + '_hist_areas' + suffix)


### cumulative distributions

def plot_cdf_for_condition(df, metric, condition='image_set', condition_values=['A', 'B', 'C', 'D'],
                           colors=sns.color_palette(), show_stats=True, ylabel='fraction of cells',
                           cre_line=None, cdf_range=(0, 1), show_legend=True, ax=None, save_figures=False,
                           save_dir=None, folder=None):
    if ax is None:
        figsize = (5, 5)
        fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(cre_line)
    ax.set_xlabel(metric)
    ax.set_ylabel(ylabel)
    ax.set_xlim(cdf_range)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    #     ax.set_xlim(cdf_range)
    for i, condition_value in enumerate(condition_values):
        values = df[df[condition] == condition_value][metric].values
        values = values[~np.isnan(values)]
        if len(values) > 10:
            mean_value = np.nanmean(values)

            fraction = len(values[values < mean_value]) / float(len(values))
            ax.vlines(mean_value, ymin=ymin, ymax=fraction - 0.01, color=colors[i], linestyle='--', linewidth=1)

            #             ax.hlines(fraction, xmin=xmin, xmax=mean_value, color=colors[i], linestyle='--', linewidth=1)

            ax = sns.distplot(values, hist=False, hist_kws={'cumulative': True},
                              kde_kws={'cumulative': True, 'linewidth': 2}, ax=ax, color=colors[i],
                              label=condition_value)
        else:
            show_stats = False

    if show_stats:
        if condition == 'image_set':
            stats_df = get_stats_for_cell_summary_image_sets(df, metric)
        elif condition == 'area':
            stats_df = get_stats_for_cell_summary_areas(df, metric)
        elif condition == 'session_number':
            stats_df = get_stats_for_cell_summary_session_numbers(df, metric)
        stats = stats_df[(stats_df.group == cre_line.split('-')[0])]
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        size = 0.08 * len(condition_values)
        subpos = [xlim[1] - size - 0.1, ylim[0] + 0.25, size, size]
        #         subax = add_subplot_axes(ax,subpos)
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        subax = inset_axes(ax,
                           width='{}%'.format(np.round(len(condition_values) * 0.08 * 100)),
                           height='{}%'.format(np.round(len(condition_values) * 0.08 * 100)),
                           loc=4)
        subax = sig_subplot(subax, stats, condition_values)
        subax.set_xticklabels([])
        subax.set_yticklabels(condition_values, fontsize=12)
        [t.set_color(colors[c]) for c, t in enumerate(subax.yaxis.get_ticklabels())]
        show_legend = False

    l = ax.legend(title=condition, fontsize='small')
    plt.setp(l.get_title(), fontsize='small')
    if not show_legend:
        ax.get_legend().remove()
    plt.gcf().subplots_adjust(top=0.85)
    plt.gcf().subplots_adjust(left=0.25)
    plt.gcf().subplots_adjust(right=0.85)
    plt.gcf().subplots_adjust(bottom=0.25)
    if save_figures:
        if show_stats:
            suffix = '_stats'
        else:
            suffix = ''
        save_figure(fig, figsize, save_dir, folder,
                    metric + '_by_' + condition + '_' + cre_line.split('-')[0] + '_cdf' + suffix)
    return ax


def plot_cdf_for_image_sets(df, metric, cdf_ranges=[(0, 1), (0, 1)], xlabel=None, ylabel='fraction of cells',
                            show_legend=True, show_stats=True, save_figures=False, save_dir=None, folder=None):
    if 'image_set' in df.keys():
        condition = 'image_set'
        condition_values = ut.get_image_sets(df)
        colors = ut.get_colors_for_image_sets()
    else:
        condition = 'session_number'
        condition_values = np.sort(df[condition].unique())
        import visual_behavior.visualization.utils as utils
        colors = utils.get_colors_for_session_numbers()

    cre_lines = np.sort(df.cre_line.unique())
    figsize = (len(cre_lines)*4, 3.5)
    fig, ax = plt.subplots(1, len(cre_lines), figsize=figsize, sharex=False, sharey=True)
    for i, cre_line in enumerate(cre_lines):
        tmp = df[df.cre_line == cre_line].copy()
        ax[i] = plot_cdf_for_condition(tmp, metric, condition, condition_values, colors=colors, cre_line=cre_line,
                                       show_stats=show_stats, ylabel=ylabel,
                                       cdf_range=cdf_ranges[i], ax=ax[i], save_figures=False, save_dir=save_dir,
                                       folder=folder)
        ax[i].get_legend().remove()

        if xlabel is None:
            ax[i].set_xlabel(metric)
        else:
            ax[i].set_xlabel(xlabel)
    ax[1].set_ylabel('')
    l = ax[i].legend(title=condition, fontsize='x-small', bbox_to_anchor=(1.1,1))
    plt.setp(l.get_title(), fontsize='x-small')
    if (show_legend == False) or (show_stats == True):
        ax[i].legend_.remove()
    plt.gcf().subplots_adjust(wspace=.4)
    plt.gcf().subplots_adjust(top=0.85)
    plt.gcf().subplots_adjust(left=0.25)
    plt.gcf().subplots_adjust(right=0.85)
    plt.gcf().subplots_adjust(bottom=0.35)
    if save_figures:
        if show_stats:
            suffix = '_stats'
        else:
            suffix = ''
        sf.save_figure(fig, figsize, save_dir, folder, metric + '_by_' + condition + '_cdf' + suffix)


def plot_cdf_for_image_sets_stacked(df, metric, cdf_ranges=[(0, 1), (0, 1)], xlabel=None, show_legend=True,
                                    show_stats=True,
                                    save_figures=False, save_dir=None, folder=None):
    condition = 'image_set'
    condition_values = ut.get_image_sets(df)
    colors = ut.get_colors_for_image_sets()
    cre_lines = ut.get_cre_lines(df)

    figsize = (3.5, 8)
    fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=False, sharey=True)
    for i, cre_line in enumerate(cre_lines):
        tmp = df[df.cre_line == cre_line].copy()
        ax[i] = plot_cdf_for_condition(tmp, metric, condition, condition_values, colors=colors, cre_line=cre_line,
                                       show_stats=show_stats,
                                       cdf_range=cdf_ranges[i], ax=ax[i], save_figures=False, save_dir=save_dir,
                                       folder=folder)
        if xlabel is None:
            ax[i].set_xlabel(metric)
        else:
            ax[i].set_xlabel(xlabel)
            #     ax[1].set_ylabel('')
    l = ax[i].legend(title=condition, fontsize='x-small')
    plt.setp(l.get_title(), fontsize='x-small')
    if (show_legend == False) or (show_stats == True):
        ax[i].legend_.remove()
    plt.gcf().subplots_adjust(hspace=.6)
    plt.gcf().subplots_adjust(top=0.85)
    plt.gcf().subplots_adjust(left=0.25)
    plt.gcf().subplots_adjust(right=0.85)
    plt.gcf().subplots_adjust(bottom=0.25)
    if save_figures:
        if show_stats:
            suffix = '_stats'
        else:
            suffix = ''
        save_figure(fig, figsize, save_dir, folder, metric + '_by_' + condition + '_cdf_stacked' + suffix)


def plot_cdf_for_areas(df, metric, cdf_ranges=[(0, 1), (0, 1)], xlabel=None, show_legend=True, show_stats=True,
                       save_figures=False, save_dir=None, folder=None):
    condition = 'area'
    condition_values = get_visual_areas(df)
    colors = get_colors_for_areas(df)
    cre_lines = ut.get_cre_lines(df)

    figsize = (8, 3)
    fig, ax = plt.subplots(1, 2, figsize=figsize, sharey=True)
    for i, cre_line in enumerate(cre_lines):
        tmp = df[df.cre_line == cre_line].copy()
        ax[i] = plot_cdf_for_condition(tmp, metric, condition, condition_values, colors=colors, cre_line=cre_line,
                                       show_stats=show_stats,
                                       cdf_range=cdf_ranges[i], ax=ax[i], save_figures=False, save_dir=save_dir,
                                       folder=folder)

        if xlabel is None:
            ax[i].set_xlabel(metric)
        else:
            ax[i].set_xlabel(xlabel)
    ax[1].set_ylabel('')
    l = ax[i].legend(title=condition, fontsize='x-small')
    plt.setp(l.get_title(), fontsize='x-small')
    if (show_legend == False) or (show_stats == True):
        ax[i].legend_.remove()
    plt.gcf().subplots_adjust(wspace=.2)
    plt.gcf().subplots_adjust(top=0.85)
    plt.gcf().subplots_adjust(left=0.25)
    plt.gcf().subplots_adjust(right=0.85)
    plt.gcf().subplots_adjust(bottom=0.25)
    if save_figures:
        if show_stats:
            suffix = '_stats'
        else:
            suffix = ''
        save_figure(fig, figsize, save_dir, folder, metric + '_by_' + condition + '_' + '_cdf' + suffix)


def plot_cdf_for_areas_stacked(df, metric, cdf_ranges=[(0, 1), (0, 1)], xlabel=None, show_legend=True, show_stats=True,
                               save_figures=False, save_dir=None, folder=None):
    condition = 'area'
    condition_values = ut.get_visual_areas(df)
    colors = ut.get_colors_for_areas(df)
    cre_lines = ut.get_cre_lines(df)

    figsize = (3.5, 8)
    fig, ax = plt.subplots(2, 1, figsize=figsize, sharey=True)
    for i, cre_line in enumerate(cre_lines):
        tmp = df[df.cre_line == cre_line].copy()
        ax[i] = plot_cdf_for_condition(tmp, metric, condition, condition_values, colors=colors, cre_line=cre_line,
                                       show_stats=show_stats,
                                       cdf_range=cdf_ranges[i], ax=ax[i], save_figures=False, save_dir=save_dir,
                                       folder=folder)
        if xlabel is None:
            ax[i].set_xlabel(metric)
        else:
            ax[i].set_xlabel(xlabel)
            #     ax[1].set_ylabel('')
    l = ax[i].legend(title=condition, fontsize='x-small')
    plt.setp(l.get_title(), fontsize='x-small')
    if (show_legend == False) or (show_stats == True):
        ax[i].legend_.remove()
    plt.gcf().subplots_adjust(hspace=.6)
    plt.gcf().subplots_adjust(top=0.85)
    plt.gcf().subplots_adjust(left=0.25)
    plt.gcf().subplots_adjust(right=0.85)
    plt.gcf().subplots_adjust(bottom=0.25)
    if save_figures:
        if show_stats:
            suffix = '_stats'
        else:
            suffix = ''
        save_figure(fig, figsize, save_dir, folder, metric + '_by_' + condition + '_cdf_stacked' + suffix)


### barplots

def plot_fraction_cells_over_threshold(cell_df, metric, threshold, condition='image_set',
                                       condition_values=['A', 'B', 'C', 'D'],
                                       colors=sns.color_palette(), less_than=False, title=None, sharey=True, ymax=None,
                                       save_figures=False, save_dir=None, folder=None, suffix=''):
    df = cell_df.copy()
    cre_lines = ut.get_cre_lines(df)
    figsize = (5.75, 2.75)
    fig, ax = plt.subplots(1, 2, figsize=figsize, sharey=sharey, sharex=True)
    ax = ax.ravel()
    for i, cre_line in enumerate(cre_lines):
        for c, condition_value in enumerate(condition_values):
            t = df[(df.cre_line == cre_line) & (df[condition] == condition_value)].copy()
            if c == 0:
                x = 0
            else:
                x = x + 0.2
            if less_than:
                tmp = t[(t[metric] < threshold)].copy()
            else:
                tmp = t[(t[metric] > threshold)].copy()
            fraction = len(tmp) / float(len(t))
            ax[i].bar(x, fraction, width=0.1, color=colors[c])
        ax[i].set_title(cre_line)
        ax[i].set_xticks(np.arange(0, len(condition_values) * 0.2, 0.2))
        ax[i].set_xticklabels(condition_values)
        ax[i].set_xlabel('image set')
        if ymax:
            ax[i].set_ylim(0, ymax)
    ax[0].set_ylabel('fraction of cells')
    if title:
        plt.suptitle(title, x=0.52, y=1.0,
                     horizontalalignment='center', fontsize=16)
    elif less_than:
        plt.suptitle('fraction of cells with ' + metric + ' < ' + str(threshold), x=0.52, y=1.0,
                     horizontalalignment='center', fontsize=16)
    else:
        plt.suptitle('fraction of cells with ' + metric + ' > ' + str(threshold), x=0.52, y=1.0,
                     horizontalalignment='center', fontsize=16)
    plt.gcf().subplots_adjust(left=0.15)
    #     plt.gcf().subplots_adjust(right=0.9)
    plt.gcf().subplots_adjust(top=0.8)
    plt.gcf().subplots_adjust(bottom=0.25)
    if sharey:
        plt.gcf().subplots_adjust(wspace=.2)
    else:
        plt.gcf().subplots_adjust(wspace=.3)
    if save_figures:
        save_figure(fig, figsize, save_dir, folder, metric + '_' + condition + '_fraction' + suffix)


def plot_fraction_cells_over_threshold_stacked(cell_df, metric, threshold, condition='image_set',
                                               condition_values=['A', 'B', 'C', 'D'], colors=sns.color_palette(),
                                               less_than=False, ymax=None,
                                               suffix='', sharey=True, title=None, save_figures=False, save_dir=None,
                                               folder=None):
    df = cell_df.copy()
    cre_lines = ut.get_cre_lines(df)
    figsize = (3.5, 5.5)
    fig, ax = plt.subplots(2, 1, figsize=figsize, sharey=sharey, sharex=True)
    ax = ax.ravel()
    for i, cre_line in enumerate(cre_lines):
        for c, condition_value in enumerate(condition_values):
            t = df[(df.cre_line == cre_line) & (df[condition] == condition_value)]
            if c == 0:
                x = 0
            else:
                x = x + 0.2
            if less_than:
                tmp = t[(t[metric] < threshold)]
            else:
                tmp = t[(t[metric] > threshold)]
            fraction = len(tmp) / float(len(t))
            ax[i].bar(x, fraction, width=0.1, color=colors[c])
        ax[i].set_title(cre_line)
        ax[i].set_xticks(np.arange(0, len(condition_values) * 0.2, 0.2))
        ax[i].set_xticklabels(condition_values)
        ax[i].set_ylabel('fraction of cells')
        if ymax:
            ax[i].set_ylim(0, ymax)
    ax[1].set_xlabel('image set')
    plt.suptitle(title, x=0.63, y=.94,
                 horizontalalignment='center', fontsize=16)
    plt.gcf().subplots_adjust(left=0.35)
    #     plt.gcf().subplots_adjust(right=0.9)
    plt.gcf().subplots_adjust(top=0.8)
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.gcf().subplots_adjust(hspace=.3)

    if save_figures:
        save_figure(fig, figsize, save_dir, folder, metric + '_' + condition + '_fraction_stacked' + suffix)


def plot_percent_cells_over_threshold(cell_df, metric, threshold, condition='image_set',
                                      condition_values=['A', 'B', 'C', 'D'],
                                      colors=sns.color_palette(), less_than=False, title=None, sharey=True, ymax=None,
                                      save_figures=False, save_dir=None, folder=None, suffix=''):
    df = cell_df.copy()
    cre_lines = ut.get_cre_lines(df)
    figsize = (5.75, 2.75)
    fig, ax = plt.subplots(1, 2, figsize=figsize, sharey=sharey, sharex=True)
    ax = ax.ravel()
    for i, cre_line in enumerate(cre_lines):
        for c, condition_value in enumerate(condition_values):
            t = df[(df.cre_line == cre_line) & (df[condition] == condition_value)]
            if c == 0:
                x = 0
            else:
                x = x + 0.2
            if less_than:
                tmp = t[(t[metric] < threshold)]
            else:
                tmp = t[(t[metric] > threshold)]
            fraction = len(tmp) / float(len(t))
            ax[i].bar(x, fraction * 100, width=0.1, color=colors[c])
        ax[i].set_title(cre_line)
        ax[i].set_xticks(np.arange(0, len(condition_values) * 0.2, 0.2))
        ax[i].set_xticklabels(condition_values)
        ax[i].set_xlabel('image set')
        if ymax:
            ax[i].set_ylim(0, ymax)
    ax[0].set_ylabel('percent of cells')
    if title:
        plt.suptitle(title, x=0.52, y=1.0,
                     horizontalalignment='center', fontsize=16)
    elif less_than:
        plt.suptitle('% of cells with ' + metric + ' < ' + str(threshold), x=0.52, y=1.0,
                     horizontalalignment='center', fontsize=16)
    else:
        plt.suptitle('% of cells with ' + metric + ' > ' + str(threshold), x=0.52, y=1.0,
                     horizontalalignment='center', fontsize=16)
    plt.gcf().subplots_adjust(left=0.15)
    #     plt.gcf().subplots_adjust(right=0.9)
    plt.gcf().subplots_adjust(top=0.8)
    plt.gcf().subplots_adjust(bottom=0.25)
    if sharey:
        plt.gcf().subplots_adjust(wspace=.2)
    else:
        plt.gcf().subplots_adjust(wspace=.3)
    if save_figures:
        save_figure(fig, figsize, save_dir, folder, metric + '_' + condition + '_percent' + suffix)


def plot_percent_cells_over_threshold_stacked(cell_df, metric, threshold, condition='image_set',
                                              condition_values=['A', 'B', 'C', 'D'], colors=sns.color_palette(),
                                              less_than=False, ymax=None,
                                              suffix='', sharey=True, title=None, save_figures=False, save_dir=None,
                                              folder=None):
    df = cell_df.copy()
    cre_lines = ut.get_cre_lines(df)
    figsize = (3.5, 5.5)
    fig, ax = plt.subplots(2, 1, figsize=figsize, sharey=sharey, sharex=True)
    ax = ax.ravel()
    for i, cre_line in enumerate(cre_lines):
        for c, condition_value in enumerate(condition_values):
            t = df[(df.cre_line == cre_line) & (df[condition] == condition_value)].copy()
            if c == 0:
                x = 0
            else:
                x = x + 0.2
            if less_than:
                tmp = t[(t[metric] < threshold)].copy()
            else:
                tmp = t[(t[metric] > threshold)].copy()
            fraction = len(tmp) / float(len(t))
            ax[i].bar(x, fraction * 100, width=0.1, color=colors[c])
        ax[i].set_title(cre_line)
        ax[i].set_xticks(np.arange(0, len(condition_values) * 0.2, 0.2))
        ax[i].set_xticklabels(condition_values)
        ax[i].set_ylabel('% of cells')
        if ymax:
            ax[i].set_ylim(0, ymax)
    ax[1].set_xlabel('image set')
    plt.suptitle(title, x=0.63, y=.94,
                 horizontalalignment='center', fontsize=16)
    plt.gcf().subplots_adjust(left=0.45)
    plt.gcf().subplots_adjust(right=0.9)
    plt.gcf().subplots_adjust(top=0.8)
    plt.gcf().subplots_adjust(bottom=0.25)
    plt.gcf().subplots_adjust(hspace=.3)

    if save_figures:
        save_figure(fig, figsize, save_dir, folder, metric + '_' + condition + '_percent_stacked' + suffix)


def plot_mean_value_barplot(cell_df, metric, condition='image_set', condition_values=['A', 'B', 'C', 'D'],
                            ylabel='metric',
                            colors=sns.color_palette(), title=None, sharey=True, ymax=None,
                            save_figures=False, save_dir=None, folder=None, suffix=''):
    from scipy.stats import sem as compute_sem
    df = cell_df.copy()
    cre_lines = ut.get_cre_lines(df)
    figsize = (5.75, 2.75)
    fig, ax = plt.subplots(1, 2, figsize=figsize, sharey=sharey, sharex=True)
    ax = ax.ravel()
    for i, cre_line in enumerate(cre_lines):
        for c, condition_value in enumerate(condition_values):
            t = df[(df.cre_line == cre_line) & (df[condition] == condition_value)]
            if c == 0:
                x = 0
            else:
                x = x + 0.2
            values = t[metric].values
            mean = np.nanmean(values)
            values = values[np.isnan(values) == False]
            sem = compute_sem(values)
            ax[i].bar(x, mean, width=0.1, color=colors[c])
            ax[i].errorbar(x, mean, yerr=sem, color='k')
        ax[i].set_title(cre_line)
        ax[i].set_xticks(np.arange(0, len(condition_values) * 0.2, 0.2))
        ax[i].set_xticklabels(condition_values)
        ax[i].set_xlabel('image set')
        if ymax:
            ax[i].set_ylim(0, ymax)
    ax[0].set_ylabel('mean ' + ylabel)
    if title:
        plt.suptitle(title, x=0.52, y=1.0,
                     horizontalalignment='center', fontsize=16)
    plt.gcf().subplots_adjust(left=0.15)
    #     plt.gcf().subplots_adjust(right=0.9)
    plt.gcf().subplots_adjust(top=0.8)
    plt.gcf().subplots_adjust(bottom=0.25)
    if sharey:
        plt.gcf().subplots_adjust(wspace=.2)
    else:
        plt.gcf().subplots_adjust(wspace=.3)
    if save_figures:
        save_figure(fig, figsize, save_dir, folder, metric + '_' + condition + '_mean_bar_plot' + suffix)


### summary figures by image set and area

def generate_figures_for_session_metric_image_sets(session_summary_df, metric, range=(0, 1), points_range=(0, 1),
                                                   ylabel='fraction of cells', xlabel=None, use_events=False,
                                                   title=None, plot_swarm=False, show_stats=True, save_figures=False,
                                                   save_dir=None, folder=None):
    hue = 'image_set'
    hue_order = ut.get_image_sets(session_summary_df)
    colors = ut.get_colors_for_image_sets()

    df = session_summary_df.copy()
    df[metric] = pd.to_numeric(df[metric])

    df['cre line'] = [cre_line.split('-')[0] for cre_line in df.cre_line.values]
    condition = 'cre line'
    condition_values = np.sort(df['cre line'].unique())

    plot_boxplot_and_swarm_for_condition(df, metric, condition, condition_values, colors, hue, hue_order=hue_order,
                                         plot_swarm=plot_swarm, ylabel=ylabel, xlabel=xlabel, title=title, range=range,
                                         ax=None, use_events=use_events,
                                         save_figures=save_figures, save_dir=save_dir, folder=folder,
                                         show_stats=show_stats)

    plot_pointplot_for_condition(df, metric, condition, condition_values, colors, hue, hue_order=hue_order,
                                 ylabel=ylabel, xlabel=xlabel, title=title, range=points_range, ax=None,
                                 use_events=use_events,
                                 plot_swarm=True, show_stats=False, save_figures=save_figures, save_dir=save_dir,
                                 folder=folder)


def generate_figures_for_cell_summary_image_sets(cell_summary_df, metric, cdf_ranges=[(0, 1), (0, 1)], xlabel=None,
                                                 show_stats=True, ylabel='', title='',
                                                 hist_ranges=[(-5, 3), (-1, 1)], points_range=(0, 1), show_kde=True,
                                                 show_legend=True, bins=30, label_kde=False,
                                                 ytitle='', stacked=True, offset=False, save_figures=False,
                                                 save_dir=None, folder=None):
    df = cell_summary_df.copy()
    if stacked:
        plot_cdf_for_image_sets_stacked(df, metric, cdf_ranges=cdf_ranges, show_legend=show_legend, xlabel=xlabel,
                                        show_stats=show_stats,
                                        save_figures=save_figures, save_dir=save_dir, folder=folder)
    else:
        plot_cdf_for_image_sets(df, metric, cdf_ranges=cdf_ranges, show_legend=show_legend, xlabel=xlabel,
                                show_stats=show_stats,
                                save_figures=save_figures, save_dir=save_dir, folder=folder)

    plot_hist_for_image_sets_cre_lines(df, metric, hist_ranges=hist_ranges, xlabel=xlabel, bins=bins, offset=offset,
                                       show_legend=show_legend, show_kde=show_kde, label_kde=label_kde,
                                       show_stats=show_stats,
                                       save_figures=save_figures, save_dir=save_dir, folder=folder)


def generate_figures_for_cell_summary_areas(cell_summary_df, metric, cdf_ranges=[(-0.05, 1.05), (-0.05, 1.05)],
                                            xlabel=None,
                                            hist_ranges=[(-5, 3), (-1, 1)], show_kde=True, show_legend=True,
                                            show_stats=True, bins=30, label_kde=False,
                                            offset=False, save_figures=False, save_dir=None, folder=None):
    df = cell_summary_df.copy()
    plot_cdf_for_areas_stacked(df, metric, cdf_ranges=cdf_ranges, show_legend=show_legend, xlabel=xlabel,
                               show_stats=show_stats,
                               save_figures=save_figures, save_dir=save_dir, folder=folder)

    condition = 'area'
    condition_values = ut.get_visual_areas(df)
    colors = ut.get_colors_for_areas(df)

    plot_hist_for_areas(df, metric, hist_ranges=hist_ranges, xlabel=xlabel, bins=bins, offset=offset,
                        show_legend=show_legend, show_kde=show_kde, label_kde=label_kde, show_stats=show_stats,
                        save_figures=save_figures, save_dir=save_dir, folder=folder)


def generate_figures_for_session_metric_areas(session_summary_df, metric, range=(0, 1), points_range=(0, 1),
                                              ylabel='fraction of cells', xlabel=None,
                                              show_stats=False, title=None, plot_swarm=False, save_figures=False,
                                              save_dir=None, folder=None):
    condition = 'image_set'
    condition_values = ut.get_image_sets(session_summary_df)
    hue = 'area'
    hue_order = ut.get_visual_areas(session_summary_df)
    colors = ut.get_colors_for_areas(session_summary_df)
    cre_lines = ut.get_cre_lines(session_summary_df)

    df = session_summary_df.copy()
    df[metric] = pd.to_numeric(df[metric])

    figsize = (4.25 * len(cre_lines), 4)
    fig, ax = plt.subplots(1, len(cre_lines), figsize=figsize, sharey=True)
    ax = ax.ravel()
    for i, cre_line in enumerate(cre_lines):
        tmp = df[df.cre_line == cre_line].copy()
        ax[i] = plot_pointplot_for_condition(tmp, metric, condition, condition_values, colors, hue, hue_order=hue_order,
                                             ylabel=ylabel, xlabel=xlabel, title=title, range=points_range, ax=ax[i],
                                             plot_swarm=True, show_stats=False, save_figures=False, save_dir=save_dir,
                                             folder=folder)
        if xlabel:
            ax[i].set_xlabel(xlabel)
        else:
            if len(condition.split('_')) > 1:
                ax[i].set_xlabel(condition.split('_')[0] + ' ' + condition.split('_')[1])
            else:
                ax[i].set_xlabel(condition)
        ax[i].set_ylabel('')
        ax[i].set_title(cre_line)
    ax[0].set_ylabel(ylabel)
    #     ax[1].legend()
    plt.gcf().subplots_adjust(wspace=.38)
    plt.gcf().subplots_adjust(bottom=.25)
    if save_figures:
        suffix = ''
        if plot_swarm:
            suffix = suffix + '_swarm'
        if show_stats:
            suffix = suffix + '_stats'
        save_figure(fig, figsize, save_dir, folder, metric + '_by_' + condition + '_points' + suffix)


### heatmaps

def plot_tuning_curve_heatmap(df, cre_line, image_set=None, vmax=0.3, title=None, ax=None, save_dir=None, folder=None, use_events=False,
                              colorbar=True, horizontal_legend=False, include_omitted=False):
    if 'image_set' is None:
        if 'image_set' in df.keys():
            image_set = df.image_set.unique()[0]
        else:
            image_set = df.session_number.unique()[0]
    if 'image_name' in df.keys():
        image_name = 'image_name'
        suffix = '_flashes'
        if ('omitted' in df.image_name.unique()) and (include_omitted is False):
            df = df[df.image_name != 'omitted']
    else:
        image_name = 'change_image_name'
        suffix = '_trials'
    if use_events:
        label = 'mean response'
        suffix = suffix + '_events'
    else:
        label = 'mean dF/F'
        suffix = suffix
    images = np.sort(df[image_name].unique())
    cell_list = []
    for image in images:
        tmp = df[(df[image_name] == image) & (df.pref_stim == True)]
        order = np.argsort(tmp.mean_response.values)[::-1]
        cell_ids = list(tmp.cell_specimen_id.values[order])
        cell_list = cell_list + cell_ids
    response_matrix = np.empty((len(cell_list), len(images)))
    for i, cell in enumerate(cell_list):
        responses = []
        for image in images:
            response = df[(df.cell_specimen_id == cell) & (df[image_name] == image)].mean_response.values[0]
            responses.append(response)
        response_matrix[i, :] = np.asarray(responses)
    if ax is None:
        figsize = (5, 8)
        fig, ax = plt.subplots(figsize=figsize)
        fig.tight_layout()
    if horizontal_legend:
        ax = sns.heatmap(response_matrix, cmap='magma', linewidths=0, linecolor='white', square=False,
                         vmin=0, vmax=vmax, robust=True, cbar=colorbar,
                         cbar_kws={"drawedges": False, "shrink": 1, "label": label, "orientation": "horizontal"}, ax=ax)
    else:
        ax = sns.heatmap(response_matrix, cmap='magma', linewidths=0, linecolor='white', square=False,
                         vmin=0, vmax=vmax, robust=True, cbar=colorbar,
                         cbar_kws={"drawedges": False, "shrink": 1, "label": label}, ax=ax)
    ax.set_title(cre_line, va='bottom', ha='center')
    ax.set_xticklabels(images, rotation=90, fontsize=12)
    ax.set_ylabel('cells')
    ax.set_yticks((0, response_matrix.shape[0]))
    ax.set_yticklabels((0, response_matrix.shape[0]), fontsize=12)
    if save_dir:
        plt.suptitle('image set ' + image_set, x=0.46, y=0.99, fontsize=18)
        fig.tight_layout()
        plt.gcf().subplots_adjust(top=0.9)
        save_figure(fig, figsize, save_dir, folder, 'tuning_curve_heatmap_' + cre_line + '_' + image_set)
    return ax


def plot_response_across_conditions_population(df, condition='image_set', conditions=['A', 'B', 'C', 'D'],
                                               plot_flashes=True,
                                               window=[-0.5, 0.75], save_figures=False, colors=None, autoscale=False,
                                               save_dir=None, folder=None, ax=None, pref_stim=True, omitted=False,
                                               frame_rate=30., show_variability=False):
    image_set = df.image_set.unique()[0]
    cre_line = df.cre_line.unique()[0]
    colors = colors[::-1]
    if pref_stim:
        df = df[df.pref_stim == True].copy()
    if ax is None:
        figsize = (5, 5)
        fig, ax = plt.subplots(figsize=figsize)
    if np.abs(window[0]) >= 1:
        interval_sec = 1
    else:
        interval_sec = 0.5
    for c, condition_value in enumerate(conditions[::-1]):
        tmp = df[df[condition] == condition_value]
        if colors is None:
            image_lookup = get_image_color_lookup(mdf)
            image_names = df[df.image_set == image_set].image_name.unique()
            colors = get_colors_for_image_names(image_names, image_lookup)
        traces = tmp.mean_trace.values
        traces = np.asarray([trace for trace in traces if len(trace) == len(traces[0])])
        trace = np.nanmean(traces, axis=0)

        if show_variability:
            ax = sf.plot_mean_trace_with_variability(traces, frame_rate, ylabel='dF/F', label=condition_value,
                                                     color=colors[c], interval_sec=interval_sec,
                                                     xlims=window, ax=ax, flashes=False)
        else:
            ax = sf.plot_mean_trace(traces, frame_rate, legend_label=condition_value, color=colors[c],
                                    interval_sec=interval_sec,
                                    xlims=window, ax=ax)

    if plot_flashes:
        ax = psf.plot_flashes_on_trace(ax, alpha=0.3, window=window, omitted=omitted, frame_rate=frame_rate)
        xticks, xticklabels = sf.get_xticks_xticklabels(trace, frame_rate, interval_sec=interval_sec, window=window)
    ax.set_xticks(xticks)
    if interval_sec >= 1:
        ax.set_xticklabels([int(x) for x in xticklabels])
    else:
        ax.set_xticklabels(xticklabels)
    ax.set_xlim(0, (np.abs(window[0]) + window[1]) * frame_rate)
    ax.legend(bbox_to_anchor=(1.1, 1), title=condition)
    if not autoscale:
        ymin, ymax = ax.get_ylim()
        if ymin > 0:
            ax.set_ylim(0, ymax * 1.2)
        else:
            ax.set_ylim(ymin * 1.2, ymax * 1.2)
    ax.set_title(image_set)
    if save_figures:
        plt.gcf().subplots_adjust(top=0.9)
        plt.gcf().subplots_adjust(left=0.25)
        plt.gcf().subplots_adjust(bottom=0.2)
        # plt.gcf().subplots_adjust(wspace=0.4)
        psf.save_figure(fig, figsize, save_dir, folder,
                        str(int(cdf.experiment_id.unique()[0])) + '_' + str(int(cell_specimen_id)))
        plt.close()
    return ax


def plot_population_activity_heatmap(df, vmax=0.01, colorbar=False, ax=None, save_dir=None, folder=None, interval_sec=2,
                                     use_events=False, window=[-3, 3]):
    if use_events:
        label = 'mean response'
        suffix = '_events'
    else:
        label = 'mean dF/F'
        suffix = ''
    image_set = df.image_set.unique()[0]
    cre_line = df.cre_line.unique()[0]
    if ax is None:
        figsize = (4, 7)
        fig, ax = plt.subplots(figsize=figsize)

    order = np.argsort(df.mean_response.values)[::-1]
    cells = list(df.cell_specimen_id.values[order])

    len_trace = len(df.mean_trace.values[0])
    response_array = np.empty((len(cells), len_trace))
    for x, cell in enumerate(cells):
        cdf = df[df.cell_specimen_id == cell]
        if len(cdf) >= 1:
            trace = cdf.mean_trace.values[0]
        else:
            trace = np.empty((len_trace))
            trace[:] = np.nan
        response_array[x, :] = trace
    sns.heatmap(data=response_array, vmin=0, vmax=vmax, ax=ax, cmap='magma', cbar=colorbar,
                cbar_kws={'label': label})
    xticks, xticklabels = sf.get_xticks_xticklabels(trace, 31., interval_sec=interval_sec, window=window)
    ax.set_xticks(xticks)
    ax.set_xticklabels([int(xticklabel) for xticklabel in xticklabels])
    ax.set_yticks((0, response_array.shape[0]))
    ax.set_yticklabels((0, response_array.shape[0]), fontsize=14)
    ax.set_xlabel('time after change (s)', fontsize=16)
    ax.set_title(cre_line)
    ax.set_ylabel('cells')
    if save_dir:
        plt.suptitle('image set ' + image_set, x=0.59, y=.99, fontsize=18)
        fig.tight_layout()
        plt.gcf().subplots_adjust(top=0.9)
        save_figure(fig, figsize, save_dir, folder,
                    'population_activity_heatmap_' + cre_line + '_' + image_set + '_' + suffix)
    return ax


def plot_scatter_for_cre_line(df, cre_line, x, y, x_label, y_label, lim=None, xlim=None, ylim=(-1, 1),
                              annotate=False, show_fit=True, ax=None, legend=True, fit_image_sets=True,
                              save_figures=False, save_dir=None, folder='response_dynamics', suffix=''):
    import scipy.stats as st
    df = df[df.cre_line == cre_line]
    colors = ut.get_colors_for_image_sets()
    if ax is None:
        figsize = (7.3, 4)
        fig, ax = plt.subplots(figsize=figsize)
    ax = sns.scatterplot(data=df, x=x, y=y, hue='image_set',
                         ax=ax, palette=colors, legend=False, s=10, linewidth=0)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    if lim:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    elif xlim:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    xmin, xmax = ax.get_xlim()
    if fit_image_sets:
        for c, image_set in enumerate(ut.get_image_sets(df)):
            x_vals = df[(df.image_set == image_set) & (df[x].isnull() == False) & (df[y].isnull() == False)][x].values
            y_vals = df[(df.image_set == image_set) & (df[y].isnull() == False) & (df[x].isnull() == False)][y].values
            slope, intercept, r_value, p_value, std_err = st.linregress(x_vals, y_vals)
            x_axis = np.arange(xmin, xmax, 0.01)
            line_values = np.asarray([slope * i + intercept for i in x_axis])
            r = np.round(r_value, 2)
            p = np.round(p_value, 6)
            if p_value < 0.005:
                p_string = ', p<0.005'
            else:
                p_string = ', p = ' + str(p)
            if show_fit:
                ax.plot(x_axis, line_values, '-', color=colors[c], label=image_set + ', r: ' + str(r) + p_string)
    else:
        x_vals = df[(df[x].isnull() == False) & (df[y].isnull() == False)][x].values
        y_vals = df[(df[y].isnull() == False) & (df[x].isnull() == False)][y].values
        slope, intercept, r_value, p_value, std_err = st.linregress(x_vals, y_vals)
        x_axis = np.arange(xmin, xmax, 0.01)
        line_values = np.asarray([slope * i + intercept for i in x_axis])
        r = np.round(r_value, 2)
        p = np.round(p_value, 6)
        if p_value < 0.005:
            p_string = ', p<0.005'
        else:
            p_string = ', p = ' + str(p)
        if show_fit:
            ax.plot(x_axis, line_values, '-', color='gray', label='r: ' + str(r) + p_string)
    ax.set_title(cre_line, fontsize=16)
    if annotate and lim:
        ymin, ymax = ax.get_ylim()
        ax.vlines(0, ymin=ymin, ymax=ymax, color='gray', linestyle='--')
        xmin, xmax = ax.get_xlim()
        ax.hlines(0, xmin=xmin, xmax=xmax, color='gray', linestyle='--')
        ax.text(xmin + (lim * .1), ymax - (lim * .2), s='Q1', fontsize=14)
        ax.text(xmax - (lim * .3), ymax - (lim * .2), s='Q2', fontsize=14)
        ax.text(xmin + (lim * .1), ymin + (lim * .1), s='Q3', fontsize=14)
        ax.text(xmax - (lim * .3), ymin + (lim * .1), s='Q4', fontsize=14)
    else:
        print('cannot annotate if lim not provided')
    if legend:
        ax.legend(bbox_to_anchor=(1.2, 1), fontsize=12)
    if save_figures:
        plt.gcf().subplots_adjust(top=0.75)
        plt.gcf().subplots_adjust(right=0.6)
        plt.gcf().subplots_adjust(bottom=0.2)
        plt.gcf().subplots_adjust(left=0.3)
        sf.save_figure(fig, figsize, save_dir, folder, cre_line + '_' + x + '_' + y + suffix)
    return ax


def multivariateGrid(df, col_x, col_y, col_k, lim=None, colors=sns.color_palette(), scatter_alpha=.5):
    import scipy.stats as st
    def colored_scatter(x, y, c=None):
        def scatter(*args, **kwargs):
            args = (x, y)
            if c is not None:
                kwargs['c'] = c
            kwargs['alpha'] = scatter_alpha
            plt.scatter(*args, **kwargs)

        return scatter

    g = sns.JointGrid(x=col_x, y=col_y, data=df, height=4.5)
    color = None
    legends = []
    for name, df_group in df.groupby(col_k):
        legends.append(name)
        name_position = np.where(np.sort(df[col_k].unique()) == name)[0][0]
        if name_position == 0:
            zorder = 100
        else:
            zorder = 0
        color = colors[name_position]
        g.plot_joint(colored_scatter(df_group[col_x], df_group[col_y], color), zorder=zorder, s=10)

        sns.distplot(df_group[col_x].values, ax=g.ax_marg_x, color=color, bins=20)
        sns.distplot(df_group[col_y].values, ax=g.ax_marg_y, color=color, bins=20, vertical=True)
    plt.legend(legends, fontsize='xx-small')

    if lim:
        g.ax_joint.set_xlim(-lim, lim)
        g.ax_joint.set_ylim(-lim, lim)
    x_vals = df[(df[col_x].isnull() == False) & (df[col_y].isnull() == False)][col_x].values
    y_vals = df[(df[col_y].isnull() == False) & (df[col_x].isnull() == False)][col_y].values
    slope, intercept, r_value, p_value, std_err = st.linregress(x_vals, y_vals)
    xmin, xmax = g.ax_joint.get_xlim()
    ymin, ymax = g.ax_joint.get_ylim()
    x_axis = np.arange(xmin, xmax, 0.01)
    line_values = np.asarray([slope * i + intercept for i in x_axis])
    r = np.round(r_value, 2)
    p = np.round(p_value, 6)
    if p_value < 0.005:
        p_string = ', p<0.005'
    else:
        p_string = ', p = ' + str(p)
    g.ax_joint.plot(x_axis, line_values, '-', color='gray')  # , label='r: '+str(r)+p_string)
    g.ax_joint.text(x=xmin + (xmax * 0.05), y=ymin + (ymax * 0.07), s='r: ' + str(r) + p_string, fontsize=14)

    return plt


def plot_fraction_cells_per_response_type(cell_df, response_types, condition='image_set',
                                          condition_values=['A', 'B', 'C', 'D'],
                                          colors=sns.color_palette(), less_than=False, title=None, sharey=True,
                                          ymax=None,
                                          save_figures=False, save_dir=None, folder=None, suffix=''):
    df = cell_df.copy()
    cre_lines = ut.get_cre_lines(df)
    figsize = (6, 2.75)
    fig, ax = plt.subplots(1, 2, figsize=figsize, sharey=sharey, sharex=True)
    ax = ax.ravel()
    for i, cre_line in enumerate(cre_lines):
        for c, condition_value in enumerate(condition_values):
            for r, response_type in enumerate(response_types):
                t = df[(df.cre_line == cre_line) & (df[condition] == condition_value)]
                x = (c * .5) + (r * .1)
                tmp = t[(t.response_type == response_type)]
                fraction = len(tmp) / float(len(t))
                ax[i].bar(x, fraction, width=0.1, color=colors[r])
        ax[i].set_title(cre_line)
        ax[i].set_xticks(np.arange(.15, (len(condition_values) * 0.5), 0.5))
        ax[i].set_xticklabels(condition_values)
        ax[i].set_xlabel('image set')
        if ymax:
            ax[i].set_ylim(0, ymax)
    ax[0].set_ylabel('fraction of cells')
    plt.suptitle('fraction of cells across response types', x=0.52, y=1.0,
                 horizontalalignment='center', fontsize=16)
    plt.gcf().subplots_adjust(left=0.15)
    #     plt.gcf().subplots_adjust(right=0.9)
    plt.gcf().subplots_adjust(top=0.8)
    plt.gcf().subplots_adjust(bottom=0.25)
    if sharey:
        plt.gcf().subplots_adjust(wspace=.2)
    else:
        plt.gcf().subplots_adjust(wspace=.3)
    if save_figures:
        psf.save_figure(fig, figsize, save_dir, folder, 'response_types_' + condition + '_fraction' + suffix)


def plot_mean_trace(mean_trace, frame_rate, ylabel='dF/F', legend_label=None, color='k', interval_sec=1, xlims=[-4, 4],
                    ax=None):
    xlim = [0, xlims[1] + np.abs(xlims[0])]
    if ax is None:
        fig, ax = plt.subplots()
    times = np.arange(0, len(mean_trace), 1)
    ax.plot(mean_trace, label=legend_label, linewidth=3, color=color)
    xticks, xticklabels = sf.get_xticks_xticklabels(mean_trace, frame_rate, interval_sec, window=xlims)
    ax.set_xticks(xticks)
    if interval_sec < 1:
        ax.set_xticklabels(xticklabels)
    else:
        ax.set_xticklabels([int(x) for x in xticklabels])
    ax.set_xlim(xlim[0] * int(frame_rate), xlim[1] * int(frame_rate))
    ax.set_xlabel('time (sec)')
    ax.set_ylabel(ylabel)
    sns.despine(ax=ax)
    return ax


def plot_session_metrics_lines(df, x='image_set', y='mean_response', ylabel='mean response', hue=None, show_stats=True,
                               colors=None, legend=False, save_dir=None, suffix='', ax=None):
    colors = ut.get_colors_for_image_sets()
    image_sets = ut.get_image_sets(df)
    hue_order = image_sets
    if ax is None:
        figsize = (3.5, 3.5)
        fig, ax = plt.subplots(figsize=figsize)
    for specimen_id in df.specimen_id.unique():
        y_vals = []
        for image_set in image_sets:
            t = df[(df.specimen_id == specimen_id) & (df.image_set == image_set)]
            y_vals.append(np.nanmean(t[y].values))
        y_vals = np.asarray(y_vals)
        x_vals = np.arange(0, len(y_vals))
        ax.plot(x_vals, y_vals, '.-', color='gray', linewidth=1, alpha=0.3)
    if colors:
        ax = sns.pointplot(data=df, x=x, y=y, hue=hue, hue_order=hue_order, order=hue_order, palette=colors, ax=ax)
    else:
        ax = sns.pointplot(data=df, x=x, y=y, color='gray', ax=ax, order=hue_order)
    ax.set_xlim(-0.2, len(df[x].unique()) - 1 + 0.2)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('')
    ax.legend_.remove()
    ax.set_ylim(ymin=0)
    ax.set_xlabel(x.split('_')[0] + ' ' + x.split('_')[1])
    if x is 'image_name':
        ax.set_xticklabels(df.image_name.unique(), rotation=90)

    if save_dir:
        plt.gcf().subplots_adjust(top=0.85)
        plt.gcf().subplots_adjust(right=0.85)
        plt.gcf().subplots_adjust(left=0.25)
        sf.save_figure(fig, figsize, save_dir, 'session_metrics', y + '_' + x + '_' + hue + suffix + '_lines')
    return ax
