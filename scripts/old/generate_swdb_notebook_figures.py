#!/usr/bin/env python

from __future__ import print_function

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_notebook_figures(experiment_id, save_dir):
    print(experiment_id)
    figsize = (20, 15)
    fig, ax = plt.subplots(6, 3, figsize=figsize)
    ax = ax.ravel()
    x = 0

    cache_dir = save_dir

    from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset
    dataset = VisualBehaviorOphysDataset(experiment_id, cache_dir=cache_dir)

    ax[x].imshow(dataset.max_projection, cmap='gray', )
    ax[x].axis('off')
    x += 1

    cell_index = 8
    ax[x].plot(dataset.ophys_timestamps, dataset.dff_traces[cell_index])
    ax[x].set_xlabel('seconds')
    ax[x].set_ylabel('dF/F')
    x += 1

    cell_specimen_id = dataset.get_cell_specimen_id_for_cell_index(cell_index)
    ax[x].imshow(dataset.roi_mask_dict[str(cell_specimen_id)])
    ax[x].grid('off')
    x += 1

    ax[x].plot(dataset.stimulus_timestamps, dataset.running_speed.running_speed.values)
    ax[x].set_xlabel('time (sec)')
    ax[x].set_ylabel('running speed (cm/s)')
    x += 1

    ax[x].plot(dataset.stimulus_timestamps, dataset.running_speed.running_speed.values)
    ax[x].set_xlim(600, 660)
    # plot licks
    lick_y_vals = np.repeat(-10, repeats=len(dataset.licks.time.values))
    ax[x].plot(dataset.licks.time.values, lick_y_vals, '.', label='licks')
    # plot rewards
    reward_y_vals = np.repeat(-10, repeats=len(dataset.rewards.time.values))
    ax[x].plot(dataset.rewards.time.values, reward_y_vals, 'o', label='rewards')
    ax[x].set_xlabel('time (sec)')
    ax[x].set_ylabel('running speed (cm/s)')
    ax[x].legend(loc=9, bbox_to_anchor=(1.2, 1))
    x += 1

    # plot running
    ax[x].plot(dataset.stimulus_timestamps, dataset.running_speed.running_speed.values)
    ax[x].set_xlim(600, 660)
    # plot licks
    lick_y_vals = np.repeat(-10, repeats=len(dataset.licks.time.values))
    ax[x].plot(dataset.licks.time.values, lick_y_vals, '.', label='licks')
    # plot rewards
    reward_y_vals = np.repeat(-10, repeats=len(dataset.rewards.time.values))
    ax[x].plot(dataset.rewards.time.values, reward_y_vals, 'o', label='rewards')
    for flash_number in dataset.stimulus_table.flash_number.values:
        row_data = dataset.stimulus_table[dataset.stimulus_table.flash_number == flash_number]
        ax[x].axvspan(xmin=row_data.start_time.values[0], xmax=row_data.end_time.values[0], facecolor='gray', alpha=0.3)
    ax[x].set_xlim(600, 620)
    ax[x].set_xlabel('time (sec)')
    ax[x].set_ylabel('running speed (cm/s)')
    ax[x].legend(loc=9, bbox_to_anchor=(1.2, 1))
    x += 1

    stimulus_metadata = dataset.stimulus_metadata
    stimulus_template = dataset.stimulus_template

    image_index = 2
    ax[x].imshow(stimulus_template[image_index], cmap='gray')
    image_name = stimulus_metadata[stimulus_metadata.image_index == image_index].image_name.values[0]
    ax[x].set_title(image_name)
    x += 1

    trials = dataset.trials
    images = trials.change_image_name.unique()

    trial_type = 'go'
    for i, image in enumerate(images):
        selected_trials = trials[(trials.change_image_name == image) & (trials.trial_type == trial_type)]
        response_probability = selected_trials.response.mean()
        ax[i].plot(i, response_probability, 'o')
    ax[x].set_xticks(np.arange(0, len(images), 1))
    ax[x].set_xticklabels(images, rotation=90)
    ax[x].set_ylabel('response probability')
    ax[x].set_xlabel('change image')
    ax[x].set_title('go trials')
    ax[x].set_ylim(0, 1)
    x += 1

    trial_type = 'catch'
    for i, image in enumerate(images):
        selected_trials = trials[(trials.change_image_name == image) & (trials.trial_type == trial_type)]
        response_probability = selected_trials.response.mean()
        ax[x].plot(i, response_probability, 'o')
    ax[x].set_xticks(np.arange(0, len(images), 1))
    ax[x].set_xticklabels(images, rotation=90)
    ax[x].set_ylabel('response probability')
    ax[x].set_xlabel('change image')
    ax[x].set_title('catch trials')
    ax[x].set_ylim(0, 1)
    x += 1

    colors = sns.color_palette()
    trial_types = trials.trial_type.unique()
    for i, image in enumerate(images):
        for t, trial_type in enumerate(trial_types):
            selected_trials = trials[(trials.change_image_name == image) & (trials.trial_type == trial_type)]
            response_probability = selected_trials.response.mean()
            ax[x].plot(i, response_probability, 'o', color=colors[t])
    ax[x].set_ylim(0, 1)
    ax[x].set_xticks(np.arange(0, len(images), 1))
    ax[x].set_xticklabels(images, rotation=90)
    ax[x].set_ylabel('response probability')
    ax[x].set_xlabel('change image')
    ax[x].set_title('response probability by trial type & image')
    ax[x].legend(['go', 'catch'])
    x += 1

    def make_lick_raster(trials, ax):
        fig, ax = plt.subplots(figsize=(5, 10))
        for trial in trials.trial.values:
            trial_data = trials.iloc[trial]
            # get times relative to change time
            trial_start = trial_data.start_time - trial_data.change_time
            lick_times = [(t - trial_data.change_time) for t in trial_data.lick_times]
            reward_time = [(t - trial_data.change_time) for t in trial_data.reward_times]
            # plot trials as colored rows
            ax.axhspan(trial, trial + 1, -200, 200, color=trial_data.trial_type_color, alpha=.5)
            # plot reward times
            if len(reward_time) > 0:
                ax.plot(reward_time[0], trial + 0.5, '.', color='b', label='reward', markersize=6)
            ax.vlines(trial_start, trial, trial + 1, color='black', linewidth=1)
            # plot lick times
            ax.vlines(lick_times, trial, trial + 1, color='k', linewidth=1)
            # annotate change time
            ax.vlines(0, trial, trial + 1, color=[.5, .5, .5], linewidth=1)
        # gray bar for response window
        ax.axvspan(trial_data.response_window[0], trial_data.response_window[1], facecolor='gray', alpha=.4,
                   edgecolor='none')
        ax.grid(False)
        ax.set_ylim(0, len(trials))
        ax.set_xlim([-1, 4])
        ax.set_ylabel('trials')
        ax.set_xlabel('time (sec)')
        ax.set_title('lick raster')
        plt.gca().invert_yaxis()

    make_lick_raster(dataset.trials, ax=ax[x])
    x += 1

    from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis
    analysis = ResponseAnalysis(dataset)

    tdf = analysis.trial_response_df

    largest_response = tdf[tdf.mean_response == tdf.mean_response.max()]
    cell = largest_response.cell.values[0]
    image_name = largest_response.change_image_name.values[0]
    trial_type = largest_response.trial_type.values[0]

    import visual_behavior.visualization.ophys.summary_figures as sf

    trace = largest_response.trace.values[0]
    frame_rate = analysis.ophys_frame_rate
    ax[x] = sf.plot_single_trial_trace(trace, frame_rate, ylabel='dF/F', legend_label=None, color='k', interval_sec=1,
                                       xlims=[-4, 4], ax=ax[x])
    x += 1

    traces = tdf[(tdf.cell == cell) & (tdf.trial_type == trial_type) & (tdf.change_image_name == image_name)].trace
    traces = traces.values
    ax[x] = sf.plot_mean_trace(traces, analysis.ophys_frame_rate, ylabel='dF/F', legend_label='go', color='k',
                               interval_sec=1,
                               xlims=[-4, 4], ax=ax[x])
    ax[x] = sf.plot_flashes_on_trace(ax[x], analysis, trial_type='go', omitted=False, alpha=0.4)
    x += 1

    traces = tdf[(tdf.cell == cell) & (tdf.trial_type == 'go') & (tdf.change_image_name == image_name)].trace
    traces = np.asarray(traces)
    ax[x] = sf.plot_mean_trace(traces, analysis.ophys_frame_rate, ylabel='dF/F', legend_label='go', color='k',
                               interval_sec=1, xlims=[-4, 4], ax=ax[x])
    ax[x] = sf.plot_flashes_on_trace(ax[x], analysis, trial_type='go', omitted=False, alpha=0.4)
    x += 1

    traces = tdf[(tdf.cell == cell) & (tdf.trial_type == 'catch') & (tdf.change_image_name == image_name)].trace
    traces = np.asarray(traces)
    ax[x] = sf.plot_mean_trace(traces, analysis.ophys_frame_rate, ylabel='dF/F', legend_label=None, color='k',
                               interval_sec=1, xlims=[-4, 4], ax=ax[x])
    ax[x] = sf.plot_flashes_on_trace(ax[x], analysis, trial_type=None, omitted=False, alpha=0.4)
    x += 1

    fdf = analysis.flash_response_df

    from scipy.stats import sem

    images = np.sort(fdf.image_name.unique())
    for i, image_name in enumerate(images):
        responses = fdf[(fdf.cell == cell) & (fdf.image_name == image_name)].mean_response.values
        mean_response = np.mean(responses)
        std_err = sem(responses)
        ax[x].plot(i, mean_response, 'o', color='k')
        ax[x].errorbar(i, mean_response, yerr=std_err, color='k')
    ax[x].set_xticks(np.arange(0, len(images), 1))
    ax[x].set_xticklabels(images, rotation=90)
    ax[x].set_ylabel('mean dF/F')
    ax[x].set_xlabel('image')
    ax[x].set_title('image tuning curve - all flashes')
    x += 1

    df = analysis.trial_response_df[(analysis.trial_response_df.trial_type == 'go')]

    images = np.sort(df.change_image_name.unique())
    image_responses = []
    sem_responses = []
    for i, change_image_name in enumerate(images):
        responses = df[(df.change_image_name == change_image_name) & (df.cell == cell)].mean_response.values
        mean_response = np.mean(responses)
        sem_response = sem(responses)
        image_responses.append(mean_response)
        sem_responses.append(sem_response)
    image_responses = np.asarray(image_responses)
    sem_responses = np.asarray(sem_responses)

    x_vals = np.arange(0, len(images), 1)
    ax[x].plot(x_vals, image_responses, 'o', color='k')
    ax[x].errorbar(x_vals, image_responses, yerr=sem_responses, color='k')
    ax[x].set_xticks(np.arange(0, len(images), 1))
    ax[x].set_xticklabels(images, rotation=90)
    ax[x].set_ylabel('mean dF/F')
    ax[x].set_ylabel('image')
    ax[x].set_title('image tuning curve - go trials')
    x += 1

    def compute_lifetime_sparseness(image_responses):
        # image responses should be an array of the trial averaged responses to each image
        # sparseness = 1-(sum of trial averaged responses to images / N)squared / (sum of (squared mean responses / n)) / (1-(1/N))
        # N = number of images
        # after Vinje & Gallant, 2000; Froudarakis et al., 2014
        N = float(len(image_responses))
        ls = (
            (1 - (1 / N) * ((np.power(image_responses.sum(axis=0), 2)) / (np.power(image_responses, 2).sum(axis=0)))) /
            (1 - (1 / N)))
        return ls

    responsive_cells = []
    for cell in df.cell.unique():
        cell_data = df[(df.cell == cell)]
        total_trials = len(cell_data)
        responsive_trials = len(cell_data[cell_data.p_value < 0.005])
        fraction_responsive_trials = responsive_trials / float(total_trials)
        if fraction_responsive_trials > 0.1:
            responsive_cells.append(cell)
    print('fraction responsive cells = ', len(responsive_cells) / float(len(analysis.trial_response_df.cell.unique())))

    images = np.sort(df.change_image_name.unique())
    lifetime_sparseness_values = []
    for cell in responsive_cells:
        image_responses = []
        for i, change_image_name in enumerate(images):
            responses = df[(df.cell == cell) & (df.change_image_name == change_image_name)].mean_response.values
            mean_response = np.mean(responses)
            std_err = sem(responses)
            image_responses.append(mean_response)
        ls = compute_lifetime_sparseness(np.asarray(image_responses))
        lifetime_sparseness_values.append(ls)
    lifetime_sparseness_values = np.asarray(lifetime_sparseness_values)
    mean_lifetime_sparseness = np.mean(lifetime_sparseness_values)
    print('mean lifetime sparseness for go trials:', mean_lifetime_sparseness)

    df = analysis.trial_response_df[(analysis.trial_response_df.trial_type == 'catch')]
    responsive_cells = []
    for cell in df.cell.unique():
        cell_data = df[(df.cell == cell)]
        total_trials = len(cell_data)
        responsive_trials = len(cell_data[cell_data.p_value < 0.005])
        fraction_responsive_trials = responsive_trials / float(total_trials)
        if fraction_responsive_trials > 0.1:
            responsive_cells.append(cell)

    images = np.sort(df.change_image_name.unique())
    lifetime_sparseness_values = []
    for cell in responsive_cells:
        image_responses = []
        for i, change_image_name in enumerate(images):
            responses = df[(df.cell == cell) & (df.change_image_name == change_image_name)].mean_response.values
            mean_response = np.mean(responses)
            std_err = sem(responses)
            image_responses.append(mean_response)
        ls = compute_lifetime_sparseness(np.asarray(image_responses))
        lifetime_sparseness_values.append(ls)
    lifetime_sparseness_values = np.asarray(lifetime_sparseness_values)
    mean_lifetime_sparseness = np.mean(lifetime_sparseness_values)

    print('mean lifetime sparseness for catch trials:', mean_lifetime_sparseness)

    plt.suptitle(str(experiment_id))
    fig.tight_layout()
    sf.save_figure(fig, figsize, save_dir, 'summary_figures', str(experiment_id))
    print(x)
    plt.close()


if __name__ == '__main__':

    sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})
    sns.set_style('white')
    sns.set_palette('deep')

    import sys

    # save_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\visual_behavior_pilot_analysis'

    experiment_id = sys.argv[1]
    # experiment_id = 696136550
    plot_notebook_figures(experiment_id, save_dir)
