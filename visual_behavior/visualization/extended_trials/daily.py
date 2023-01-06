import numpy as np
import matplotlib.pyplot as plt
from visual_behavior.plotting import placeAxesOnGrid
from visual_behavior.utilities import flatten_list, get_response_rates
from visual_behavior.translator.core.annotate import check_responses
from visual_behavior.translator.core.annotate import colormap, trial_translator, assign_trial_description
from visual_behavior.change_detection.trials import session_metrics


def get_reward_window(extended_trials):
    try:
        reward_window = extended_trials.iloc[0].response_window
    except Exception:
        reward_window = [0.15, 1]
    return reward_window


def make_info_table(extended_trials, ax):
    '''
    generates a table with info extracted from the dataframe
    DRO - 10/13/16
    '''
    # define the data
    try:
        user_id = extended_trials.iloc[0]['user_id']  # NOQA: F841
    except Exception:
        user_id = 'unspecified'  # NOQA: F841

    # note: training day is only calculated if the dataframe was loaded from the data folder, as opposed to individually
    try:
        training_day = extended_trials.iloc[0].training_day  # NOQA: F841
    except Exception:
        training_day = np.nan  # NOQA: F841

    # I'm using a list of lists instead of a dictionary so that it maintains order
    # the second entries are in quotes so they can be evaluated below in a try/except
    data = [['Date', 'extended_trials.iloc[0].startdatetime.strftime("%m-%d-%Y")'],
            ['Time', 'extended_trials.iloc[0].startdatetime.strftime("%H:%M")'],
            ['Duration (minutes)', 'round(extended_trials.iloc[0]["session_duration"]/60.,2)'],
            ['Total water received (ml)', 'extended_trials["cumulative_volume"].max()'],
            ['Mouse ID', 'extended_trials.iloc[0]["mouse_id"]'],
            ['Task ID', 'extended_trials.iloc[0]["task"]'],
            ['Trained by', 'user_id'],
            ['Rig ID', 'extended_trials.iloc[0]["rig_id"]'],
            ['Stimulus Description', 'extended_trials.iloc[0]["stimulus"]'],
            ['Time between flashes', 'extended_trials.iloc[0].blank_duration_range[0]'],
            ['Black screen on timeout', 'extended_trials.iloc[0].blank_screen_timeout'],
            ['Minimum pre-change time', 'extended_trials.iloc[0]["prechange_minimum"]'],
            ['Trial duration', 'extended_trials.iloc[0].trial_duration'],
            ['Number of contingent trials', 'session_metrics.num_contingent_trials(extended_trials)']]

    cell_text = []
    for x in data:
        try:
            cell_text.append([eval(x[1])])
        except Exception:
            cell_text.append([np.nan])

    # define row colors
    row_colors = [['lightgray'], ['white']] * (len(data))

    # make the table
    table = ax.table(
        cellText=cell_text,
        rowLabels=[x[0] for x in data],
        rowColours=flatten_list(row_colors)[:len(data)],
        colLabels=None,
        loc='center',
        cellLoc='left',
        rowLoc='right',
        cellColours=row_colors[:len(data)]
    )
    ax.grid(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    # do some cell resizing
    cell_dict = table.get_celld()
    for cell in cell_dict:
        if cell[1] == -1:
            cell_dict[cell].set_width(0.1)
        if cell[1] == 0:
            cell_dict[cell].set_width(0.5)


def make_session_timeline_plot(extended_trials, ax, palette='trial_types', demarcate_trials=False):
    licks = list(extended_trials['lick_times'])
    rewards = list(extended_trials['reward_times'])
    stimuli = list(extended_trials['change_time'])

    # make a local copy of the extended trial dataframe containing only the columns necessary to make this plot:
    local_df = extended_trials[['trial_type', 'auto_rewarded', 'response', 'starttime', 'trial_length']].copy()
    local_df['trial_outcome'] = local_df.apply(assign_trial_description, axis=1)

    # This plots a vertical span of a defined color for every trial type
    # to save time, I'm only plotting a span when the trial type changes
    spanstart = 0
    trial = 0
    for trial in range(1, len(local_df)):
        if demarcate_trials:
            ax.axvline(local_df.iloc[trial]['starttime'], color='k', linewidth=0.5, alpha=0.75)
        if local_df.iloc[trial]['trial_outcome'] != local_df.iloc[trial - 1]['trial_outcome']:
            # if the trial_outcome is different on this trial than the last, end the previous span at the start of this trial
            #  then start another at the start of this trial that will continue until the trial type changes again
            ax.axvspan(
                spanstart,
                local_df.iloc[trial]['starttime'],
                color=colormap(local_df.iloc[trial - 1]['trial_outcome'], palette),
                alpha=0.75
            )
            spanstart = local_df.iloc[trial]['starttime']
    # plot a span for the final trial(s)
    trial_type = trial_translator(local_df.iloc[trial - 1]['trial_type'], local_df.iloc[trial - 1]['response'])
    ax.axvspan(
        spanstart,
        local_df.iloc[trial]['starttime'] + local_df.iloc[trial]['trial_length'],
        color=colormap(trial_type, palette),
        alpha=0.75
    )

    rewards = np.array(flatten_list(rewards))
    licks = np.array(flatten_list(licks))
    stimuli = np.array(flatten_list(stimuli))

    ax.plot(licks, np.ones_like(licks), '.', color='black')
    ax.plot(
        rewards,
        1.1 * np.ones_like(rewards),
        'o',
        color='blue',
        markersize=6,
        alpha=0.75
    )
    ax.plot(
        stimuli,
        1.2 * np.ones_like(stimuli),
        'd',
        color='indigo'
    )

    ax.set_ylim(0.95, 1.25)
    ax.set_xlim(0, extended_trials.iloc[trial]['starttime'] + extended_trials.iloc[trial]['trial_length'])
    ax.set_yticklabels([])
    ax.set_xlabel('session time (s)', fontsize=14)
    ax.set_title('Full Session Timeline', fontsize=14)


def make_lick_raster_plot(extended_trials, ax, reward_window=None, xlims=(-1, 5), show_reward_window=True, y_axis_limit=None, palette='trial_types'):
    # VBA MOVED
    warnings.warn("VBA function is deprecated. Find in mindcope_qc or brain_observatory_utilites.", DeprecationWarning)
    if reward_window is None:
        try:
            reward_window = get_reward_window(extended_trials)
        except Exception:
            reward_window = [0.15, 1]

    if show_reward_window == True:  # NOQA E712
        ax.axvspan(reward_window[0], reward_window[1], facecolor='k', alpha=0.5)
    lick_x = []
    lick_y = []

    reward_x = []
    reward_y = []
    for ii, idx in enumerate(extended_trials.index):
        if len(extended_trials.loc[idx]['lick_times']) > 0:
            lt = np.array(extended_trials.loc[idx]['lick_times']) - extended_trials.loc[idx]['change_time']
            lick_x.append(lt)
            lick_y.append(np.ones_like(lt) * ii)

        if len(extended_trials.loc[idx]['reward_times']) > 0:
            rt = np.array(extended_trials.loc[idx]['reward_times']) - extended_trials.loc[idx]['change_time']
            reward_x.append(rt)
            reward_y.append(np.ones_like(rt) * ii)

        trial_type = trial_translator(extended_trials.loc[idx]['trial_type'], extended_trials.loc[idx]['response'])
        ax.axhspan(ii - 0.5, ii + 0.5, facecolor=colormap(trial_type, palette), alpha=0.5)

    ax.plot(flatten_list(lick_x), flatten_list(lick_y), '.k')
    ax.plot(flatten_list(reward_x), flatten_list(reward_y), 'o', color='blue', alpha=0.5)

    ax.set_xlim(xlims[0], xlims[1])
    if y_axis_limit is None or y_axis_limit is False:
        ax.set_ylim(-0.5, ii + 0.5)
    else:
        ax.set_ylim(-0.5, y_axis_limit + 0.5)
    ax.invert_yaxis()

    ax.set_title('Lick Raster', fontsize=16)
    ax.set_ylabel('Trial Number', fontsize=14)
    ax.set_xlabel('Time from \nstimulus onset (s)', fontsize=14)


def make_cumulative_volume_plot(df_in, ax):
    ax.barh(np.arange(len(df_in)), df_in.cumulative_volume, height=1.0, linewidth=0)
    ax.set_xlabel('Cumulative \nVolume (mL)', fontsize=14)
    ax.set_title('Cumulative Volume', fontsize=16)
    ax.set_xlim(0, 2)


def make_rolling_response_probability_plot(hit_rate, fa_rate, ax, palette='trial_types'):
    ax.plot(hit_rate, np.arange(len(hit_rate)), color=colormap('hit', palette), linewidth=2)
    ax.plot(fa_rate, np.arange(len(fa_rate)), color=colormap('false_alarm', palette), linewidth=2)

    ax.set_title('Resp. Prob.', fontsize=16)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel('Response Prob.', fontsize=14)
    ax.set_xlim(-0.1, 1.1)


def make_rolling_dprime_plot(d_prime, ax, format='vertical', peak_dprime=None, color='black', append_ticks=False, line_label=None):
    # if d_prime is a scalar, make it an array to avoid issues below
    if not hasattr(d_prime, '__len__'):
        d_prime = np.array([d_prime])
    if format == 'vertical':
        line, = ax.plot(d_prime, np.arange(len(d_prime)), color=color, linewidth=2)
        ax.set_xlabel("d'", fontsize=14)
        if peak_dprime is not None:
            ax.axvline(peak_dprime, linestyle='--', color=color)  # peak dprime line
    elif format == 'horizontal':
        line, = ax.plot(np.arange(len(d_prime)), d_prime, color=color, linewidth=2)
        ax.set_ylabel("d'", fontsize=14)
        if peak_dprime is not None:
            ax.axhline(peak_dprime, linestyle='--', color=color)  # peak dprime line
    ax.set_title("Rolling d'", fontsize=16)

    if line_label is not None:
        line.set_label(line_label)

    if peak_dprime is not None:
        if append_ticks:
            current_xticks = ax.get_xticks()
            new_xticks = np.append(current_xticks, round(peak_dprime, ndigits=2))
            ax.set_xticks(new_xticks)
        else:
            ax.set_xticks([0, round(peak_dprime, ndigits=2), 5, ])  # this is more readable?
    else:  # ticks from original implementatin
        ax.set_xlim(0, 5)


def make_legend(ax, palette='trial_types'):
    ax.plot(np.nan, np.nan, marker='.', linestyle='none', color='black')
    ax.plot(np.nan, np.nan, marker='o', linestyle='none', color='blue')
    ax.plot(np.nan, np.nan, 'd', color='indigo')
    ax.axvspan(np.nan, np.nan, color=colormap('aborted', palette))
    ax.axvspan(np.nan, np.nan, color=colormap('auto_rewarded', palette))
    ax.axvspan(np.nan, np.nan, color=colormap('hit', palette))
    ax.axvspan(np.nan, np.nan, color=colormap('miss', palette))
    ax.axvspan(np.nan, np.nan, color=colormap('false_alarm', palette))
    ax.axvspan(np.nan, np.nan, color=colormap('correct_reject', palette))
    ax.legend([
        'licks',
        'rewards',
        'stimulus\nchanges',
        'aborted\ntrials',
        'free reward\ntrials',
        'hit\ntrials',
        'miss\ntrials',
        'false alarm\ntrials',
        'correct rejection\ntrials'
    ], loc='upper center', ncol=3, fontsize=9, frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])


def make_daily_figure(
        extended_trials,
        mouse_id=None,
        reward_window=None,
        sliding_window=100,
        mouse_image_before=None,
        mouse_image_after=None,
        y_axis_limit=False,
        palette='trial_types'
):
    '''
    Generates a daily summary plot for the detection of change task
    '''

    if y_axis_limit is True:
        y_axis_limit = 475  # approximate maximum number of trials in a one hour session

    date = extended_trials.startdatetime.iloc[0].strftime('%Y-%m-%d')
    if mouse_id is None:
        mouse_id = extended_trials.mouse_id.unique()[0]
    df_nonaborted = extended_trials[(extended_trials.trial_type != 'aborted') & (extended_trials.trial_type != 'other')]

    if reward_window == None:  # NOQA: E711
        try:
            reward_window = get_reward_window(extended_trials)
        except Exception:
            reward_window = [0.15, 1]
    if sliding_window is None:
        sliding_window = len(df_nonaborted)

    fig = plt.figure(figsize=(12, 8))

    # place axes
    ax = placeAxesOnGrid(fig, dim=(1, 4), xspan=(0, 1), yspan=(0.425, 1), sharey=True)
    ax_timeline = placeAxesOnGrid(fig, xspan=(0.5, 1), yspan=(0.225, 0.3))
    ax_table = placeAxesOnGrid(fig, xspan=(0.1, 0.6), yspan=(0, 0.25), frameon=False)
    ax_legend = placeAxesOnGrid(fig, xspan=(0.5, 1), yspan=(0, 0.225), frameon=False)

    make_legend(ax_legend, palette)

    if mouse_image_before is not None:
        try:
            titles = ['before session', 'after session']
            ax_image = placeAxesOnGrid(fig, dim=(1, 2), xspan=(0.5, 1), yspan=(0, 0.18), frameon=False)
            for index, im in enumerate([mouse_image_before, mouse_image_after]):
                if im is not None:
                    ax_image[index].imshow(im, cmap='gray')
                ax_image[index].grid(False)
                print(index)
                ax_image[index].axis('off')
                ax_image[index].set_title(titles[index], fontsize=14)
        except Exception:
            pass

    # make table
    make_info_table(extended_trials, ax_table)

    # make timeline plot
    make_session_timeline_plot(extended_trials, ax_timeline, palette)

    # make trial-based plots
    make_lick_raster_plot(df_nonaborted, ax[0], reward_window=reward_window, y_axis_limit=y_axis_limit, palette=palette)
    make_cumulative_volume_plot(df_nonaborted, ax[1])
    # note (DRO - 10/31/17): after removing the autorewarded trials from the calculation, will these vectors be of different length than the lick raster?
    hit_rate, fa_rate, d_prime = get_response_rates(
        df_nonaborted,
        sliding_window=sliding_window,
    )
    make_rolling_response_probability_plot(hit_rate, fa_rate, ax[2], palette=palette)
    mean_rate = np.mean(check_responses(df_nonaborted, reward_window=reward_window) == 1.0)
    ax[2].axvline(mean_rate, color='0.5', linestyle=':')

    # CB added to display both the trial adjusted and legacy dprime metric on same plot
    for apply_trial_number_limit, first_valid_trial, append_ticks, line_label, color \
            in zip([False, True], [50, 100], [False, True], ['raw', 'adjusted'], ['black', 'grey']):
        hit_rate, fa_rate, d_prime = get_response_rates(
            df_nonaborted,
            sliding_window=sliding_window,
            apply_trial_number_limit=apply_trial_number_limit
        )
        peak_dprime = session_metrics.peak_dprime(extended_trials, first_valid_trial=first_valid_trial, apply_trial_number_limit=apply_trial_number_limit)
        if not np.isnan(peak_dprime):
            make_rolling_dprime_plot(d_prime, ax[3], peak_dprime=peak_dprime, append_ticks=append_ticks, line_label=line_label, color=color)
        else:
            make_rolling_dprime_plot(d_prime, ax[3], append_ticks=append_ticks, line_label=line_label, color=color)
    ax[3].legend(loc=0)
    plt.subplots_adjust(top=0.9)
    fig.suptitle('mouse = ' + mouse_id + ', ' + date, fontsize=20)

    return fig
