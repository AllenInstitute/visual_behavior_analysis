import matplotlib.pyplot as plt
from visual_behavior.data_access import loading as data_loading


# BEHAVIOR

def plot_running_speed(ophys_session_id, ax=None):
    """
    plot running speed for a given session
    will create a new fig/ax if no axis is passed

    Arguments:
        ophys_session_id {int} -- ophys session ID

    Keyword Arguments:
        ax {matplotlib figure axis} -- axis to plot on. Will create new if none is passed (default: {None})

    Returns:
        matplotlib figure axis -- ax
    """
    running_speed = data_loading.get_sdk_running_speed(ophys_session_id)
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 3))
    ax.plot(running_speed)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('running speed\n(cm/s)')
    ax.set_title('running speed for ophys_session_id {}'.format(ophys_session_id))

    return ax


def plot_lick_raster(ophys_session_id, ax=None, response_window=[0.15, 0.75]):
    trials = data_loading.get_sdk_trials(ophys_session_id)
    if ax is None:
        figsize = (5, 10)
        fig, ax = plt.subplots(figsize=figsize)
    for trial in trials.index.values:
        trial_data = trials.iloc[trial]
        # get times relative to change time
        trial_start = trial_data.start_time - trial_data.change_time
        lick_times = [(t - trial_data.change_time) for t in trial_data.lick_times]
        reward_time = [trial_data.reward_time - trial_data.change_time]
        # plot reward times
        if len(reward_time) > 0:
            ax.plot(reward_time[0], trial + 0.5, '.', color='b', label='reward', markersize=6)
        ax.vlines(trial_start, trial, trial + 1, color='black', linewidth=1)
        # plot lick times
        ax.vlines(lick_times, trial, trial + 1, color='k', linewidth=1)
        # annotate change time
        ax.vlines(0, trial, trial + 1, color=[.5, .5, .5], linewidth=1)
    # gray bar for response window
    ax.axvspan(response_window[0], response_window[1], facecolor='gray', alpha=.4,
               edgecolor='none')
    ax.grid(False)
    ax.set_ylim(0, len(trials))
    ax.set_xlim([-1, 4])
    ax.set_ylabel('trials')
    ax.set_xlabel('time (sec)')
    ax.set_title('lick raster')
    return ax
