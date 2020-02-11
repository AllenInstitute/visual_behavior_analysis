import matplotlib.pyplot as plt

from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession


def plot_running_speed(oeid, width=11, height=2, ax=None):
    """
    plot running speed for a given session
    will create a new fig/ax if no axis is passed

    Arguments:
        oeid {int} -- ophys experiment ID

    Keyword Arguments:
        width {int} -- figure width if creating new (default: {11})
        height {int} -- figure height if creating new (default: {2})
        ax {matplotlib figure axis} -- axis to plot on. Will create new if none is passed (default: {None})

    Returns:
        matplotlib figure axis -- ax
    """
    session = BehaviorOphysSession.from_lims(oeid)

    if ax is None:
        fig, ax = plt.subplots(figsize=(width, height))

    ax.plot(session.running_data_df['speed'])
    ax.set_xlabel('time (s)')
    ax.set_ylabel('speed (cm/s)')
    ax.set_title('running speed for ophys_session_id {}'.format(oeid))

    return ax
