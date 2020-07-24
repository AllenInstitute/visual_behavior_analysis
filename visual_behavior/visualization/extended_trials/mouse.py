import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from visual_behavior.utilities import flatten_list
from visual_behavior.change_detection.trials import summarize
from visual_behavior.translator.core.annotate import colormap


def modify_xticks(ax, xticks, xticklabels=None, vertical_gridlines=None, gridline_alpha=0.25):
    ax.set_xticks(xticks)
    if xticklabels:
        ax.set_xticklabels(xticklabels)
    if vertical_gridlines:
        for gridline in vertical_gridlines:
            ax.axvline(gridline, color='gray', alpha=gridline_alpha, linewidth=1, zorder=-1)


def make_ILI_plot(dfm, session_dates, ax):
    ILIs = []
    positions = []
    dates = []
    for ii, date in enumerate(session_dates[:]):
        df1 = dfm[(dfm.startdatetime == date)]
        dates.append(df1.startdatetime.iloc[0].strftime('%Y-%m-%d'))

        ILI = np.diff(np.array(flatten_list(list(df1.lick_times))))
        ILI = ILI[ILI > 0.500]
        # if no licks are recorded, this will keep the violin plot from failing below
        if len(ILI) == 0:
            ILI = np.array([0])
        ILIs.append(ILI)
        ax.scatter(
            ILI,
            ii + 0.075 * np.random.randn(len(ILI)),
            color='dodgerblue'
        )

        med = np.median(ILI[np.logical_and(ILI > 2, ILI < 15)])
        ax.plot([med, med], [ii - 0.25, ii + 0.25], color='white', linewidth=3)

        positions.append(ii)

    ax.set_xlim(0, 15)
    ax.set_xlabel('time between licks (s)')
    ax.set_yticks(np.arange(0, len(ILIs)))
    ax.set_ylim(-1, len(session_dates[:]))
    # ax.set_yticklabels(dates)
    ax.set_title('Inter-lick \nintervals')
    vplot = ax.violinplot(ILIs, positions, widths=0.8, showmeans=False, showextrema=False, vert=False)
    for patch in vplot['bodies']:
        patch.set_color('black'), patch.set_alpha(0.5)


def make_lick_count_plot(df_summary, ax, height=0.8):

    ax.barh(
        np.arange(len(df_summary)),
        df_summary['number_of_licks'],
        height=height,
        color='royalblue'
    )

    ax.set_xlabel('total lick count')
    ax.set_yticks(np.arange(0, len(df_summary)))
    ax.set_title('Lick count')


def make_trial_type_plot(df_summary, ax, palette='trial_types'):

    trial_types = ['aborted', 'auto_rewarded', 'hit', 'miss', 'false_alarm', 'correct_reject']

    cumsum = np.zeros(len(df_summary))
    for ii, trial_type in enumerate(trial_types):
        fractions = df_summary['fraction_time_{}'.format(trial_type)]
        ax.barh(
            np.arange(len(df_summary)),
            fractions,
            color=colormap(trial_type, palette),
            left=cumsum,
            height=0.6
        )
        cumsum += fractions  # ensure that next bar starts at edge of existing bars

    ax.set_xlim(0, 1)
    ax.set_ylim(-1, len(df_summary))
    ax.set_title('Fraction of time in \neach trial type')
    ax.set_xlabel('Time fraction\nof session')
    ax.set_yticks(np.arange(len(df_summary)))
    # note: nudge the outermost ticklabels slightly inward to avoid overlap with the next plot
    modify_xticks(ax, xticks=[0.025, 0.5, 0.975], xticklabels=[0, 0.5, 1], vertical_gridlines=None)


def make_performance_plot(df_summary, ax, palette='trial_types'):

    dates = [pd.to_datetime(date).strftime('%Y-%m-%d') for date in df_summary.startdatetime.unique()]
    max_hit_rates = df_summary['hit_rate_peak'].values
    max_false_alarm_rates = df_summary['false_alarm_rate_peak'].values

    height = 0.35

    ax.barh(
        np.arange(len(max_hit_rates)) - height / 2,
        max_hit_rates,
        height=height,
        color=colormap('hit', palette),
        alpha=1
    )
    ax.barh(
        np.arange(len(max_false_alarm_rates)) + height / 2,
        max_false_alarm_rates,
        height=height,
        color=colormap('false_alarm', palette),
        alpha=1
    )

    ax.set_title('PEAK Hit \nand FA rates')
    ax.set_xlabel('Max Response\nProbability')
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, len(dates))
    # note: nudge the outermost ticklabels slightly inward to avoid overlap with the next plot
    modify_xticks(ax, xticks=[0.025, 0.5, 0.975], xticklabels=[0, 0.5, 1], vertical_gridlines=[0.5])


def make_dprime_plot(df_summary, ax, reward_window=None, sliding_window=100, height=0.8):

    ax.barh(
        np.arange(len(df_summary)),
        df_summary['d_prime_peak'].values,
        height=height,
        color='DimGray',
        alpha=1
    )

    ax.set_title('PEAK \ndprime')
    ax.set_xlabel('Max dprime')
    ax.set_xlim(0, 4.75)
    # note: nudge the outermost ticklabels slightly inward to avoid overlap with the next plot
    modify_xticks(ax, xticks=[0.025, 1, 2, 3, 4], xticklabels=[0, 1, 2, 3, 4], vertical_gridlines=[0, 1, 2, 3, 4])


def make_total_volume_plot(df_summary, ax):

    ax.barh(
        np.arange(len(df_summary)),
        df_summary['total_water'],
        color='royalblue',
    )

    ax.set_title('Total \nvolume earned')
    ax.set_xlabel('Volume (mL)')
    ax.set_xlim(0, 1.5)
    ticks = [0.5, 1, 1.5]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    # note: nudge the outermost ticklabels slightly inward to avoid overlap with the next plot
    modify_xticks(ax, xticks=[0.025, 0.5, 1, 1.425], xticklabels=[0, 0.5, 1, 1.5], vertical_gridlines=[0, 0.5, 1, 1.5])


def make_trial_count_plot(df_summary, ax, palette='trial_types'):

    trial_types = ['hit', 'miss', 'false_alarm', 'correct_reject']

    cumsum = np.zeros(len(df_summary))
    for ii, trial_type in enumerate(trial_types):
        counts = df_summary['number_of_{}_trials'.format(trial_type)]
        ax.barh(
            np.arange(len(df_summary)),
            counts,
            color=colormap(trial_type, palette),
            left=cumsum,
            height=0.6
        )
        cumsum += counts  # ensure that next bar starts at edge of existing bars

    ax.set_title('Trial count\nby trial type')
    ax.set_xlabel('number of trials')
    ax.set_xlim(0, 500)
    # note: nudge the outermost ticklabels slightly inward to avoid overlap with the next plot
    modify_xticks(ax, xticks=[0.025, 100, 200, 300, 400, 500], xticklabels=['0', '', '200', '', '400', ''], vertical_gridlines=[0, 100, 200, 300, 400, 500])


def add_y_labels(df_summary, ax):

    dates = [d.strftime('%Y-%m-%d') for d in df_summary.startdatetime]
    days_of_week = df_summary['startdatetime'].map(lambda x: pd.to_datetime(x).day_name())
    stages = [s for s in df_summary.stage]
    users = [user for user in df_summary.user_id]
    rigs = [rig for rig in df_summary.rig_id]
    font_colors = ['black']

    fontsize = 8

    labels = ['{} ({}), trainer: {}, rig: {}\n{}'.format(date, day_of_week, user, rig, stage) for date, day_of_week, user, rig, stage in zip(dates, days_of_week, users, rigs, stages)]

    ax.set_yticklabels(labels, fontsize=fontsize)
    for color, tick in zip(font_colors, ax.yaxis.get_major_ticks()):
        tick.label1.set_color(color)


def make_summary_figure(df_input, mouse_id=None, palette='trial_types', row_height='variable'):

    if row_height == 'variable':
        # plots with few rows look overcrowded. They need a larger row size to accomodate.
        # decrease row height linearly to a minimum of 0.5
        row_height = max((-0.5 * len(df_input) + 2.25, 0.75))

    if len(df_input['startdatetime'].unique()) == len(df_input):
        df_summary = df_input
    else:
        df_summary = summarize.session_level_summary(df_input)

    if mouse_id is None:
        mouse_id = df_summary.mouse_id.unique()[0]
        if len(df_summary.mouse_id.unique()) > 1:
            warnings.warn('More than one mouse ID present in this data, using only {}'.format(mouse_id))

    df_summary.sort_values(by='startdatetime', inplace=True)

    fig, ax = plt.subplots(1, 6, figsize=(11.5, row_height * len(df_summary)), sharey=True)

    make_lick_count_plot(df_summary, ax[0])

    make_trial_type_plot(df_summary, ax[1], palette=palette)
    make_trial_count_plot(df_summary, ax[2], palette=palette)
    make_performance_plot(df_summary, ax[3], palette=palette)
    make_dprime_plot(df_summary, ax[4])
    make_total_volume_plot(df_summary, ax[5])

    add_y_labels(df_summary, ax[0])

    ax[0].invert_yaxis()

    # make alternating horizontal bands on plot
    bar_colors = ['lightgray', 'darkgray']
    for col, axis in enumerate(ax):
        axis.tick_params(bottom=False, left=False)
        for i in range(len(df_summary)):
            axis.axhspan(i - 0.5, i + 0.5, color=bar_colors[i % 2], zorder=-1, alpha=0.25)

    if len(df_summary) > 1:
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.075)

    return fig
