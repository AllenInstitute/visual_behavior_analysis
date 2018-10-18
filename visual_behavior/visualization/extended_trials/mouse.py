import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from visual_behavior.plotting import placeAxesOnGrid
from visual_behavior.utilities import flatten_list
from visual_behavior.change_detection.trials import summarize
from visual_behavior.translator.core.annotate import colormap, assign_color, categorize_one_trial


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
            ILI = [0]
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
    ax.invert_yaxis()
    vplot = ax.violinplot(ILIs, positions, widths=0.8, showmeans=False, showextrema=False, vert=False)
    for patch in vplot['bodies']:
        patch.set_color('black'), patch.set_alpha(0.5)


def make_trial_type_plot(dfm, session_dates, ax, palette='trial_types'):

    trial_types = ['aborted', 'auto_rewarded', 'hit', 'miss', 'false_alarm', 'correct_reject']
    # reassign trial types and colors based on the passed palette
    dfm['trial_type'] = dfm.apply(categorize_one_trial, axis=1)
    dfm['color'] = dfm.apply(assign_color, axis=1, palette=palette)
    colors = [colormap(trial_type, palette) for trial_type in trial_types]
    sums = dfm.groupby(['startdatetime', 'color', ]).sum()
    all_vals = []

    dates = [pd.to_datetime(date).strftime('%Y-%m-%d') for date in dfm.startdatetime.unique()]

    for ii, date in enumerate(session_dates[:]):

        total_dur = sums.loc[date]['trial_length'].sum()
        vals = []
        for color in colors:
            if color in sums.loc[date].index:
                fraction = sums.loc[date].loc[color]['trial_length'] / total_dur
            else:
                fraction = 0
            vals.append(fraction)
        all_vals.append(vals)

    all_vals = np.array(all_vals)
    cumsum = np.hstack((np.zeros((np.shape(all_vals)[0], 1)), np.cumsum(all_vals, axis=1)))
    width = 0.8  # NOQA: F841
    for jj in range(np.shape(all_vals)[1]):
        ax.barh(np.arange(np.shape(all_vals)[0]) - 0.25, all_vals[:, jj], height=0.6, color=colors[jj], left=cumsum[:, jj])
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, len(dates))
    ax.set_title('fraction of time in \neach trial type')
    ax.set_xlabel('Time fraction of session')
    ax.invert_yaxis()


def make_performance_plot(df_summary, ax, reward_window=None, sliding_window=None, palette='trial_types'):

    dates = [pd.to_datetime(date).strftime('%Y-%m-%d') for date in df_summary.startdatetime.unique()]
    max_hit_rates = df_summary['hit_rate_peak'].values
    max_false_alarm_rates = df_summary['false_alarm_rate_peak'].values

    height = 0.35

    ax.barh(np.arange(len(max_hit_rates)) - height, max_hit_rates, height=height, color=colormap('hit', palette), alpha=1)
    ax.barh(np.arange(len(max_false_alarm_rates)), max_false_alarm_rates, height=height, color=colormap('false_alarm', palette), alpha=1)

    ax.set_title('PEAK Hit \nand FA Rates')
    ax.set_xlabel('Max Response Probability')
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, len(dates))
    ax.invert_yaxis()


def make_dprime_plot(
        df_summary,
        ax,
        reward_window=None,
        return_vals=False,
        sliding_window=100,
):

    dates = [pd.to_datetime(date).strftime('%Y-%m-%d') for date in df_summary.startdatetime.unique()]

    max_dprime = df_summary['d_prime_peak'].values

    height = 0.7
    ax.barh(np.arange(len(max_dprime)) - height / 2, max_dprime, height=height, color='black', alpha=1)

    ax.set_title('PEAK \ndprime')
    ax.set_xlabel('Max dprime')
    ax.set_xlim(0, 4.75)
    ax.set_ylim(-1, len(dates))
    ax.invert_yaxis()
    if return_vals == True:
        return max_dprime


def make_total_volume_plot(df_summary, ax):

    dates = [pd.to_datetime(date).strftime('%Y-%m-%d') for date in df_summary.startdatetime.unique()]
    total_volume = df_summary['total_water'].values

    ax.plot(
        total_volume,
        np.arange(len(total_volume)),
        'o-',
        color='blue',
        alpha=1
    )

    ax.set_title('Total \nVolume Earned')
    ax.set_xlabel('Volume (mL)')
    ax.set_xlim(0, 1.5)
    ax.set_ylim(-1, len(dates))
    ax.invert_yaxis()


def add_y_labels(df_summary, ax):
    dates = [d.strftime('%Y-%m-%d') for d in df_summary.startdatetime]
    days_of_week = df_summary['startdatetime'].dt.weekday_name
    stages = [s for s in df_summary.stage]
    font_colors = ['black']
    for i in range(1,len(stages)):
        if stages[i] != stages[i-1]:
            font_colors.append('red')
        else:
            font_colors.append('black')

    if len(dates) < 25:
        fontsize = 10
    elif len(dates) in range(25, 35):
        fontsize = 8
    else:
        fontsize = 6
    
    # alternate dates if more than 15 to avoid crowding
    if len(dates)<25:
        labels = ['{} ({})\n{}'.format(date, day_of_week, stage) for date, day_of_week, stage in zip(dates, days_of_week, stages)]
    else:
        ax.set_yticks(range(0,len(dates),2))
        labels = ['{} ({})\n{}'.format(date, day_of_week, stage) for date, day_of_week, stage in zip(dates[::2], days_of_week[::2], stages[::2])]
        font_colors = font_colors[::2]

    ax.set_yticklabels(labels, fontsize=fontsize)
    for color,tick in zip(font_colors,ax.yaxis.get_major_ticks()):
        tick.label1.set_color(color)

def make_summary_figure(df, mouse_id=None, palette='trial_types'):
    if mouse_id is None:
        mouse_id = df.mouse_id.unique()[0]
        if len(df.mouse_id.unique()) > 1:
            warnings.warn('More than one mouse ID present in this data, using only {}'.format(mouse_id))

    dfm = df[(df.mouse_id == mouse_id)]
    df_summary = summarize.session_level_summary(dfm).sort_values(by='startdatetime').reset_index(drop=True)

    session_dates = np.sort(df_summary.startdatetime.unique())

    # fig,ax=plt.subplots(1,5,sharey=True,figsize=(11.5,8))
    fig = plt.figure(figsize=(11.5, 8))
    ax = []
    ax.append(placeAxesOnGrid(fig, xspan=(0, 0.2), yspan=(0, 1)))
    ax.append(placeAxesOnGrid(fig, xspan=(0.22, 0.4), yspan=(0, 1)))
    ax.append(placeAxesOnGrid(fig, xspan=(0.42, 0.6), yspan=(0, 1)))
    ax.append(placeAxesOnGrid(fig, xspan=(0.62, 0.8), yspan=(0, 1)))
    ax.append(placeAxesOnGrid(fig, xspan=(0.82, 1), yspan=(0, 1)))

    make_ILI_plot(dfm, session_dates, ax[0])

    make_trial_type_plot(dfm, session_dates, ax[1], palette)
    make_performance_plot(df_summary, ax[2], palette=palette)
    make_dprime_plot(df_summary, ax[3])

    make_total_volume_plot(df_summary, ax[4])

    add_y_labels(df_summary, ax[0])

    for i in range(1, len(ax)):
        ax[i].set_yticklabels([])

    fig.tight_layout()
    plt.subplots_adjust(top=0.9, wspace=0.125)
    fig.suptitle('MOUSE = ' + mouse_id, fontsize=18)

    return fig
