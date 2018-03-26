import numpy as np
import matplotlib.pyplot as plt
from visual_behavior.plotting import placeAxesOnGrid
from visual_behavior.utilities import flatten_list, get_response_rates


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
    ax.set_yticklabels(dates)
    ax.set_title('Inter-lick \nintervals')
    ax.invert_yaxis()
    vplot = ax.violinplot(ILIs, positions, widths=0.8, showmeans=False, showextrema=False, vert=False)
    for patch in vplot['bodies']:
        patch.set_color('black'), patch.set_alpha(0.5)


def make_trial_type_plot(dfm, session_dates, ax):
    sums = dfm.groupby(['startdatetime', 'color', ]).sum()
    colors = ['blue', 'red', 'darkgreen', 'lightgreen', 'darkorange', 'yellow']
    dates = []
    all_vals = []
    for ii, date in enumerate(session_dates[:]):

        dates.append(dfm[(dfm.startdatetime == date)].startdatetime.iloc[0].strftime('%Y-%m-%d'))
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


def make_performance_plot(df_in, session_dates, ax, reward_window=None, sliding_window=None):
    if sliding_window == None:  # NOQA: E711
        calculate_sliding_window = True  # NOQA: F841
    else:
        caclulate_sliding_window = False  # NOQA: F841

    dates = []
    max_hit_rates = []
    mean_hit_rates = []
    max_false_alarm_rates = []
    mean_false_alarm_rates = []
    max_dprime = []
    mean_dprime = []
    for ii, date in enumerate(session_dates[:]):

        dates.append(df_in[(df_in.startdatetime == date)].startdatetime.iloc[0].strftime('%Y-%m-%d'))

        df1 = df_in[(df_in.startdatetime == date) & (df_in.trial_type != 'aborted')]

        if calculate_sliding_window == True:
            sliding_window = len(df1)

        hit_rate, fa_rate, d_prime = get_response_rates(df1, sliding_window=sliding_window, reward_window=reward_window)

        max_hit_rates.append(np.nanmax(hit_rate[int(len(hit_rate) / 3):]))
        max_false_alarm_rates.append(np.nanmax(fa_rate[int(len(hit_rate) / 3):]))
        max_dprime.append(np.nanmax(d_prime[int(len(hit_rate) / 3):]))

        mean_hit_rates.append(np.nanmean(hit_rate[int(len(hit_rate) / 3):]))
        mean_false_alarm_rates.append(np.nanmean(fa_rate[int(len(hit_rate) / 3):]))
        mean_dprime.append(np.nanmean(d_prime[int(len(hit_rate) / 3):]))

    height = 0.35
    ax.barh(np.arange(len(max_hit_rates)) - height, max_hit_rates, height=height, color='darkgreen', alpha=1)
    ax.barh(np.arange(len(max_hit_rates)), max_false_alarm_rates, height=height, color='orange', alpha=1)

    ax.set_title('PEAK Hit \nand FA Rates')
    ax.set_xlabel('Max Response Probability')
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, len(dates))
    ax.invert_yaxis()


def make_dprime_plot(
        df_in,
        session_dates,
        ax,
        reward_window=None,
        return_vals=False,
        sliding_window=None
):
    if sliding_window is None:
        calculate_sliding_window = True
    else:
        calculate_sliding_window = False

    dates = []
    max_hit_rates = []
    mean_hit_rates = []
    max_false_alarm_rates = []
    mean_false_alarm_rates = []
    max_dprime = []
    mean_dprime = []
    for ii, date in enumerate(session_dates[:]):

        dates.append(df_in[(df_in.startdatetime == date)].startdatetime.iloc[0].strftime('%Y-%m-%d'))

        df1 = df_in[(df_in.startdatetime == date) & (df_in.trial_type != 'aborted')]

        if calculate_sliding_window == True:
            sliding_window = len(df1)

        hit_rate, fa_rate, d_prime = get_response_rates(
            df1,
            sliding_window=sliding_window,
            reward_window=reward_window
        )

        max_hit_rates.append(np.nanmax(hit_rate[int(len(hit_rate) / 3):]))
        max_false_alarm_rates.append(np.nanmax(fa_rate[int(len(hit_rate) / 3):]))
        max_dprime.append(np.nanmax(d_prime[int(len(hit_rate) / 3):]))

        mean_hit_rates.append(np.nanmean(hit_rate[int(len(hit_rate) / 3):]))
        mean_false_alarm_rates.append(np.nanmean(fa_rate[int(len(hit_rate) / 3):]))
        mean_dprime.append(np.nanmean(d_prime[int(len(hit_rate) / 3):]))

    height = 0.7
    ax.barh(np.arange(len(max_hit_rates)) - height / 2, max_dprime, height=height, color='black', alpha=1)
    # ax.barh(np.arange(len(max_hit_rates)),mean_dprime,height=height,color='gray',alpha=1)

    ax.set_title('PEAK \ndprime')
    ax.set_xlabel('Max dprime')
    ax.set_xlim(0, 4.75)
    ax.set_ylim(-1, len(dates))
    ax.invert_yaxis()
    if return_vals == True:
        return max_dprime


def make_total_volume_plot(df_in, session_dates, ax):
    dates = []
    total_volume = []
    number_correct = []
    for ii, date in enumerate(session_dates[:]):

        dates.append(df_in[(df_in.startdatetime == date)].startdatetime.iloc[0].strftime('%Y-%m-%d'))
        df1 = df_in[(df_in.startdatetime == date) & (df_in.trial_type != 'aborted')]

        total_volume.append(df1.number_of_rewards.sum() * df1.reward_volume.max())
        number_correct.append(df1.number_of_rewards.sum())

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


def make_summary_figure(df, mouse_id):
    dfm = df[(df.mouse_id == mouse_id)]
    session_dates = np.sort(dfm.startdatetime.unique())
    # fig,ax=plt.subplots(1,5,sharey=True,figsize=(11.5,8))
    fig = plt.figure(figsize=(11.5, 8))
    ax = []
    ax.append(placeAxesOnGrid(fig, xspan=(0, 0.2), yspan=(0, 1)))
    ax.append(placeAxesOnGrid(fig, xspan=(0.22, 0.4), yspan=(0, 1)))
    ax.append(placeAxesOnGrid(fig, xspan=(0.42, 0.6), yspan=(0, 1)))
    ax.append(placeAxesOnGrid(fig, xspan=(0.62, 0.8), yspan=(0, 1)))
    ax.append(placeAxesOnGrid(fig, xspan=(0.82, 1), yspan=(0, 1)))

    make_ILI_plot(dfm, session_dates, ax[0])

    make_trial_type_plot(dfm, session_dates, ax[1])

    make_performance_plot(dfm, session_dates, ax[2])

    make_dprime_plot(dfm, session_dates, ax[3])

    make_total_volume_plot(dfm, session_dates, ax[4])

    for i in range(1, len(ax)):
        ax[i].set_yticklabels([])

    # fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig.suptitle('MOUSE = ' + mouse_id, fontsize=18)

    return fig
