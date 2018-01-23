from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
try:
    import seaborn as sns
    sns.set_style('white')
except ImportError:
    pass

from visual_behavior import utilities as vbu


def make_daily_figure(
        df_in,
        mouse_id=None,
        reward_window=None,
        sliding_window=100,
        mouse_image_before=None,
        mouse_image_after=None
):
    '''
    Generates a daily summary plot for the detection of change task
    '''
    date = df_in.startdatetime.iloc[0].strftime('%Y-%m-%d')
    if mouse_id is None:
        mouse_id = df_in.mouse_id.unique()[0]
    df_nonaborted = df_in[(df_in.trial_type != 'aborted') & (df_in.trial_type != 'other')]

    if reward_window == None:  # NOQA: E711
        try:
            reward_window = vbu.get_reward_window(df_in)
        except Exception:
            reward_window = [0.15, 1]
    if sliding_window == None:  # NOQA: E712
        sliding_window = len(df_nonaborted)

    fig = plt.figure(figsize=(12, 8))

    # place axes
    ax = placeAxesOnGrid(fig, dim=(1, 4), xspan=(0, 1), yspan=(0.425, 1), sharey=True)
    ax_timeline = placeAxesOnGrid(fig, xspan=(0.5, 1), yspan=(0.225, 0.3))
    ax_table = placeAxesOnGrid(fig, xspan=(0.1, 0.6), yspan=(0, 0.25), frameon=False)

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
    make_info_table(df_in, ax_table)

    # make timeline plot
    make_session_timeline_plot(df_in, ax_timeline)

    # make trial-based plots
    make_lick_raster_plot(df_nonaborted, ax[0], reward_window=reward_window)
    make_cumulative_volume_plot(df_nonaborted, ax[1])
    # note (DRO - 10/31/17): after removing the autorewarded trials from the calculation, will these vectors be of different length than the lick raster?
    hit_rate, fa_rate, d_prime = vbu.get_response_rates(
        df_nonaborted,
        sliding_window=sliding_window,
        reward_window=reward_window
    )
    make_rolling_response_probability_plot(hit_rate, fa_rate, ax[2])
    mean_rate = np.mean(vbu.check_responses(df_nonaborted, reward_window=reward_window) == 1.0)
    ax[2].axvline(mean_rate, color='0.5', linestyle=':')
    make_rolling_dprime_plot(d_prime, ax[3])

    plt.subplots_adjust(top=0.9)
    fig.suptitle('mouse = ' + mouse_id + ', ' + date, fontsize=20)

    return fig


def make_daily_plot(*args, **kwargs):
    '''original function name. Putting this here to avoid breaking old code'''
    make_daily_figure(*args, **kwargs)


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

    make_trial_type_plot(dfm, ax[1])

    make_performance_plot(dfm, ax[2])

    make_dprime_plot(dfm, ax[3])

    make_total_volume_plot(dfm, ax[4])

    for i in range(1, len(ax)):
        ax[i].set_yticklabels([])

    # fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig.suptitle('MOUSE = ' + mouse_id, fontsize=18)

    return fig


def make_cumulative_volume_plot(df_in, ax):
    ax.barh(np.arange(len(df_in)), df_in.cumulative_volume, height=1.0, linewidth=0)
    ax.set_xlabel('Cumulative \nVolume (mL)', fontsize=14)
    ax.set_title('Cumulative Volume', fontsize=16)
    if np.max(df_in.cumulative_volume) <= 0.5:
        ax.set_xticks(np.arange(0, 0.5, 0.1))
    else:
        ax.set_xticks(np.arange(0, np.max(df_in.cumulative_volume), 0.25))


def make_rolling_response_probability_plot(hit_rate, fa_rate, ax):
    ax.plot(hit_rate, np.arange(len(hit_rate)), color='darkgreen', linewidth=2)
    ax.plot(fa_rate, np.arange(len(fa_rate)), color='orange', linewidth=2)

    ax.set_title('Resp. Prob.', fontsize=16)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel('Response Prob.', fontsize=14)


def make_rolling_dprime_plot(d_prime, ax, format='vertical'):
    if format == 'vertical':
        ax.plot(d_prime, np.arange(len(d_prime)), color='black', linewidth=2)
        ax.set_xlabel("d'", fontsize=14)
    elif format == 'horizontal':
        ax.plot(np.arange(len(d_prime)), d_prime, color='black', linewidth=2)
        ax.set_ylabel("d'", fontsize=14)
    ax.set_title("Rolling d'", fontsize=16)


def make_lick_raster_plot(df_in, ax, reward_window=None):
    if reward_window is None:
        try:
            reward_window = vbu.get_reward_window(df_in)
        except Exception:
            reward_window = [0.15, 1]

    ax.axvspan(reward_window[0], reward_window[1], facecolor='k', alpha=0.5)
    lick_x = []
    lick_y = []

    reward_x = []
    reward_y = []
    for ii, idx in enumerate(df_in.index):
        if len(df_in.loc[idx]['lick_times']) > 0:
            lt = np.array(df_in.loc[idx]['lick_times']) - df_in.loc[idx]['change_time']
            lick_x.append(lt)
            lick_y.append(np.ones_like(lt) * ii)

        if len(df_in.loc[idx]['reward_times']) > 0:
            rt = np.array(df_in.loc[idx]['reward_times']) - df_in.loc[idx]['change_time']
            reward_x.append(rt)
            reward_y.append(np.ones_like(rt) * ii)

        ax.axhspan( ii - 0.5, ii + 0.5, facecolor=df_in.loc[idx]['color'], alpha=0.5)

    ax.plot(vbu.flatten_list(lick_x), vbu.flatten_list(lick_y), '.k')
    ax.plot(vbu.flatten_list(reward_x), vbu.flatten_list(reward_y), 'o', color='blue')

    ax.set_xlim(-1, 5)
    ax.set_ylim(-0.5, ii + 0.5)
    ax.invert_yaxis()

    ax.set_title('Lick Raster', fontsize=16)
    ax.set_ylabel('Trial Number', fontsize=14)
    ax.set_xlabel('Time from \nstimulus onset (s)', fontsize=14)


def make_info_table(df, ax):
    '''
    generates a table with info extracted from the dataframe
    DRO - 10/13/16
    '''
    # define the data
    try:
        user_id = df.iloc[0]['user_id']  # NOQA: F841
    except Exception:
        user_id = 'unspecified'  # NOQA: F841

    # note: training day is only calculated if the dataframe was loaded from the data folder, as opposed to individually
    try:
        training_day = df.iloc[0].training_day  # NOQA: F841
    except Exception:
        training_day = np.nan  # NOQA: F841

    # I'm using a list of lists instead of a dictionary so that it maintains order
    # the second entries are in quotes so they can be evaluated below in a try/except
    data = [['Date', 'df.iloc[0].startdatetime.strftime("%m-%d-%Y")'],
            ['Training Day', 'training_day'],
            ['Time', 'df.iloc[0].startdatetime.strftime("%H:%M")'],
            ['Duration (minutes)', 'round(df.iloc[0]["session_duration"]/60.,2)'],
            ['Total water received (ml)', 'df["cumulative_volume"].max()'],
            ['Mouse ID', 'df.iloc[0]["mouse_id"]'],
            ['Task ID', 'df.iloc[0]["task"]'],
            ['Trained by', 'user_id'],
            ['Rig ID', 'df.iloc[0]["rig_id"]'],
            ['Stimulus Description', 'df.iloc[0]["stimulus"]'],
            ['Time between flashes', 'df.iloc[0].blank_duration_range[0]'],
            ['Black screen on timeout', 'df.iloc[0].blank_screen_timeout'],
            ['Minimum pre-change time', 'df.iloc[0]["prechange_minimum"]'],
            ['Trial duration', 'df.iloc[0].trial_duration']]

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
        rowColours=vbu.flatten_list(row_colors)[:len(data)],
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


def make_session_timeline_plot(df_in, ax):
    licks = list(df_in['lick_times'])
    rewards = list(df_in['reward_times'])
    stimuli = list(df_in['change_time'])

    # This plots a vertical span of a defined color for every trial type
    # to save time, I'm only plotting a span when the trial type changes
    spanstart = 0
    trial = 0
    for trial in range(1, len(df_in)):
        if df_in.iloc[trial]['color'] != df_in.iloc[trial - 1]['color']:
            ax.axvspan(
                spanstart,
                df_in.iloc[trial]['starttime'],
                color=df_in.iloc[trial - 1]['color'],
                alpha=0.75
            )
            spanstart = df_in.iloc[trial]['starttime']
    # plot a span for the final trial(s)
    ax.axvspan(
        spanstart,
        df_in.iloc[trial]['starttime'] + df_in.iloc[trial]['trial_length'],
        color=df_in.iloc[trial - 1]['color'],
        alpha=0.75
    )

    rewards = np.array(vbu.flatten_list(rewards))
    licks = np.array(vbu.flatten_list(licks))
    stimuli = np.array(vbu.flatten_list(stimuli))

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
    ax.set_xlim(0, df_in.iloc[trial]['starttime'] + df_in.iloc[trial]['trial_length'])
    ax.set_yticklabels([])
    ax.set_xlabel('session time (s)', fontsize=14)
    ax.set_title('Full Session Timeline', fontsize=14)


def save_figure(fig, fname, formats=['.pdf'], transparent=False, dpi=300, facecolor=None, **kwargs):
    import matplotlib as mpl
    mpl.rcParams['pdf.fonttype'] = 42

    if 'size' in kwargs.keys():
        fig.set_size_inches(kwargs['size'])

    elif 'figsize' in kwargs.keys():
        fig.set_size_inches(kwargs['figsize'])
    else:
        fig.set_size_inches(11, 8.5)
    for f in formats:
        fig.savefig(
            fname + f,
            transparent=transparent,
            orientation='landscape',
            dpi=dpi
        )


def make_ILI_plot(dfm, session_dates, ax):
    ILIs = []
    positions = []
    dates = []
    for ii, date in enumerate(session_dates[:]):
        df1 = dfm[(dfm.startdatetime == date)]
        dates.append(df1.startdatetime.iloc[0].strftime('%Y-%m-%d'))

        ILI = np.diff(np.array(vbu.flatten_list(list(df1.lick_times))))
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


def make_trial_type_plot(dfm, ax):
    sums = dfm.groupby(['startdatetime', 'color', ]).sum()
    colors = ['blue', 'red', 'darkgreen', 'lightgreen', 'darkorange', 'yellow']
    dates = dfm.startdatetime.unique()
    all_vals = []
    for date in dates:
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


def make_performance_plot(df_in, ax, reward_window=None, sliding_window=None):
    if sliding_window == None:  # NOQA: E711
        calculate_sliding_window = True  # NOQA: F841
    else:
        caclulate_sliding_window = False  # NOQA: F841

    dates = df_in.startdatetime.unique()
    max_hit_rates = []
    mean_hit_rates = []
    max_false_alarm_rates = []
    mean_false_alarm_rates = []
    max_dprime = []
    mean_dprime = []
    for ii, date in enumerate(dates):
        df1 = df_in[(df_in.startdatetime == date) & (df_in.trial_type != 'aborted')]

        if calculate_sliding_window == True:  # NOQA: E712
            sliding_window = len(df1)

        hit_rate, fa_rate, d_prime = vbu.get_response_rates(df1, sliding_window=sliding_window, reward_window=reward_window)

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
        ax,
        reward_window=None,
        return_vals=False,
        sliding_window=None
):
    if sliding_window == None:  # NOQA: E712
        calculate_sliding_window = True
    else:
        calculate_sliding_window = False

    dates = df_in.startdatetime.unique()
    max_hit_rates = []
    mean_hit_rates = []
    max_false_alarm_rates = []
    mean_false_alarm_rates = []
    max_dprime = []
    mean_dprime = []
    for ii, date in enumerate(dates):
        df1 = df_in[(df_in.startdatetime == date) & (df_in.trial_type != 'aborted')]

        if calculate_sliding_window == True:  # NOQA: E712
            sliding_window = len(df1)

        hit_rate, fa_rate, d_prime = vbu.get_response_rates(
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
    if return_vals == True:  # NOQA: E712
        return max_dprime


def make_total_volume_plot(df_in, ax):
    dates = df_in.startdatetime.unique()
    total_volume = []
    number_correct = []
    for ii, date in enumerate(dates):
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


def DoC_PsychometricCurve(
        input,
        ax=None,
        parameter='delta_ori',
        title="",
        linecolor='black',
        linewidth=2,
        alpha=0.75,
        xval_jitter=0,
        initial_guess=(np.log10(20), 1, 0.2, 0.2),
        fontsize=14,
        logscale=True,
        minval=0.4,
        xticks=[0.0, 1.0, 2.5, 5.0, 10.0, 20.0, 45.0, 90.0],
        xlim=(0, 90),
        returnvals=False,
        mintrials=20,
        **kwargs
):

    '''
    A specialized function for plotting psychometric curves in the delta_orientation versinon of the detection of change task
    Makes some specific assumptions about plotting parameters

    Important note: the 'mintrials' argument will disregard any datapoints with fewer observations than its set value
    '''
    if isinstance(input, str):
        # if a string is input, assume it's a filname. Load it
        df = vbu.create_doc_dataframe(input)
        response_df = vbu.make_response_df(df[((df.trial_type == 'go') | (df.trial_type == 'catch'))])
    elif isinstance(input, pd.DataFrame) and 'response_probability' not in input.columns:
        # this would mean that a response_probability dataframe has not been passed. Create it
        df = input
        response_df = vbu.make_response_df(df[((df.trial_type == 'go') | (df.trial_type == 'catch'))])
    elif isinstance(input, pd.DataFrame) and 'response_probability' in input.columns:
        response_df = input
    else:
        print("can't deal with input")

    if ax == None:  # NOQA: E711
        fig, ax = plt.subplots()

    if parameter == 'delta_ori':
        xlabel = '$\Delta$Orientation'
    else:
        xlabel = parameter

    params = plot_psychometric(
        response_df[response_df.attempts >= mintrials][parameter].values,
        response_df[response_df.attempts >= mintrials]['response_probability'].values,
        CI=response_df[response_df.attempts >= mintrials]['CI'].values,
        xlim=xlim,
        xlabel=xlabel,
        xticks=xticks,
        title=title,
        minval=minval,
        logscale=logscale,
        ax=ax,
        linecolor=linecolor,
        linewidth=linewidth,
        alpha=alpha,
        xval_jitter=xval_jitter,
        initial_guess=initial_guess,
        fontsize=fontsize,
        returnvals=returnvals
    )

    return params


def plot_psychometric(
        x,
        y,
        initial_guess=(0.1, 1, 0.5, 0.5),
        alpha=1,
        xval_jitter=0,
        **kwargs
):
    '''
    Uses the psychometric plotting function in psy to make a psychometric curve with a fit
    '''

    ax = kwargs.get('ax', None)
    ylabel = kwargs.get('ylabel', 'Respone Probability')
    title = kwargs.get('title', '')
    show_line = kwargs.get('show_line', True)
    show_points = kwargs.get('show_points', True)
    linecolor = kwargs.get('linecolor', 'k')
    linewidth = kwargs.get('linewidth', 2)
    linestyle = kwargs.get('linestyle', '-')
    fontsize = kwargs.get('fontsize', 10)
    yerr = kwargs.get('yerr', None)
    CI = kwargs.get('CI', None)
    logscale = kwargs.get('logscale', False)
    marker = kwargs.get('marker', 'o')
    markersize = kwargs.get('markersize', 9)
    fittype = kwargs.get('fittype', 'Weibull')
    returnvals = kwargs.get('returnvals', False)
    showXLabel = kwargs.get('showXLabel', True)
    showYLabel = kwargs.get('showYLabel', True)
    xticks = kwargs.get('xticks', None)
    xlim = kwargs.get('xlim', [-0.1, 1.1])
    minval = kwargs.get('minval', None)
    zorder = kwargs.get("zorder", np.inf)
    linealpha = alpha

    x = np.array(x, dtype=np.float)

    if logscale == True:  # NOQA: E712
        xlabel = kwargs.get('xlabel', 'Contrast (log scale)')
    else:
        xlabel = kwargs.get('xlabel', 'Contrast')

    # turn confidence intervals into lower and upper errors
    if CI is not None:
        lerr = []
        uerr = []
        for i in range(len(x)):
            lerr.append(y[i] - CI[i][0])
            uerr.append(CI[i][1] - y[i])
        yerr = [lerr, uerr]

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    if logscale is True:
        if xticks is None:
            minval = 0.03 if minval == None else minval  # NOQA: E711
            ax.set_xticks(np.log10([minval, 0.05, 0.1, 0.25, 0.5, 1]))
            ax.set_xticklabels([0, 0.05, 0.1, 0.25, 0.5, 1])
        else:
            minval = 0.03 if minval == None else minval  # NOQA: E712
            ax.set_xticks(np.log10([minval] + xticks[1:]))
            ax.set_xticklabels(xticks)

        ax.set_xlim([np.log10(minval - 0.001), np.log10(xlim[1])])
    else:
        if xticks is None:
            ax.set_xlim(xlim)
        else:
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks)

        ax.set_xlim(xlim)

    # because the log of 0 is -inf, we need to replace a 0 contrast with a small positive number to avoid an error
    if logscale is True and x[0] == 0:
        x[0] = minval

    x = np.float64(x)
    y = np.float64(y)
    if logscale is False and show_points is True:
        xvals_to_plot = x
        if xval_jitter != 0:
            xvals_to_plot = [xval_jitter * np.random.randn() + v for v in xvals_to_plot]
        l = ax.plot(  # NOQA: E741
            xvals_to_plot,
            y,
            marker=marker,
            markersize=markersize,
            color=linecolor,
            linestyle='None',
            zorder=zorder,
            alpha=linealpha
        )
    elif logscale == True and show_points is True:  # NOQA: E712
        xvals_to_plot = np.log10(x)
        if xval_jitter != 0:
            xvals_to_plot = [
                xval_jitter * np.random.randn() + v
                for v in xvals_to_plot
            ]
        l = ax.plot(  # NOQA: E741
            xvals_to_plot,
            y,
            marker=marker,
            markersize=markersize,
            color=linecolor,
            linestyle='None',
            zorder=zorder,
            alpha=linealpha
        )
    else:
        l = None  # NOQA: E741
    try:
        # Plot error bars
        if 'yerr' is not None and show_points is True:
            # Plot error on data points
            if logscale is False:
                (l_err, caps, _) = ax.errorbar(
                    xvals_to_plot,
                    y,
                    markersize=markersize,
                    yerr=yerr,
                    color=linecolor,
                    linestyle='None',
                    zorder=zorder,
                    alpha=linealpha
                )
            else:
                (l_err, caps, _) = ax.errorbar(
                    xvals_to_plot,
                    y,
                    markersize=markersize,
                    yerr=yerr,
                    color=linecolor,
                    linestyle='None',
                    zorder=zorder,
                    alpha=linealpha
                )
            for cap in caps:
                cap.set_markeredgewidth(0)
                cap.set_linewidth(2)
        else:
            l_err = 'None'
    except Exception as e:
        print("failed to add error bars", e)
    if show_line is True:
        try:
            # Fit with either 'Weibull' for 'Logistic'
            p_guess = initial_guess
            # NOTE: changed to scipy.optimize.leastsquares on 2/15/17 to allow bounds to be explicitly passed
            result = optimize.least_squares(
                residuals,
                p_guess,
                args=(x, y, fittype),
                bounds=([-np.inf, -np.inf, 0, 0], [np.inf, np.inf, 1, 1])
            )
            p = result.x
            alpha, beta, Lambda, Gamma = p
            # Plot curve fit
            xp = np.linspace(min(x), max(x), 1001)
            pxp = curve_fit(p, xp, fittype)
            if logscale == False:  # NOQA: E712
                l_fit = ax.plot(xp, pxp, linestyle=linestyle, linewidth=linewidth, color=linecolor, alpha=linealpha)
            else:
                l_fit = ax.plot(np.log10(xp), pxp, linestyle=linestyle, linewidth=linewidth, color=linecolor, alpha=linealpha)
        except Exception as e:
            print("failed to plot sigmoid", e)

    if showYLabel == True:  # NOQA: E712
        ax.set_ylabel(ylabel, fontsize=fontsize)
    else:
        ax.set_yticklabels([])
    if showXLabel == True:  # NOQA: E712
        ax.set_xlabel(xlabel, fontsize=fontsize)
    else:
        ax.set_xticklabels([])

    ax.set_ylim([-0.05, 1.05])
    ax.set_title(title, fontsize=fontsize + 1)
    ax.tick_params(labelsize=fontsize - 1)

    if returnvals == True:  # NOQA: E712
        # c50 = np.true_divide(np.diff(pxp[:1:-1]),2)[0]
        # print "C50:",c50
        # closest_idx = (np.abs(pxp-c50)).argmin()
        # c50_xval = xp[closest_idx]
        (c50_xval, c50) = getThreshold(p, x, criterion=0.5, fittype='Weibull')
        return ax, l, l_err, l_fit, p, (c50_xval, c50)
    else:
        return ax


def curve_fit(p, x, fittype='Weibull'):
    x = np.array(x)
    if fittype.lower() == 'logistic':
        alpha, beta, Lambda, Gamma = p
        y = Gamma + (1 - Gamma - Lambda) * (1 / (1 + np.exp(-(x - alpha) / beta)))
    elif fittype.lower() == 'weibull':
        alpha, beta, Lambda, Gamma = p
        y = Gamma + (1 - Gamma - Lambda) * (1 - np.exp(-(x / alpha) ** beta))
    elif fittype.lower() == 'gaussian_like':
        mu, sigma, a = p
        y = a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    else:
        print("NO FIT TYPE DEFINED")
    return y


def residuals(p, x, y, fittype='Weibull'):
    res = y - curve_fit(p, x, fittype)
    return res


def getThreshold(p, x=np.linspace(0, 1, 1001), criterion=0.5, fittype='Weibull'):
    '''
    given fit parameters for a sigmoid, returns the x and y values corresponding to a particular criterion
    '''
    y = curve_fit(p, x, fittype=fittype)

    yval = criterion * (1 - p[2] - p[3]) + p[3]
    xval = np.interp(yval, y, x)
    return xval, yval


def plot_first_licks(pkl):
    """
    plots distribution of first lick times for a file.

    tries to gues about flasht times (but might not be very trustworthy)
    author: justin


    """
    trials = vbu.create_doc_dataframe(pkl)

    trials['first_lick'] = trials['lick_times'].map(lambda l: l[0] if len(l) > 0 else np.nan)
    trials['first_lick'] = trials['first_lick'] - trials['starttime']

    aborted = (
        trials['trial_type'].isin(['aborted', ])
        & ~pd.isnull(trials['first_lick'])
    )
    catch = (
        trials['trial_type'].isin(['catch', ])
        & ~pd.isnull(trials['first_lick'])
    )
    go = (
        trials['trial_type'].isin(['go', ])
        & ~pd.isnull(trials['first_lick'])
    )

    f, ax = plt.subplots(1, figsize=(8, 4), sharex=True)

    bar_width = 0.1
    bins = np.arange(0, 6, bar_width)

    x1, _ = np.histogram(trials[aborted]['first_lick'].values, bins)
    x2, _ = np.histogram(trials[catch]['first_lick'].values, bins)
    x3, _ = np.histogram(trials[go]['first_lick'].values, bins)

    ax.bar(bins[:-1], x1, width=bar_width, edgecolor='none', color='indianred')
    ax.bar(bins[:-1], x2, width=bar_width, edgecolor='none', color='orange', bottom=x1)
    ax.bar(bins[:-1], x3, width=bar_width, edgecolor='none', color='limegreen', bottom=x1 + x2)

    ax.set_title(pkl.split('/')[-1])

    if ('500ms' in pkl) or ('NaturalImages' in pkl):
        for flash in (np.arange(0, 6, 0.7) + 0.2):
            ax.axvspan(flash, flash + 0.2, color='lightblue', zorder=-10)
    ax.set_xlim(0, 6)

    return f, ax


def show_image(
        img,
        x=None,
        y=None,
        figsize=(10, 10),
        ax=None,
        cmin=None,
        cmax=None,
        cmap=None,
        colorbar=False,
        colorbarlabel="",
        fontcolor='black',
        show_grid=False,
        colorbarticks=None,
        colorbarlocation='right',
        title=None,
        alpha=1,
        origin='upper',
        hide_ticks=True,
        aspect=1,
        interpolation='none',
        fontsize=16,
        returnval='image'
):
    '''
    A simple image display function
    '''
    if cmin == None:  # NOQA: E711
        cmin = np.min(img)
    if cmax == None:  # NOQA: E711
        cmax = np.max(img)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if cmap == None:  # NOQA: E711
        cmap = plt.cm.gray
    im = ax.imshow(
        img,
        cmap=cmap,
        clim=[cmin, cmax],
        alpha=alpha,
        origin=origin,
        aspect=aspect,
        interpolation=interpolation
    )
    ax.grid(show_grid)
    if hide_ticks == True:  # NOQA: E712
        ax.set_xticks([])
        ax.set_yticks([])
    if title is not None:
        ax.set_title(title, color=fontcolor)

    if colorbar == True:  # NOQA: E712
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        if colorbarlocation == 'right':
            cax = divider.append_axes("right", size="5%", pad=0.05, aspect=2.3 * aspect / 0.05)
            cbar = plt.colorbar(im, cax=cax, extendfrac=20, label=colorbarlabel, orientation='vertical')
            cbar.set_alpha(1)
            cbar.set_label(colorbarlabel, size=fontsize, rotation=90)
            cbar.draw_all()

        elif colorbarlocation == 'bottom':
            cax = divider.append_axes("bottom", size="5%", pad=0.05, aspect=1 / (2.3 * aspect / 0.05))
            cbar = plt.colorbar(im, cax=cax, extendfrac=20, label=colorbarlabel, orientation='horizontal')
            cbar.solids.set_edgecolor("face")
            cbar.set_label(colorbarlabel, size=fontsize)
        if colorbarticks is not None:
            cbar.set_ticks(colorbarticks)

    if returnval == 'axis':
        return ax
    elif returnval == 'image':
        return im


def initialize_legend(ax, colors, linewidth=1, linestyle='-', marker=None, markersize=8, alpha=1):
    """ initializes a legend on an axis to ensure that first entries match desired line colors

    Parameters
    ----------
    ax : matplotlib axis
        the axis to apply the legend to
    colors : list
        marker colors for the legend items
    linewdith : int, optional
        width of lines (default 1)
    linestyle : str, optional
        matplotlib linestyle (default '-')
    marker : str, optional
        matplotlib marker style (default None)
    markersize : int, optional
        matplotlib marker size (default 8)
    alpha : float, optional
        matplotlib opacity, varying from 0 to 1 (default 1)

    """
    for color in colors:
        ax.plot(np.nan, np.nan, color=color, linewidth=linewidth, linestyle=linestyle, marker=marker, markersize=markersize, alpha=alpha)


def placeAxesOnGrid(
        fig,
        dim=[1, 1],
        xspan=[0, 1],
        yspan=[0, 1],
        wspace=None,
        hspace=None,
        sharex=False,
        sharey=False,
        frameon=True
):
    '''
    Takes a figure with a gridspec defined and places an array of sub-axes on a portion of the gridspec
    DRO

    Takes as arguments:
        fig: figure handle - required
        dim: number of rows and columns in the subaxes - defaults to 1x1
        xspan: fraction of figure that the subaxes subtends in the x-direction (0 = left edge, 1 = right edge)
        yspan: fraction of figure that the subaxes subtends in the y-direction (0 = top edge, 1 = bottom edge)
        wspace and hspace: white space between subaxes in vertical and horizontal directions, respectively

    returns:
        subaxes handles
    '''
    import matplotlib.gridspec as gridspec

    outer_grid = gridspec.GridSpec(100, 100)
    inner_grid = gridspec.GridSpecFromSubplotSpec(
        dim[0],
        dim[1],
        subplot_spec=outer_grid[int(100 * yspan[0]):int(100 * yspan[1]), int(100 * xspan[0]):int(100 * xspan[1])],
        wspace=wspace,
        hspace=hspace
    )

    # NOTE: A cleaner way to do this is with list comprehension:
    # inner_ax = [[0 for ii in range(dim[1])] for ii in range(dim[0])]
    inner_ax = dim[0] * [dim[1] * [fig]]  # filling the list with figure objects prevents an error when it they are later replaced by axis handles
    inner_ax = np.array(inner_ax)
    idx = 0
    for row in range(dim[0]):
        for col in range(dim[1]):
            if row > 0 and sharex == True:  # NOQA: E712
                share_x_with = inner_ax[0][col]
            else:
                share_x_with = None

            if col > 0 and sharey == True:  # NOQA: E712
                share_y_with = inner_ax[row][0]
            else:
                share_y_with = None

            inner_ax[row][col] = plt.Subplot(
                fig,
                inner_grid[idx],
                sharex=share_x_with,
                sharey=share_y_with,
                frameon=frameon,
            )

            if row == dim[0] - 1 and sharex == True:  # NOQA: E712
                inner_ax[row][col].xaxis.set_ticks_position('bottom')
            elif row < dim[0] and sharex == True:  # NOQA: E712
                plt.setp(inner_ax[row][col].get_xtick)  # NOQA: E712

            if col == 0 and sharey == True:  # NOQA: E712
                inner_ax[row][col].yaxis.set_ticks_position('left')
            elif col > 0 and sharey == True:  # NOQA: E712
                plt.setp(inner_ax[row][col].get_yticklabels(), visible=False)

            fig.add_subplot(inner_ax[row, col])
            idx += 1

    inner_ax = np.array(inner_ax).squeeze().tolist()  # remove redundant dimension
    return inner_ax
