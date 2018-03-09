from __future__ import print_function
import six
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


def save_figure(fig, fname, formats=['.pdf'], transparent=False, dpi=300, facecolor=None, **kwargs):
    import matplotlib as mpl
    mpl.rcParams['pdf.fonttype'] = 42

    if 'size' in kwargs.keys():
        fig.set_size_inches(kwargs['size'])

    elif 'figsize' in kwargs.keys():
        fig.set_size_inches(kwargs['figsize'])
    else:
        fig.set_size_inches(fig.get_figwidth(), fig.get_figheight())
        # fig.set_size_inches(11, 8.5)
    for f in formats:
        fig.savefig(
            fname + f,
            transparent=transparent,
            orientation='landscape',
            dpi=dpi
        )



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
    if isinstance(input, six.string_types):
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

    if logscale == True:
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
            minval = 0.03 if minval is None else minval
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
    elif logscale == True and show_points is True:
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
            if logscale == False:
                l_fit = ax.plot(xp, pxp, linestyle=linestyle, linewidth=linewidth, color=linecolor, alpha=linealpha)
            else:
                l_fit = ax.plot(np.log10(xp), pxp, linestyle=linestyle, linewidth=linewidth, color=linecolor, alpha=linealpha)
        except Exception as e:
            print("failed to plot sigmoid", e)

    if showYLabel == True:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    else:
        ax.set_yticklabels([])
    if showXLabel == True:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    else:
        ax.set_xticklabels([])

    ax.set_ylim([-0.05, 1.05])
    ax.set_title(title, fontsize=fontsize + 1)
    ax.tick_params(labelsize=fontsize - 1)

    if returnvals == True:
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


def animate_array(array,
                  figsize=(9, 5),
                  clim=None,
                  cmap='gray',
                  saveloc=None,
                  dpi=300,
                  fps=10,
                  annotation=None,
                  annotation_location=(10, 30),
                  fontsize=10,
                  fontcolor='orange',
                  fontweight='normal',
                  axis='off',
                  repeat=False
                  ):
    '''
    takes in array (shape = NFrames x ROWS x COLUMNS), creates animation in axis

    Optional arguments:
        figsize,
        clim: range of colormap
        cmap
        saveloc: location to save movie to. If None, will only display locally
        dpi: quality of saved movie
        fps: speed of playback, both for saving and local display
        annotation: list of text values to print on frames
        annotation_location: location to print annotation text

    returns: nothing
    '''
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    n_frames = np.shape(array)[0]

    if annotation is not None:
        assert len(annotation) == n_frames, 'length of annotation must equal frame number'

    def update_array(frame):
        # check if animation is at the last frame, and if so, stop the animation

        if frame == n_frames:
            anim.event_source.stop()
        else:
            ax.cla()
            ax.imshow(array[frame, :, :], cmap=cmap, clim=clim)
            if annotation is not None:
                ax.text(
                    annotation_location[0],
                    annotation_location[1],
                    annotation[frame],
                    fontsize=fontsize,
                    color=fontcolor,
                    fontweight=fontweight
                )
            ax.axis(axis)

    fig, ax = plt.subplots(figsize=figsize)
    anim = animation.FuncAnimation(
        fig,
        update_array,
        frames=n_frames,
        interval=1000. / fps,
        repeat=repeat,
    )

    if saveloc is not None:
        anim.save(saveloc, writer='ffmpeg', fps=fps, dpi=dpi)


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
    if hide_ticks == True:
        ax.set_xticks([])
        ax.set_yticks([])
    if title is not None:
        ax.set_title(title, color=fontcolor)

    if colorbar == True:
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
            if row > 0 and sharex == True:
                share_x_with = inner_ax[0][col]
            else:
                share_x_with = None

            if col > 0 and sharey == True:
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

            if row == dim[0] - 1 and sharex == True:
                inner_ax[row][col].xaxis.set_ticks_position('bottom')
            elif row < dim[0] and sharex == True:
                plt.setp(inner_ax[row][col].get_xtick)

            if col == 0 and sharey == True:
                inner_ax[row][col].yaxis.set_ticks_position('left')
            elif col > 0 and sharey == True:
                plt.setp(inner_ax[row][col].get_yticklabels(), visible=False)

            fig.add_subplot(inner_ax[row, col])
            idx += 1

    inner_ax = np.array(inner_ax).squeeze().tolist()  # remove redundant dimension
    return inner_ax
