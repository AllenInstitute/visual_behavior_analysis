import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from colorsys import hls_to_rgb
from . import utilities as vbu


def generate_random_colors(n, lightness_range=(0, 1), saturation_range=(0, 1), random_seed=0, order_colors=False):
    '''
    get n distinct colors specified in HLS (Hue, Lightness, Saturation) colorspace
    hue is random
    lightness is random in range between all black (0) and all white (1)
    saturation is random in range between lightness value (0) and pure color (1)

    inputs:
        n (int) - number of desired colors
        lightness_range (2 value tuple) - desired range of lightness values (from 0 to 1)
        saturation_range (2 value tuple) - desired range of saturation values (from 0 to 1)
        random_seed (int) - seed for random number generator (ensures repeatability)
        order_colors (bool) - if True, colors will be ordered by hue, if False, hue order will be random

    returns:
        list of tuples containing RGB values (which can be used as a matplotlib palette)
    '''
    np.random.seed(random_seed)
    colors = []

    hues = np.random.rand(n)
    if order_colors:
        hues = np.sort(hues)

    for hue in hues:
        lightness = np.random.uniform(lightness_range[0], lightness_range[1])
        saturation = np.random.uniform(saturation_range[0], saturation_range[1])
        colors.append(hls_to_rgb(hue, lightness, saturation))

    return colors


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


def save_figure(
    fig,
    fname,
    formats=['.png'],
    transparent=False,
    dpi=300,
    **kwargs
):

    import matplotlib as mpl
    mpl.rcParams['pdf.fonttype'] = 42
    if 'size' in kwargs.keys():
        fig.set_size_inches(kwargs['size'])
    else:
        fig.set_size_inches(11, 8.5)
    for f in formats:
        fig.savefig(
            fname + f,
            transparent=transparent,
            orientation='landscape',
            dpi=dpi
        )


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
        ax.plot(np.nan, np.nan, color=color, linewidth=linewidth,
                linestyle=linestyle, marker=marker, markersize=markersize, alpha=alpha)


def get_fig_ax(fig, ax_index):
    '''
    a flexible method for getting figure and axis handles

    inputs:
        fig: matplotlib figure object
        ax_index: index of desired axis

    if fig is None
        will create a new figure and axis
        returns fig, ax

    if fig exists, but no axes exist:
        will create a single subplot axis
        returns fig, ax

    if fig exists and has associated axes:
        returns fig, ax[ax_index]
    '''
    if fig is None:
        fig, ax = plt.subplots()

    if fig and len(fig.get_axes()) == 0:
        ax = fig.add_subplot(111)
    else:
        ax = fig.get_axes()[ax_index]

    return fig, ax


def event_triggered_raster(df, x_value, event_times, fig=None, ax_index=0, var_name='', value_name='', t_before=10, t_after=10, plot_type='matplotlib', marker='|', color='black'):
    if plot_type == 'plotly':
        assert False, 'not yet implemented'
    elif plot_type == 'matplotlib':
        fig, ax = get_fig_ax(fig, ax_index)

        for ii, event_time in enumerate(event_times):
            query_string = "{0} > (@event_time - @t_before) and {0} < (@event_time + @t_after)".format(x_value)
            events = df.query(query_string)[x_value].values - event_time
            ax.plot(
                events,
                ii * np.ones_like(events),
                marker=marker,
                color=color,
                linestyle='none'
            )
        # assert False
        ax.set_xlabel(var_name)
        ax.set_ylabel(value_name)


def event_triggered_response_plot(df, x_value, y_values, fig=None, ax_index=0, var_name='', value_name='', plot_type='matplotlib', cmap='viridis'):
    if plot_type == 'plotly':
        return plotly_event_triggered_response_plot(df, x_value, y_values, var_name=var_name, value_name=value_name)
    elif plot_type == 'matplotlib':
        cmap = cm.get_cmap(cmap)
        fig, ax = get_fig_ax(fig, ax_index)

        for ii, col in enumerate(y_values):
            ax.plot(df[x_value], df[col], alpha=0.25,
                    color=cmap(ii / (len(y_values) - 1)))
        ax.plot(df[x_value], df[y_values].mean(
            axis=1), color='black', linewidth=3)

        ax.set_xlabel(var_name)
        ax.set_ylabel(value_name)

        return fig


def plotly_event_triggered_response_plot(df, x_value, y_values, var_name='line', value_name='value'):

    df_melted = pd.melt(
        df,
        id_vars=[x_value],
        value_vars=y_values,
        var_name=var_name,
        value_name=value_name
    )

    fig = px.line(
        df_melted,
        x=x_value,
        y=value_name,
        color=var_name,
        hover_name=var_name,
        render_mode="svg"
    )

    fig.add_trace(
        go.Scatter(
            x=df[x_value],
            y=df[y_values].mean(axis=1),
            line=dict(color='black', width=3),
            name='grand average'
        )
    )

    return fig


def designate_flashes(ax, omit=None, pre_color='blue', post_color='blue'):
    '''add vertical spans to designate stimulus flashes'''
    lims = ax.get_xlim()
    for flash_start in np.arange(0, lims[1], 0.75):
        if flash_start != omit:
            ax.axvspan(flash_start, flash_start + 0.25,
                       color=post_color, alpha=0.25, zorder=-np.inf)
    for flash_start in np.arange(-0.75, lims[0] - 0.001, -0.75):
        if flash_start != omit:
            ax.axvspan(flash_start, flash_start + 0.25,
                       color=pre_color, alpha=0.25, zorder=-np.inf)


def designate_flashes_plotly(fig, omit=None, pre_color='blue', post_color='blue', alpha=0.25, plotnumbers=[1], lims=[-10, 10]):
    '''add vertical spans to designate stimulus flashes'''

    post_flashes = np.arange(0, lims[1], 0.75)
    post_flash_colors = np.array([post_color] * len(post_flashes))
    pre_flashes = np.arange(-0.75, lims[0] - 0.001, -0.75)
    pre_flash_colors = np.array([pre_color] * len(pre_flashes))

    flash_times = np.hstack((pre_flashes, post_flashes))
    flash_colors = np.hstack((pre_flash_colors, post_flash_colors))

    shape_list = list(fig.layout.shapes)

    for plotnumber in plotnumbers:
        for flash_start, flash_color in zip(flash_times, flash_colors):
            if flash_start != omit:
                shape_list.append(
                    go.layout.Shape(
                        type="rect",
                        x0=flash_start,
                        x1=flash_start + 0.25,
                        y0=-100,
                        y1=100,
                        fillcolor=flash_color,
                        opacity=alpha,
                        layer="below",
                        line_width=0,
                        xref='x{}'.format(plotnumber),
                        yref='y{}'.format(plotnumber),
                    ),
                )

    fig.update_layout(shapes=shape_list)


def make_multi_cmap_heatmap(df, heatmap_defs, figsize=(6, 6), n_cbar_rows=2, cbar_spacing=5, top_buffer=0.1, bottom_buffer=0.1, heatmap_div=0.8, cbar_buffer=0.05):
    '''
    a function to make a heatmap where the colormaps are defined individually for all columns
    Useful for plotting heatmaps when scales/types of data vary across columns

    inputs:
        df (dataframe) - dataframe containing data to plot
        heatmap_defs (list of dictionaries) - a list of dictionaries specifiying the different heatmap parameters. Each dict contains the following key/value pairs:
            'columns': (required) a list of column names for which this parameter set applies
            'cbar_label': (required) label to apply to the colorbar
            'cmap': (optional) colormap to apply. Uses seaborn default if not specified
            'cbar_ticks': (optional) ticks for cbar. Uses seaborn default if not specified
            'cbar_ticklabels: (optional) labels for cbar_ticks. Must be same length as 'cbar_ticks'
            'vmin': (optional) lower limit for colormap. Uses seaborn default if not specified
            'vmax': (optional) upper limit for colormap. Uses seaborn default if not specified
        figsize (tuple): figsize default = (6,6))
        n_cbar_rows (int): number of rows of colorbars (default = 2)
        cbar_spacing (float): horizontal space betweewn colorbars (default = 5)
        top_buffer (float): fraction of figure canvas to leave blank at top (default = 0.1)
        bottom_buffer (float): fraction of figure canvas to leave blank at bottom (default = 0.1)
        heatmap_div (float, range of (0,1)): fraction of figure canvas to devote to heatmap (from left edge) (default = 0.6)
        cbar_buffer (float, range of (0,1)): fraction of figure canvas to devote to buffer between heatmap and cbars (default = 0.05)

    returns:
        fig, ax (fig = figure handle, ax = dictionary of axes with keys of 'heatmap' for axis containing heatmap, 'cmaps' for list of cmap axes)

    sample useage:

        ## generate some data to plot
        size=20
        np.random.seed(0)

        # add 10 rows of mean-zero random data
        data_dict = {'var_{}'.format(i):np.random.randn(size) for i in range(10)}

        # add some additional rows with very different scales
        data_dict.update({
            'binary':np.random.choice([0,1],size=size),
            'hundreds_1': np.random.choice(np.arange(0,100),size=size),
            'hundreds_2': np.random.choice(np.arange(0,100),size=size),
            'three_categories': np.random.choice([0,1,2],size=size)
        })

        # convert to a dataframe
        df = pd.DataFrame(data_dict)

        ## plot the data with varying colorbars
        heatmap_defs = [
            {
                'columns':[col for col in df if col.startswith('var')],
                'cbar_label':'zero mean data',
                'cbar_ticks':[-1,0,1],
                'vmin':-1,
                'vmax':1,
                'cmap':'viridis',
            },
            {
                'columns':['binary'],
                'cbar_label':'binary data',
                'cbar_ticks':[0,1],
                'cmap':sns.color_palette("magma", 2)
            },
            {
                'columns':['hundreds_1','hundreds_2'],
                'cbar_label':'hundreds',
                'cmap':'Reds'
            },
            {
                'columns':['three_categories'],
                'cbar_label':'three category data',
                'cbar_ticks':[0,1,2],
                'cmap':sns.color_palette("hls", 3)
            },
        ]

        fig, axes = make_multi_cmap_heatmap(df, heatmap_defs, figsize=(8,8))
    '''
    fig = plt.figure(figsize=figsize)
    axes = {
        'heatmap': placeAxesOnGrid(
            fig,
            xspan=[0, heatmap_div],
            yspan=[top_buffer, 1 - bottom_buffer]
        ),
        'cmaps': placeAxesOnGrid(
            fig,
            xspan=[heatmap_div + cbar_buffer, 1],
            yspan=[top_buffer, 1 - bottom_buffer],
            dim=[n_cbar_rows, np.int(np.ceil(len(heatmap_defs) / n_cbar_rows))],
            wspace=cbar_spacing
        ),
    }
    # special case for when the number of cbars is 1: axis needs to be cast to list
    if n_cbar_rows == np.int(np.ceil(len(heatmap_defs) / n_cbar_rows)) == 1:
        axes['cmaps'] = [axes['cmaps']]
    axes['cmaps'] = vbu.flatten_list(axes['cmaps'])

    for ii, sub_heatmap in enumerate(heatmap_defs):
        # make a copy of the dataframe so we can manipulate without affecting the original df
        df_temp = df.copy()
        # nan out all columns except the ones we want to plot
        cols_to_nan = [col for col in df.columns if col not in sub_heatmap['columns']]

        for col in cols_to_nan:
            df_temp[col] = np.nan

        sns.heatmap(
            df_temp,
            vmin=sub_heatmap['vmin'] if 'vmin' in sub_heatmap.keys() else None,
            vmax=sub_heatmap['vmax'] if 'vmax' in sub_heatmap.keys() else None,
            cmap=sub_heatmap['cmap'] if 'cmap' in sub_heatmap.keys() else None,
            ax=axes['heatmap'],
            cbar=True,
            cbar_ax=axes['cmaps'][ii],
            cbar_kws={
                'label': sub_heatmap['cbar_label'] if 'cbar_label' in sub_heatmap.keys() else None,
                'ticks': sub_heatmap['cbar_ticks'] if 'cbar_ticks' in sub_heatmap.keys() else None
            }
        )

        if 'cbar_ticklabels' in sub_heatmap.keys():
            axes['cmaps'][ii].set_yticklabels(sub_heatmap['cbar_ticklabels'])
    # turn off axes for any unused cbar axes
    for remaining_cbar_axes in range(ii + 1, len(axes['cmaps'])):
        axes['cmaps'][remaining_cbar_axes].axis('off')

    axes['heatmap'].set_xticklabels(axes['heatmap'].get_xticklabels(), rotation=45, ha='right')

    return fig, axes
