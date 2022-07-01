import visual_behavior.visualization.utils as utils
import visual_behavior.visualization.qc.experiment_plots as ep
import visual_behavior.ophys.response_analysis.response_analysis as ra
import visual_behavior.ophys.response_analysis.utilities as ut
import visual_behavior.data_access.loading as loading
import figrid as ff
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})


def make_fig_ax():
    figsize = (42, 30)
    fig = plt.figure(figsize=figsize)
    ax = {
        '0_0': ff.place_axes_on_grid(fig, xspan=[0, .1], yspan=[0, 0.15]),
        '0_1': ff.place_axes_on_grid(fig, xspan=[0.12, 0.22], yspan=[0, 0.15]),
        '0_2': ff.place_axes_on_grid(fig, xspan=[0.24, 0.34], yspan=[0, 0.15]),
        '0_3:': ff.place_axes_on_grid(fig, xspan=[0.45, 1], yspan=[0, 0.15], dim=[3, 1], hspace=0.2),

        '1_0': ff.place_axes_on_grid(fig, xspan=[0, .1], yspan=[0.2, 0.35]),
        '1_1': ff.place_axes_on_grid(fig, xspan=[0.12, 0.22], yspan=[0.2, 0.35]),
        '1_2': ff.place_axes_on_grid(fig, xspan=[0.24, 0.34], yspan=[0.2, 0.35]),
        '1_3:': ff.place_axes_on_grid(fig, xspan=[0.45, 1], yspan=[0.2, 0.35], dim=[3, 1], hspace=0.2),

        '2_0': ff.place_axes_on_grid(fig, xspan=[0, .1], yspan=[0.4, 0.55]),
        '2_1': ff.place_axes_on_grid(fig, xspan=[0.12, 0.22], yspan=[0.4, 0.55]),
        '2_2': ff.place_axes_on_grid(fig, xspan=[0.24, 0.34], yspan=[0.4, 0.55]),
        '2_3:': ff.place_axes_on_grid(fig, xspan=[0.45, 1], yspan=[0.4, 0.55]),

        '3_0': ff.place_axes_on_grid(fig, xspan=[0, .16], yspan=[0.6, 0.75]),
        '3_1': ff.place_axes_on_grid(fig, xspan=[0.22, 0.38], yspan=[0.6, 0.75]),
        #         '3_2': ff.place_axes_on_grid(fig, xspan=[0.3, 0.45], yspan=[0.6, 0.75]),
        '3_3:': ff.place_axes_on_grid(fig, xspan=[0.45, 1], yspan=[0.6, 0.75], dim=[6, 1], hspace=0),

        '4_0': ff.place_axes_on_grid(fig, xspan=[0, .16], yspan=[0.8, 0.95]),
        '4_1': ff.place_axes_on_grid(fig, xspan=[0.22, 0.38], yspan=[0.8, 0.95]),
        #         '4_2': ff.place_axes_on_grid(fig, xspan=[0.3, 0.45], yspan=[0.8, 0.95]),
        '4_3:': ff.place_axes_on_grid(fig, xspan=[0.45, 1], yspan=[0.8, 0.95], dim=[3, 1], hspace=0.1)
        #         '5_0': ff.place_axes_on_grid(fig, xspan=[0, 0.3], yspan=[0.75, 0.9]),
        #         '5_1': ff.place_axes_on_grid(fig, xspan=[0.4, 1], yspan=[0.75, 0.9]),
        #         '5_2': ff.place_axes_on_grid(fig, xspan=[0, 0.4], yspan=[0.75, 0.9]),
        #         '5_3:': ff.place_axes_on_grid(fig, xspan=[0.5, 1], yspan=[0.75, 0.9], dim=[3, 1], hspace=0.4)
    }

    return fig, ax, figsize


def plot_experiment_summary_figure(experiment_id, save_figure=True):

    dataset = loading.get_ophys_dataset(experiment_id)

    fig, ax, figsize = make_fig_ax()
    ax['0_0'] = ep.plot_max_intensity_projection_for_experiment(experiment_id, ax=ax['0_0'])
    ax['0_0'].set_title('max projection')
    ax['0_1'] = ep.plot_valid_segmentation_mask_outlines_per_cell_for_experiment(experiment_id, ax=ax['0_1'])
    # ax['0_0'].set_title('max projection')
    ax['0_2'] = ep.plot_valid_and_invalid_segmentation_mask_overlay_per_cell_for_experiment(experiment_id, ax=ax['0_2'])
    ax['0_2'].set_title('red = valid ROIs, blue = invalid ROIs')
    ax['0_3:'] = ep.plot_motion_correction_and_population_average(experiment_id, ax=ax['0_3:'])

    ax['1_0'] = ep.plot_average_image_for_experiment(experiment_id, ax=ax['1_0'])
    ax['1_1'] = ep.plot_average_image_for_experiment(experiment_id, ax=ax['1_1'])
    try:
        ax['1_2'] = ep.plot_remaining_decrosstalk_masks_for_experiment(experiment_id, ax=ax['1_2'])
    except BaseException:
        print('no decrosstalk for experiment', experiment_id)
    ax['1_3:'] = ep.plot_behavior_timeseries_for_experiment(experiment_id, ax=ax['1_3:'])

    # ax['2_0'] = population_image_selectivity(experiment_id, ax=ax['2_0'])
    # ax['2_0'] = ep.plot_average_image_for_experiment(experiment_id, ax=ax['2_1'])

    ax['2_2'] = ep.plot_cell_snr_distribution_for_experiment(experiment_id, ax=ax['2_2'])
    ax['2_3:'] = ep.plot_traces_heatmap_for_experiment(experiment_id, ax=ax['2_3:'])

    try:
        df_name = 'trials_response_df'
        df = analysis.get_response_df(df_name)
        mean_df = ut.get_mean_df(df, analysis=analysis, conditions=['cell_specimen_id', 'go'], flashes=False, omitted=False,
                                 get_reliability=False, get_pref_stim=False, exclude_omitted_from_pref_stim=True)
        ax['3_0'] = ep.plot_population_average_for_experiment(experiment_id, df, mean_df, df_name, color=None, label=None, ax=ax['3_0'])
        ax['3_0'].set_xlim(-2.5, 2.8)

        df_name = 'omission_response_df'
        df = analysis.get_response_df(df_name)
        mean_df = ut.get_mean_df(df, analysis=analysis, conditions=['cell_specimen_id'], flashes=False, omitted=True,
                                 get_reliability=False, get_pref_stim=False, exclude_omitted_from_pref_stim=False)
        ax['3_1'] = ep.plot_population_average_for_experiment(experiment_id, df, mean_df, df_name, color=None, label=None, ax=ax['3_1'])
        ax['3_1'].set_xlim(-2.5, 2.8)

        df_name = 'trials_run_speed_df'
        df = analysis.get_response_df(df_name)
        df['condition'] = True
        mean_df = ut.get_mean_df(df, analysis=analysis, conditions=['condition', 'go'], flashes=False, omitted=True,
                                 get_reliability=False, get_pref_stim=False, exclude_omitted_from_pref_stim=False)
        ax['4_0'] = ep.plot_population_average_for_experiment(experiment_id, df, df, df_name, trace_type='trace',
                                                              color=sns.color_palette()[4], label=None, ax=ax['4_0'])
        ax['4_0'].set_ylabel('run speed (cm/s)')
        ax['4_0'].set_xlim(-2.5, 2.8)

        df_name = 'omission_run_speed_df'
        df = analysis.get_response_df(df_name)
        df['condition'] = True
        mean_df = ut.get_mean_df(df, analysis=analysis, conditions=['condition'], flashes=False, omitted=False,
                                 get_reliability=False, get_pref_stim=True, exclude_omitted_from_pref_stim=True)
        ax['4_1'] = ep.plot_population_average_for_experiment(experiment_id, df, df, df_name, trace_type='trace',
                                                              color=sns.color_palette()[4], label=None, ax=ax['4_1'])
        ax['4_1'].set_ylabel('run speed (cm/s)')
        ax['4_1'].set_xlim(-2.5, 2.8)
    except:
        print('cant plot mean responses - need to update to work with mindscope_utilities')

    xlim_seconds = [int(10 * 60), int(15 * 60)]
    ax['3_3:'] = ep.plot_high_low_snr_trace_examples(experiment_id, xlim_seconds=xlim_seconds, ax=ax['3_3:'])

    ax['4_3:'] = ep.plot_behavior_timeseries_for_experiment(experiment_id, xlim_seconds=xlim_seconds, ax=ax['4_3:'])

    fig.tight_layout()
    title = ep.get_metadata_string(dataset)
    plt.suptitle(title, x=0.5, y=.91, fontsize=20)
    if save_figure:
        # save_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\qc_plots\experiment_plots'
        save_dir = loading.get_experiment_plots_dir()
        utils.save_figure(fig, figsize, save_dir, 'experiment_summary_figure', title)


if __name__ == '__main__':

    experiments_table = loading.get_filtered_ophys_experiment_table()
    experiment_id = experiments_table.index[60]
    plot_experiment_summary_figure(experiment_id, save_figure=True)
