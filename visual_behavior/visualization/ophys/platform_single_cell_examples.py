"""
Created on Wednesday February 23 2022

@author: marinag
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import visual_behavior.visualization.utils as utils
import visual_behavior.data_access.loading as loading
import visual_behavior.data_access.utilities as utilities
import visual_behavior.visualization.ophys.summary_figures as sf

import mindscope_utilities.general_utilities as ms_utils
import mindscope_utilities.visual_behavior_ophys.data_formatting as vb_ophys

import visual_behavior_glm.GLM_params as glm_params
import visual_behavior_glm.GLM_analysis_tools as gat
import visual_behavior_glm.GLM_visualization_tools as gvt

from visual_behavior.dimensionality_reduction.clustering import processing
from visual_behavior.dimensionality_reduction.clustering import plotting

# formatting
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})
sns.set_style('white', {'axes.spines.right': False, 'axes.spines.top': False})
sns.set_palette('deep')




def load_GLM_outputs(glm_version, experiments_table, cells_table, glm_output_dir=None):
    """
    loads results_pivoted and weights_df from files in base_dir, or generates them from mongo and save to base_dir
    results_pivoted and weights_df will be limited to the ophys_experiment_ids and cell_specimen_ids present in experiments_table and cells_table
    because this function simply loads the results, any pre-processing applied to results_pivoted (such as across session normalization or signed weights) will be used here

    :param glm_version: example = '24_events_all_L2_optimize_by_session'
    :param glm_output_dir: directory containing GLM output files to load and save processed data files to for this iteration of the analysis
                            if None, GLM results will be obtained from mongo and will not be saved out
    :param experiments_table: SDK ophys_experiment table limited to experiments intended for analysis
    :param cells_table: SDK ophys_cell_table limited to cell_specimen_ids intended for analysis
    :return:
        results_pivoted: table of dropout scores for all cell_specimen_ids in cells_table
        weights_df: table with model weights for all cell_specimen_ids in cells_table
    """
    # get GLM kernels and params for this version of the model
    model_output_type = 'adj_fraction_change_from_full'
    run_params = glm_params.load_run_json(glm_version)
    kernels = run_params['kernels']
    # if glm_output_dir is not None:
    # load GLM results for all cells and sessions from file if it exists otherwise load from mongo
    glm_results_path = os.path.join(glm_output_dir, glm_version + '_results_pivoted.h5')
    if os.path.exists(glm_results_path):
        results_pivoted = pd.read_hdf(glm_results_path, key='df')
    else:
        print('no results_pivoted at', glm_results_path)
        print('please generate before running single cell plots')
    #     else:
    #         results_pivoted = gat.build_pivoted_results_summary(value_to_use=model_output_type, results_summary=None,
    #                                                             glm_version=glm_version, cutoff=None)
    #         # save for next time
    #         results_pivoted.to_hdf(glm_results_path, key='df')
    # else: # if no directory is provided, get results from mongo
    #     results_pivoted = gat.build_pivoted_results_summary(value_to_use=model_output_type, results_summary=None,
    #                                                         glm_version=glm_version, cutoff=None)

    # # clean results
    # results_pivoted = results_pivoted.reset_index()
    # print(len(results_pivoted.ophys_experiment_id.unique()), 'ophys_experiment_ids in results_pivoted after loading')
    # print(len(results_pivoted.cell_specimen_id.unique()), 'cell_specimen_ids in results_pivoted after loading')
    # # limit dropouts to experiments & cells in provided tables (limit input to last familiar and second novel to ensure results are also filtered)
    # results_pivoted = results_pivoted[results_pivoted.ophys_experiment_id.isin(experiments_table.index.values)]
    # results_pivoted = results_pivoted[results_pivoted.cell_specimen_id.isin(cells_table.cell_specimen_id.values)]
    # print(len(results_pivoted.ophys_experiment_id.unique()), 'ophys_experiment_ids in results_pivoted after filtering')
    # print(len(results_pivoted.cell_specimen_id.unique()), 'cell_specimen_ids in results_pivoted after filtering')
    # # clean up
    # results_pivoted = results_pivoted.drop_duplicates(subset=['cell_specimen_id', 'ophys_experiment_id'])
    # results_pivoted = results_pivoted.reset_index()

    # if glm_output_dir is not None: # if a directory is provided, attempt to load files
        # get weights df and clean
    weights_path = os.path.join(glm_output_dir, glm_version + '_weights_df.h5')
    if os.path.exists(weights_path): # if it exists, load it
        weights_df = pd.read_hdf(weights_path, key='df')
    else:
        print('no weights at', weights_path)
        print('please generate before running single cell plots')
        # else: # if it doesnt exist, generate it and save it
        #     # need to create results pivoted from scratch for weights_df
        #     full_results_pivoted = gat.build_pivoted_results_summary(value_to_use=model_output_type, results_summary=None,
        #                                                         glm_version=glm_version, cutoff=None)
        #     weights_df = gat.build_weights_df(run_params, full_results_pivoted)
        #     weights_df = weights_df.drop_duplicates(subset=['cell_specimen_id', 'ophys_experiment_id'])
        #     weights_df = weights_df.reset_index()
        #     weights_df['identifier'] = [
        #         str(weights_df.iloc[row]['ophys_experiment_id']) + '_' + str(weights_df.iloc[row]['cell_specimen_id']) for row
        #         in range(len(weights_df))]
        #     weights_df = weights_df.set_index('identifier')
        #     weights_df.to_hdf(weights_path, key='df')
    # else: # if no directory provided, build weights df
    #     weights_df = gat.build_weights_df(run_params, results_pivoted)

    # # filter and confirm cell #s
    # print(len(weights_df.cell_specimen_id.unique()), 'cell_specimen_ids in weights_df after loading')
    # # limit weights to experiments & cells in provided tables (limit input to last familiar and second novel to ensure results are also filtered)
    # weights_df = weights_df[weights_df.ophys_experiment_id.isin(experiments_table.index.values)]
    # weights_df = weights_df[weights_df.cell_specimen_id.isin(cells_table.cell_specimen_id.values)]
    # print(len(weights_df.cell_specimen_id.unique()), 'cell_specimen_ids in weights_df after filtering')

    return results_pivoted, weights_df, kernels


# functions to get x-axis info to plot kernels
def get_frame_rate_for_example_cell(weights_df, identifier):
    if 'MESO' in weights_df.loc[identifier]['equipment_name']:
        frame_rate = 11.
    else:
        frame_rate = 31.
    return frame_rate


def get_time_window_for_kernel(kernels, feature):
    time_window = (kernels[feature]['offset'], kernels[feature]['offset'] + kernels[feature]['length'])
    return time_window


def get_t_array_for_kernel(kernels, feature, frame_rate):
    time_window = get_time_window_for_kernel(kernels, feature)
    t_array = ms_utils.get_time_array(t_start=time_window[0], t_end=time_window[1], sampling_rate=frame_rate,
                                      include_endpoint=False)
    if 'image' in feature:
        t_array = t_array[:-1]  # not sure why we have to do this
    return t_array


def plot_cell_rois_and_GLM_weights(cell_specimen_id, cells_table, experiments_table, dropout_features, results_pivoted, weights_df,
                                  weights_features, kernels, save_dir=None, folder=None, data_type='dff'):
    """
    This function limits inputs just to the provided cell_specimen_id to hand off to the function plot_matched_roi_and_traces_example_GLM
    That function will plot the following panels:
        cell ROI masks matched across sessions for a given cell_specimen_id,
        change and omission triggered average respones across sessions,
        image locked running and pupil if included 'running' and 'pupil' in included weights_features
        dropout scores across dropout_features and sessions as a heatmap,
        kernels weights across sessions for the kernels in weights_features
    :param cell_specimen_id: cell_specimen_id for cell to plot
    :param cells_table: must only include a max of one one experiment per experience level for a given container. ok if less than 1.
    :param experiments_table: must only include one experiment per experience level for a given container
    :param results_pivoted: must only include one experiment per experience level for a given container
                            must be limited to features for plotting, plus cell_specimen_id + experiment_id
    :param weights_df: must only include one experiment per experience level for a given container
    :param weights_features: columns in weights_df to use for plotting
    :param save_dir: top level directory where files exist for this run of analysis
                        code will create a folder within save_dir called 'matched_cell_examples'
    :param subfolder: name of subfolder to create within os.path.join(save_dir, 'matched_cell_examples') to save plots, ex: 'cluster_0' or 'without_exp_var_full_model' or 'with_running_and_pupil'
    :param data_type: can be 'dff', 'events', or 'filtered_events' - to be used for cell response plots
    :return:
    """

    # make sure weights and dropouts are limited to matched experiments / cells
    cells_table = loading.get_cell_table(platform_paper_only=True, limit_to_closest_active=True,
                                         limit_to_matched_cells=True, add_extra_columns=True)
    experiments_table = loading.get_platform_paper_experiment_table(add_extra_columns=True, limit_to_closest_active=True)
    matched_cells = cells_table.cell_specimen_id.unique()
    matched_experiments = cells_table.ophys_experiment_id.unique()
    weights_df = weights_df[weights_df.ophys_experiment_id.isin(matched_experiments)]
    weights_df = weights_df[weights_df.cell_specimen_id.isin(matched_cells)]
    results_pivoted = results_pivoted.reset_index()  # reset just in case
    results_pivoted = results_pivoted[results_pivoted.cell_specimen_id.isin(matched_cells)]

    # get cell info
    cell_metadata = cells_table[cells_table.cell_specimen_id == cell_specimen_id]
    ophys_container_id = cell_metadata.ophys_container_id.unique()[0]
    ophys_experiment_ids = cell_metadata.ophys_experiment_id.unique()

    # folder = 'matched_cell_examples'
    # save_dir = os.path.join(save_dir, folder)
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)
    # if subfolder: # if an additional subfolder is provided, make the above folder the top level and create a subfolder within it for plots
    #     subfolder_dir = os.path.join(save_dir, subfolder)
    #     if not os.path.exists(subfolder_dir):
    #         os.mkdir(subfolder_dir)
    # else:
    #     subfolder = folder
    # get metadata for this cell
    cell_metadata = cells_table[cells_table.cell_specimen_id == cell_specimen_id]
    # get weights for example cell
    cell_weights = weights_df[weights_df.cell_specimen_id == cell_specimen_id]
    # # limit results_pivoted to features of interest
    # rspm = processing.limit_results_pivoted_to_features_for_clustering(results_pivoted, features=dropout_features)
    # # make dropouts positive for plotting
    # for feature in dropout_features:
    #     rspm[feature] = np.abs(rspm[feature])
    # # if exp var full model is in features (must be first feature), scale it by 10x so its on similar scale as dropouts
    if 'variance_explained_full' in results_pivoted.keys():
        results_pivoted['variance_explained_full'] = results_pivoted['variance_explained_full']*10
    # merge with metadata
    # dropouts = rspm.merge(cells_table[['cell_specimen_id', 'cell_type', 'binned_depth', 'targeted_structure']], on='cell_specimen_id')
    # get dropouts just for one cell
    cell_dropouts = results_pivoted[results_pivoted.cell_specimen_id == cell_specimen_id]

    plot_matched_roi_and_traces_example_GLM(cell_metadata, cell_dropouts, cell_weights, weights_features, kernels,
                                            dropout_features, experiments_table, data_type, save_dir, folder)


def plot_matched_roi_and_traces_example_GLM(cell_metadata, cell_dropouts, cell_weights, weights_features, kernels,
                                            dropout_features, experiments_table, data_type, save_dir=None, folder=None):
    """
    This function will plot the following panels:
        cell ROI masks matched across sessions for a given cell_specimen_id,
        change and omission triggered average respones across sessions,
        image locked running and pupil if included 'running' and 'pupil' in included weights_features,
        dropout scores across features and sessions as a heatmap,
        kernels weights across sessions for the kernels in weights_features.
    Plots the ROI masks and cell traces for a cell matched across sessions, along with dropout scores and weights for images, hits, misses and omissions
    Cell_metadata is a subset of the ophys_cells_table limited to the cell_specimen_id of interest
    cell_dropouts is a subset of the results_pivoted version of GLM output limited to cell_specimen_id of interest
    cell_weights is a subset of the weights matrix from GLM limited to cell_specimen_id of interest
    all input dataframes must be limited to last familiar and second novel active (i.e. max of one session per type)
    if one session type is missing, the max projection but no ROI will be plotted and the traces and weights will be missing for that experience level
    """

    if len(cell_metadata.cell_specimen_id.unique()) > 1:
        print('There is more than one cell_specimen_id in the provided cell_metadata table')
        print('Please limit input to a single cell_specimen_id')

    # set up plotting for each experience level
    experience_levels = ['Familiar', 'Novel 1', 'Novel >1']
    colors = utils.get_experience_level_colors()
    n_exp_levels = len(experience_levels)
    # get relevant info for this cell
    cell_metadata = cell_metadata.sort_values(by='experience_level')
    cell_type = cell_metadata.cell_type.unique()[0]
    cell_specimen_id = cell_metadata.cell_specimen_id.unique()[0]
    ophys_container_id = cell_metadata.ophys_container_id.unique()[0]
    cre_line = cell_metadata.cre_line.unique()[0]
    # need to get all experiments for this container, not just for this cell
    ophys_experiment_ids = experiments_table[experiments_table.ophys_container_id==ophys_container_id].index.values
    n_expts = len(ophys_experiment_ids)
    if n_expts > 3:
        print('There are more than 3 experiments for this cell. There should be a max of 1 experiment per experience level')
        print('Please limit input to only one experiment per experience level')

    # set up labels for different trace types
    if data_type == 'dff':
        ylabel = 'dF/F'
    else:
        ylabel = 'response'

    # number of columns is one for each experience level,
    # plus additional columns for stimulus and omission traces, and running and pupil averages (TBD)
    extra_cols = 2
    if 'running' in weights_features:
        extra_cols +=1
    if 'running' in weights_features:
        extra_cols += 1
    n_cols = n_exp_levels + extra_cols
    print(extra_cols, 'extra cols')

    figsize = (3.5 * n_cols, 6)
    fig, ax = plt.subplots(2, n_cols, figsize=figsize)
    ax = ax.ravel()

    print('cell_specimen_id:', cell_specimen_id)
    # loop through experience levels for this cell
    for e, experience_level in enumerate(experience_levels):
        print('experience_level:', experience_level)

        # ophys_experiment_id = cell_metadata[cell_metadata.experience_level == experience_level].ophys_experiment_id.values[0]
        # get ophys_experiment_id for this experience level
        # experiments_table must only include one experiment per experience level for a given container
        ophys_experiment_id = experiments_table[(experiments_table.ophys_container_id == ophys_container_id)&
                                                (experiments_table.experience_level==experience_level)].index.values[0]
        print('ophys_experiment_id:', ophys_experiment_id)
        ind = experience_levels.index(experience_level)
        color = colors[ind]

        # load dataset for this experiment
        dataset = loading.get_ophys_dataset(ophys_experiment_id, get_extended_stimulus_presentations=False)

        try:  # attempt to generate plots for this cell in this this experience level. if cell does not have this exp level, skip
            # plot ROI mask for this experiment
            ct = dataset.cell_specimen_table.copy()
            cell_roi_id = ct.loc[cell_specimen_id].cell_roi_id  # typically will fail here if the cell_specimen_id isnt in the session
            roi_masks = dataset.roi_masks.copy()  # save this to get approx ROI position if subsequent session is missing the ROI (fails if the first session is the one missing the ROI)
            ax[e] = sf.plot_cell_zoom(dataset.roi_masks, dataset.max_projection, cell_roi_id,
                                      spacex=50, spacey=50, show_mask=True, ax=ax[e])
            ax[e].set_title(experience_level, color=color)

            # get change responses and plot on second to next axis after ROIs (there are n_expts # of ROIs)
            window = [-1, 1.5]  # window around event
            sdf = loading.get_stimulus_response_df(dataset, time_window=window, interpolate=True, output_sampling_rate=30,
                                                   data_type=data_type, event_type='changes', load_from_file=True)
            cell_data = sdf[(sdf.cell_specimen_id == cell_specimen_id) & (sdf.is_change == True)]

            ax[n_expts] = utils.plot_mean_trace(cell_data.trace.values, cell_data.trace_timestamps.values[0],
                                                ylabel=ylabel, legend_label=None, color=color, interval_sec=1,
                                                xlim_seconds=window, plot_sem=True, ax=ax[n_expts])
            ax[n_expts] = utils.plot_flashes_on_trace(ax[n_expts], cell_data.trace_timestamps.values[0],
                                                      change=True, omitted=False)
            ax[n_expts].set_title('changes')

            # get omission responses and plot on last axis
            sdf = loading.get_stimulus_response_df(dataset, time_window=window, interpolate=True, output_sampling_rate=30,
                                                   data_type=data_type, event_type='omissions', load_from_file=True)
            cell_data = sdf[(sdf.cell_specimen_id == cell_specimen_id) & (sdf.omitted == True)]

            ax[n_expts + 1] = utils.plot_mean_trace(cell_data.trace.values, cell_data.trace_timestamps.values[0],
                                                    ylabel=ylabel, legend_label=None, color=color, interval_sec=1,
                                                    xlim_seconds=window, plot_sem=True, ax=ax[n_expts + 1])
            ax[n_expts + 1] = utils.plot_flashes_on_trace(ax[n_expts + 1], cell_data.trace_timestamps.values[0],
                                                          change=False, omitted=True)
            ax[n_expts + 1].set_title('omissions')

            if 'running' in weights_features:
                pass
            if 'pupil' in weights_features:
                pass

        except:  # plot area of max projection where ROI would have been if it was in this session
            # plot the max projection image with the xy location of the previous ROI
            # this will fail if the familiar session is the one without the cell matched
            print('no cell ROI for', experience_level)
            ax[e] = sf.plot_cell_zoom(roi_masks, dataset.max_projection, cell_roi_id,
                                      spacex=50, spacey=50, show_mask=False, ax=ax[e])
            ax[e].set_title(experience_level)


        # try: # try plotting GLM outputs for this experience level
        if 'running' in weights_features:
            pass
        if 'pupil' in weights_features:
            pass

        # GLM plots start after n_expts for each ROI mask, plus n_extra_cols more axes for omission and change responses (and running and pupil if added)
        # plus one more axes for dropout heatmaps
        i = n_expts + extra_cols + 1

        # weights
        exp_weights = cell_weights[cell_weights.experience_level == experience_level]

        # image kernels
        image_weights = []
        for f, feature in enumerate(weights_features[:8]): # first 8 are images
            image_weights.append(exp_weights[feature + '_weights'].values[0])
        mean_image_weights = np.mean(image_weights, axis=0)

        # frame_rate = get_frame_rate_for_example_cell(cell_weights, identifier=cell_weights.index.values[0])
        # GLM output is all resampled to 30Hz now
        frame_rate = 31
        t_array = get_t_array_for_kernel(kernels, feature, frame_rate)
        ax[i].plot(t_array, mean_image_weights, color=color)
        ax[i].set_ylabel('weight')
        ax[i].set_title('images')
        ax[i].set_xlabel('time (s)')
        ax_to_share = i

        i += 1
        # all other kernels
        for f, feature in enumerate(weights_features[8:]):
            if f == 0:
                first_ax = f
            kernel_weights = exp_weights[feature + '_weights'].values[0]
            if feature == 'omissions':
                n_frames_to_clip = int(kernels['omissions']['length'] * frame_rate) + 1
                kernel_weights = kernel_weights[:n_frames_to_clip]
            t_array = get_t_array_for_kernel(kernels, feature, frame_rate)
            ax[i + f].plot(t_array, kernel_weights, color=color)
            ax[i + f].set_ylabel('')
            ax[i + f].set_title(feature)
            ax[i + f].set_xlabel('time (s)')
            ax[i + f].get_shared_y_axes().join(ax[i + f], ax[ax_to_share])

        # except:
        #     print('could not plot GLM results for', experience_level)

    # try:
    # plot dropout score heatmaps
    i = n_expts + extra_cols  # change to extra_cols = 4 if running and pupil are added
    # cell_dropouts['cre_line'] = cre_line
    cell_dropouts = cell_dropouts.groupby(['experience_level']).mean()
    if 'ophys_experiment_id' in cell_dropouts.keys():
        cell_dropouts = cell_dropouts.drop(columns='ophys_experiment_id')
    if 'cell_specimen_id' in cell_dropouts.keys():
        cell_dropouts = cell_dropouts.drop(columns='cell_specimen_id')
    cell_dropouts = cell_dropouts[dropout_features] # order dropouts properly
    dropouts = cell_dropouts.T
    if len(np.where(dropouts < 0)[0]) > 0:
        vmin = -1
        cmap = 'RdBu'
    else:
        vmin = 0
        cmap = 'Blues'
    ax[i] = sns.heatmap(dropouts, cmap=cmap, vmin=vmin, vmax=1, ax=ax[i], cbar=False)
    # ax[i].set_title('coding scores')
    ax[i].set_yticklabels(dropouts.index.values, rotation=0, fontsize=14)
    ax[i].set_xticklabels(dropouts.columns.values, rotation=90, fontsize=14)
    ax[i].set_ylim(0, dropouts.shape[0])
    ax[i].set_xlabel('')
    # ax[i] = plotting.plot_dropout_heatmap(cell_metadata, feature_matrix, cre_line, cluster_id, small_fontsize=False, ax=ax[i])
    # except:
    #     print('could not plot dropout heatmap for', cell_specimen_id)

    # metadata_string = utils.get_container_metadata_string(dataset.metadata)

    # fig.tight_layout()
    # fig.subplots_adjust(hspace=0.6, wspace=0.7)
    # fig.suptitle(str(cell_specimen_id) + '_' + metadata_string, x=0.53, y=1.02,
    #              horizontalalignment='center', fontsize=16)

    if save_dir:
        print('saving plot for', cell_specimen_id)
        utils.save_figure(fig, figsize, save_dir, folder, str(cell_specimen_id) + '_' + metadata_string + '_' + data_type)
        print('saved')
