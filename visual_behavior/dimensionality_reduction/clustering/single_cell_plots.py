
import os
import numpy as np
import pandas as pd
import matplotlib.colors
import matplotlib.pyplot as plt

import visual_behavior.visualization.utils as utils

import visual_behavior.data_access.loading as loading
import visual_behavior.data_access.utilities as utilities
from visual_behavior.dimensionality_reduction.clustering import plotting
from visual_behavior.dimensionality_reduction.clustering import processing

import seaborn as sns
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 1})




def plot_single_cell_coding_heatmap(cluster_id, cell_specimen_id, feature_matrix, cells_table, save_dir=None, folder=None, ax=None):
    '''
    plots one row of the coding score heatmap, i.e. coding scores for a single cell across experience levels & features
    each exp level / feature is a box, colored based on the strength of coding for that experience level

    :param cluster_id:
    :param cell_specimen_id:
    :param feature_matrix:
    :param cells_table:
    :param save_dir:
    :param folder:
    :param ax:
    :return:
    '''
    cell_data = feature_matrix.loc[cell_specimen_id]
    cell_data = pd.DataFrame(cell_data).reset_index().rename(
        columns={cell_specimen_id: 'coding_score', 'level_0': 'feature'})

    # get experience level labels
    new_labels = plotting.get_clean_labels_for_coding_scores_df(feature_matrix, columns=True)
    features = cell_data.feature.unique()
    # get abbreviated cre line
    cre_line = cells_table[cells_table.cell_specimen_id == cell_specimen_id].cre_line.values[0]
    cell_type = utils.get_abbreviated_cell_type(cre_line)

    # get data to plot
    cell_data = cell_data.set_index(['feature', 'experience_level']).T
    cell_data, cmap, vmax = plotting.remap_coding_scores_to_session_colors(cell_data)
    cell_data = cell_data.melt()
    cell_data = cell_data.set_index(['feature', 'experience_level'])

    figsize = (5, 2)
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(cell_data.T, cmap=cmap, vmin=0, vmax=vmax, square=True, linewidths=1, linecolor='black', ax=ax,
                     cbar=False)
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    ax.set_xticklabels(new_labels, rotation=0)
    ax.set_xlabel('')
    ax.set_ylabel('')

    # colorize y axis labels
    plotting.color_yaxis_labels_by_experience(ax)

    x_start = -0.8
    rotation = 0
    fontsize = 14
    features = processing.get_features_for_clustering()
    for i, feature in enumerate(features):
        if feature == 'all-images':
            features[i] = 'images'
    ax.text(s=features[0], y=x_start + 0.25, x=1.47, rotation=rotation, color='black', fontsize=fontsize, va='center',ha='center')
    ax.text(s=features[1], y=x_start + 0.25, x=4.45, rotation=rotation, color='black', fontsize=fontsize, va='center', ha='center')
    ax.text(s=features[2], y=x_start + 0.25, x=7.5, rotation=rotation, color='black', fontsize=fontsize, va='center', ha='center')
    ax.text(s=features[3], y=x_start + 0.25, x=10.5, rotation=rotation, color='black', fontsize=fontsize,va='center', ha='center')

    plt.suptitle(str(cell_specimen_id) + ' ' + cell_type + ' cluster ' + str(cluster_id + 1), x=0.5, y=0.9,fontsize=14)
    if save_dir:
        utils.save_figure(fig, figsize, os.path.join(save_dir, folder), 'cell_examples',
                          'cluster_' + str(cluster_id + 1) + '_' + str(cell_specimen_id) + '_coding_scores_heatmap')
    return ax


def plot_coding_scores_example_cell(cluster_id, cell_specimen_id, feature_matrix, cells_table,
                                    single_axis=True, save_dir=None, folder=None, ax=None):
    '''
    plot barplot of coding scores for each feature category across experience levels
    cell_specimen_id provided must be a cell that is in all 3 experience levels and is in the feature_matrix dataframe
    '''
    experience_level_colors = utils.get_experience_level_colors()

    # get coding scores for this cell
    cell_data = feature_matrix.loc[cell_specimen_id]
    cell_data = pd.DataFrame(cell_data).reset_index().rename(
        columns={cell_specimen_id: 'coding_score', 'level_0': 'feature'})

    # get experience level labels
    new_labels = plotting.get_clean_labels_for_coding_scores_df(feature_matrix, columns=True)
    new_labels = new_labels[:3]
    features = cell_data.feature.unique()
    # get abbreviated cre line
    cre_line = cells_table[cells_table.cell_specimen_id == cell_specimen_id].cre_line.values[0]
    cell_type = utils.get_abbreviated_cell_type(cre_line)

    if single_axis:
        suffix = '_single_axis_v'
        features = ['images', 'omissions', 'task', 'behavior']

        # vertical
        if ax is None:
            figsize = (5, 2)
            fig, ax = plt.subplots(figsize=figsize)
        ax = sns.barplot(data=cell_data, x='feature', y='coding_score', hue='experience_level',
                         palette=experience_level_colors, orient='v', width=0.5, ax=ax)
        ax.legend(fontsize='xx-small')
        ax.set_xticklabels(features, rotation=0)
        ax.set_xlabel('')
        ax.set_ylim(0, 1)
        ax.set_ylabel('coding score')

        ax.set_title('cluster ' + str(cluster_id + 1) + '\n' + cell_type + ' cell')

        plt.suptitle('csid: ' + str(cell_specimen_id), x=0.5, y=1.3, fontsize=14)

    else:
        suffix = ''
        if ax is None:
            figsize = (9, 2)
            fig, ax = plt.subplots(1, 4, figsize=figsize, sharex=True, sharey=True)
            add_suptitle = True
        else:
            add_suptitle = False
        for i, feature in enumerate(features):
            feature_data = cell_data[cell_data.feature == feature]
            ax[i] = sns.barplot(data=feature_data, x='experience_level', y='coding_score',
                                palette=experience_level_colors, width=0.5, alpha=0.9, ax=ax[i])
            ax[i].set_title(feature)
            ax[i].set_xticklabels(new_labels)
            ax[i].set_ylabel('')
            ax[i].set_xlabel('')
            ax[i].set_ylim(0, 1)
        ax[0].set_ylabel('coding score')

        if add_suptitle:
            plt.suptitle(str(cell_specimen_id) + ' ' + cell_type + ' cluster ' + str(cluster_id + 1), x=0.5, y=1.3,
                         fontsize=14)
    if save_dir:
        utils.save_figure(fig, figsize, os.path.join(save_dir, folder), 'cell_examples',
                          'cluster_' + str(cluster_id + 1) + '_' + str(cell_specimen_id) + '_coding_scores' + suffix)
    return ax


def plot_matched_cell_traces(cluster_id, cell_specimen_id, image_mdf, omission_mdf, change_mdf, cells_table, save_dir=None, folder=None, ax=None):
    """
    plot the average image, change, and omission traces for a matched cell
    must provide multi_session_response dataframes limited to all image presentations (image_mdf),
    omissions (omission_mdf) and changes (change_mdf)
    """
    ylabel = 'response'
    experience_levels = np.sort(omission_mdf.experience_level.unique())
    experience_level_colors = utils.get_experience_level_colors()

    if ax is None:
        figsize = (9, 2)
        fig, ax = plt.subplots(1, 3, figsize=figsize, sharey=True)
        add_suptitle = True
    else:
        add_suptitle = False

    i = 0
    for e, experience_level in enumerate(experience_levels):
        color = experience_level_colors[e]
        cell_data = image_mdf[(image_mdf.cell_specimen_id==cell_specimen_id) & (image_mdf.experience_level==experience_level)]
        ax[i] = utils.plot_mean_trace_from_mean_df(cell_data,  ylabel=ylabel, legend_label=None,
                                                    color=color, interval_sec=0.5, xlabel = 'time (s)',
                                                        xlims=[-0.5, 0.75], plot_sem=True, ax=ax[i])
        ax[i] = utils.plot_flashes_on_trace(ax[i], cell_data.trace_timestamps.values[0], change=False, omitted=False)
        ax[i].set_title('images')

    i = 1
    for e, experience_level in enumerate(experience_levels):
        color = experience_level_colors[e]
        cell_data = omission_mdf[(omission_mdf.cell_specimen_id==cell_specimen_id) & (omission_mdf.experience_level==experience_level)]
        ax[i] = utils.plot_mean_trace_from_mean_df(cell_data,  ylabel=ylabel, legend_label=None,
                                                    color=color, interval_sec=1, xlabel = 'time (s)',
                                                        xlims=[-1, 1.5], plot_sem=True, ax=ax[i])
        ax[i] = utils.plot_flashes_on_trace(ax[i], cell_data.trace_timestamps.values[0], change=False, omitted=True)
        ax[i].set_ylabel('')
        ax[i].set_title('omissions')

    i = 2
    for e, experience_level in enumerate(experience_levels):
        color = experience_level_colors[e]
        cell_data = change_mdf[(change_mdf.cell_specimen_id==cell_specimen_id) & (change_mdf.experience_level==experience_level)]
        ax[i] = utils.plot_mean_trace_from_mean_df(cell_data,  ylabel=ylabel, legend_label=None,
                                                    color=color, interval_sec=0.5, xlabel = 'time (s)',
                                                        xlims=[-1, 0.75], plot_sem=True, ax=ax[i])
        ax[i] = utils.plot_flashes_on_trace(ax[i], cell_data.trace_timestamps.values[0], change=True, omitted=False)
        ax[i].set_ylabel('')
        ax[i].set_title('changes')

    ax[0].set_ylabel(ylabel)

    if add_suptitle:
        plt.subplots_adjust(wspace=0.2)
        # get abbreviated cre line
        cre_line = cells_table[cells_table.cell_specimen_id==cell_specimen_id].cre_line.values[0]
        cell_type = utils.get_abbreviated_cell_type(cre_line)
        plt.suptitle(str(cell_specimen_id)+' '+cell_type+' cluster '+str(cluster_id+1), x=0.5, y=1.3, fontsize=14)
    if save_dir:
        utils.save_figure(fig, figsize, os.path.join(save_dir, folder), 'cell_examples', 'cluster_'+str(cluster_id+1)+'_'+str(cell_specimen_id)+'_traces')
    return ax


def plot_matched_cell_traces(cluster_id, cell_specimen_id, image_mdf, omission_mdf, change_mdf, cells_table, save_dir=None, folder=None, ax=None):
    """
    plot the average image, change, and omission traces for a matched cell

    """
    ylabel = 'response'
    experience_levels = np.sort(omission_mdf.experience_level.unique())
    experience_level_colors = utils.get_experience_level_colors()

    if ax is None:
        figsize = (9, 2)
        fig, ax = plt.subplots(1, 3, figsize=figsize, sharey=True)
        add_suptitle = True
    else:
        add_suptitle = False

    i = 0
    for e, experience_level in enumerate(experience_levels):
        color = experience_level_colors[e]
        cell_data = image_mdf[(image_mdf.cell_specimen_id==cell_specimen_id) & (image_mdf.experience_level==experience_level)]
        ax[i] = utils.plot_mean_trace_from_mean_df(cell_data,  ylabel=ylabel, legend_label=None,
                                                    color=color, interval_sec=0.5, xlabel = 'time (s)',
                                                        xlims=[-0.5, 0.75], plot_sem=True, ax=ax[i])
        ax[i] = utils.plot_flashes_on_trace(ax[i], cell_data.trace_timestamps.values[0], change=False, omitted=False)
        ax[i].set_title('images')

    i = 1
    for e, experience_level in enumerate(experience_levels):
        color = experience_level_colors[e]
        cell_data = omission_mdf[(omission_mdf.cell_specimen_id==cell_specimen_id) & (omission_mdf.experience_level==experience_level)]
        ax[i] = utils.plot_mean_trace_from_mean_df(cell_data,  ylabel=ylabel, legend_label=None,
                                                    color=color, interval_sec=1, xlabel = 'time (s)',
                                                        xlims=[-1, 1.5], plot_sem=True, ax=ax[i])
        ax[i] = utils.plot_flashes_on_trace(ax[i], cell_data.trace_timestamps.values[0], change=False, omitted=True)
        ax[i].set_ylabel('')
        ax[i].set_title('omissions')

    i = 2
    for e, experience_level in enumerate(experience_levels):
        color = experience_level_colors[e]
        cell_data = change_mdf[(change_mdf.cell_specimen_id==cell_specimen_id) & (change_mdf.experience_level==experience_level)]
        ax[i] = utils.plot_mean_trace_from_mean_df(cell_data,  ylabel=ylabel, legend_label=None,
                                                    color=color, interval_sec=0.5, xlabel = 'time (s)',
                                                        xlims=[-1, 0.75], plot_sem=True, ax=ax[i])
        ax[i] = utils.plot_flashes_on_trace(ax[i], cell_data.trace_timestamps.values[0], change=True, omitted=False)
        ax[i].set_ylabel('')
        ax[i].set_title('changes')

    ax[0].set_ylabel(ylabel)

    if add_suptitle:
        plt.subplots_adjust(wspace=0.2)
        # get abbreviated cre line
        cre_line = cells_table[cells_table.cell_specimen_id==cell_specimen_id].cre_line.values[0]
        cell_type = utils.get_abbreviated_cell_type(cre_line)
        plt.suptitle(str(cell_specimen_id)+' '+cell_type+' cluster '+str(cluster_id+1), x=0.5, y=1.3, fontsize=14)
    if save_dir:
        utils.save_figure(fig, figsize, os.path.join(save_dir, folder), 'cell_examples', 'cluster_'+str(cluster_id+1)+'_'+str(cell_specimen_id)+'_traces')
    return ax


def plot_matched_cell_rois(cluster_id, cell_specimen_id, cells_table, save_dir=None, folder=None, ax=None):
    """
    plot the cell mask and max projection image for the 3 experience levels the cell is matched in,

    """
    import visual_behavior.visualization.ophys.summary_figures as sf

    experience_levels = np.sort(cells_table.experience_level.unique())

    if ax is None:
        figsize = (9, 2)
        fig, ax = plt.subplots(1, 3, figsize=figsize, )
        add_suptitle = True
    else:
        add_suptitle = False
    experience_levels = cells_table.experience_level.unique()
    experience_level_colors = utils.get_experience_level_colors()
    for i, experience_level in enumerate(experience_levels):
        color = experience_level_colors[i]
        # get experiment dataset for this experience level
        this_cell_metadata = cells_table[
            (cells_table.cell_specimen_id == cell_specimen_id) & (cells_table.experience_level == experience_level)]
        ophys_experiment_id = this_cell_metadata.ophys_experiment_id.values[0]
        cell_roi_id = this_cell_metadata.index.values[0]
        dataset = loading.get_ophys_dataset(ophys_experiment_id, get_extended_stimulus_presentations=False)
        ax[i] = sf.plot_cell_zoom(dataset.roi_masks, dataset.max_projection, cell_roi_id,
                                  spacex=50, spacey=50, show_mask=True, ax=ax[i])
        ax[i].set_title(experience_level, color=color)

    if add_suptitle:
        plt.subplots_adjust(wspace=0.5)
        # get abbreviated cre line
        cre_line = cells_table[cells_table.cell_specimen_id == cell_specimen_id].cre_line.values[0]
        cell_type = utils.get_abbreviated_cell_type(cre_line)
        plt.suptitle(str(cell_specimen_id) + ' ' + cell_type + ' cluster ' + str(cluster_id + 1), x=0.5, y=1.3,
                     fontsize=14)
    if save_dir:
        utils.save_figure(fig, figsize, os.path.join(save_dir, folder), 'cell_examples',
                          'cluster_' + str(cluster_id + 1) + '_' + str(cell_specimen_id) + '_rois')
    return ax






if __name__ == '__main__':

    ### set paths

    base_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\platform_paper_plots\figure_4'

    save_dir = os.path.join(base_dir, 'all_cre_clustering_113023')
    folder = 'figs'

    # load matched cells table
    cells_table = loading.get_cell_table()
    cells_table = loading.get_matched_cells_table(cells_table)
    # update experience level names
    cells_table['experience_level'] = [utils.convert_experience_level(experience_level) for experience_level in cells_table.experience_level.values]
    matched_cells = cells_table.cell_specimen_id.unique()
    matched_experiments = cells_table.ophys_experiment_id.unique()

    # get cre_lines and cell types for plot labels
    cre_lines = np.sort(cells_table.cre_line.unique())
    cell_types = utils.get_cell_types()
    cre_line_colors = utils.get_cre_line_colors()

    # get other useful info
    experience_levels = utils.get_new_experience_levels()
    experience_level_colors = utils.get_experience_level_colors()

    ### get GLM results ###

    # get full un-normalized results table to use for selecting cells with high exp var to plot
    import visual_behavior_glm.GLM_fit_dev as gfd
    glm_version = '24_events_all_L2_optimize_by_session'
    run_params, all_results, all_results_pivoted, weights_df = gfd.load_analysis_dfs(glm_version)

    # get across session normalized dropout scores
    import visual_behavior_glm.GLM_across_session as gas

    glm_version = '24_events_all_L2_optimize_by_session'

    # get across session normalized dropout scores
    across_results, failed_cells = gas.load_cells(glm_version, clean_df=True)
    across_results = across_results.set_index('identifier')

    # only use across session values
    across = across_results[[key for key in across_results.keys() if '_across' in key] + ['cell_specimen_id', 'ophys_experiment_id', 'experience_level']]
    results_pivoted = across.copy()
    # rename across session columns
    results_pivoted = results_pivoted.rename(columns={'omissions_across': 'omissions', 'all-images_across': 'all-images',
                 'behavioral_across': 'behavioral', 'task_across': 'task'})

    # limit to matched cells
    results_pivoted = results_pivoted[results_pivoted.ophys_experiment_id.isin(matched_experiments)]
    results_pivoted = results_pivoted[results_pivoted.cell_specimen_id.isin(matched_cells)]
    # drop duplicates
    results_pivoted = results_pivoted.drop_duplicates(subset=['cell_specimen_id', 'experience_level'])
    print(len(results_pivoted.cell_specimen_id.unique()), 'cells in results_pivoted after limiting to strictly matched cells')

    # limit to features used for clustering
    features = processing.get_features_for_clustering()
    features = [*features, 'ophys_experiment_id']
    results_pivoted = processing.limit_results_pivoted_to_features_for_clustering(results_pivoted, features)
    # flip sign so coding scores are positive
    results_pivoted = processing.flip_sign_of_dropouts(results_pivoted, processing.get_features_for_clustering(),
                                                       use_signed_weights=False)
    # now drop ophys_experiment_id
    results_pivoted = results_pivoted.drop(columns=['ophys_experiment_id'])
    # update experience level names
    results_pivoted['experience_level'] = [utils.convert_experience_level(experience_level) for experience_level in
                                           results_pivoted.experience_level.values]

    ### cluster on all cre lines together ###

    feature_matrix = processing.get_feature_matrix_for_clustering(results_pivoted, glm_version, save_dir=save_dir)

    # get metadata for cells that will be clustered
    cell_metadata = processing.get_cell_metadata_for_feature_matrix(feature_matrix, cells_table)

    n_clusters = 14  # empirically determined, then validated based on within vs. across cluster variance

    cluster_meta_save_path = os.path.join(save_dir, 'cluster_meta_n_' + str(n_clusters) + '_clusters.h5')

    # if clustering output exists, load it
    if os.path.exists(cluster_meta_save_path):
        cluster_meta = pd.read_hdf(cluster_meta_save_path, key='df')
        if 0 in cluster_meta.cluster_id.unique():
            # add one to cluster ID so it starts at 1
            cluster_meta['cluster_id'] = cluster_meta['cluster_id'] + 1
        # merge in cell metadata
        cell_metadata = processing.get_cell_metadata_for_feature_matrix(feature_matrix, cells_table)
        cell_metadata = cell_metadata.drop(columns=['ophys_experiment_id', 'cre_line'])
        cluster_meta = cluster_meta.merge(cell_metadata.reset_index(), on='cell_specimen_id')
        cluster_meta = cluster_meta.set_index('cell_specimen_id')
    # otherwise run it and save it
    else:
        print('cluster_meta does not exist at', cluster_meta_save_path)
        print('plotting script requires pre-saved data to run')


    ## plot single cell examples from clusters ###

    # single cell coding score heatmaps
    for cluster_id in np.sort(cluster_meta.cluster_id.unique()):
        this_cluster = cluster_meta[cluster_meta.cluster_id==cluster_id]
        csids = this_cluster.index.unique()
        cell_roi_ids = cells_table[cells_table.cell_specimen_id.isin(csids)].index.values

        max_cluster_exp_var = all_results[all_results.cell_roi_id.isin(cell_roi_ids)].sort_values(by='adj_variance_explained_full', ascending=False)
        max_roi_ids = max_cluster_exp_var.cell_roi_id.values[:50]
        max_csids = cells_table.loc[max_roi_ids].cell_specimen_id.unique()

        for cell_specimen_id in max_csids:
            single_cell_plots.plot_single_cell_coding_heatmap(cluster_id, cell_specimen_id, feature_matrix, cells_table, save_dir=save_dir, folder=folder, ax=None)

    # single cell coding score barplots
    for cluster_id in np.sort(cluster_meta.cluster_id.unique()):
        this_cluster = cluster_meta[cluster_meta.cluster_id==cluster_id]
        csids = this_cluster.index.unique()
        cell_roi_ids = cells_table[cells_table.cell_specimen_id.isin(csids)].index.values

        max_cluster_exp_var = all_results[all_results.cell_roi_id.isin(cell_roi_ids)].sort_values(by='adj_variance_explained_full', ascending=False)
        max_roi_ids = max_cluster_exp_var.cell_roi_id.values[:50]
        max_csids = cells_table.loc[max_roi_ids].cell_specimen_id.unique()

        for cell_specimen_id in max_csids:
            single_cell_plots.plot_coding_scores_example_cell(cluster_id, cell_specimen_id, feature_matrix, cells_table, single_axis=True, save_dir=save_dir, folder=folder, ax=None)


    ### single cell mean response trace plots ###

    # load multi session dfs

    # set params
    data_type = 'events'
    interpolate = True
    output_sampling_rate = 30
    inclusion_criteria = 'platform_experiment_table'
    # params for stim response df creation
    event_type = 'all'
    # params for mean response df creation
    conditions = ['cell_specimen_id', 'is_change']
    # suffix
    change_suffix = '_' + utils.get_conditions_string(data_type, conditions) + '_' + inclusion_criteria

    # load image responses
    image_mdf = loading.get_multi_session_df_for_conditions(data_type, event_type, conditions,
                                                            inclusion_criteria,
                                                            interpolate=interpolate,
                                                            output_sampling_rate=output_sampling_rate,
                                                            epoch_duration_mins=None)

    change_mdf = image_mdf[image_mdf.is_change == True]
    image_mdf = image_mdf[image_mdf.is_change == False]

    # load omission responses

    event_type = 'all'
    # params for mean response df creation
    conditions = ['cell_specimen_id', 'omitted']
    # suffix
    omission_suffix = '_'+utils.get_conditions_string(data_type, conditions)+'_'+inclusion_criteria
    # omission responses
    omission_mdf = loading.get_multi_session_df_for_conditions(data_type, event_type, conditions, inclusion_criteria,
                                                            interpolate=interpolate, output_sampling_rate=output_sampling_rate,
                                                             epoch_duration_mins=None)
    omission_mdf = omission_mdf[omission_mdf.omitted==True]

    # merge cluster ID into multi session dfs
    omission_responses = omission_mdf.merge(cluster_meta.reset_index()[['cell_specimen_id', 'cluster_id']], on='cell_specimen_id')
    image_responses = image_mdf.merge(cluster_meta.reset_index()[['cell_specimen_id', 'cluster_id']], on='cell_specimen_id')
    change_responses = change_mdf.merge(cluster_meta.reset_index()[['cell_specimen_id', 'cluster_id']],  on='cell_specimen_id')

    # matched cell traces plot
    cluster_id = 10
    for cluster in np.sort(cluster_meta.cluster_id.unique()):
        this_cluster = cluster_meta[cluster_meta.cluster_id == cluster_id]
        csids = this_cluster.index.unique()
        cell_roi_ids = cells_table[cells_table.cell_specimen_id.isin(csids)].index.values

        max_cluster_exp_var = all_results[all_results.cell_roi_id.isin(cell_roi_ids)].sort_values(
            by='adj_variance_explained_full', ascending=True)
        max_roi_ids = max_cluster_exp_var.cell_roi_id.values[:50]
        max_csids = cells_table.loc[max_roi_ids].cell_specimen_id.unique()

        for cell_specimen_id in max_csids:
            plot_matched_cell_traces(cluster_id, cell_specimen_id, image_responses, omission_responses, change_responses,
                                     cells_table, save_dir=save_dir, folder=folder, ax=None)


    # matched cell ROIs
        for cluster_id in np.sort(cluster_meta.cluster_id.unique()):
            this_cluster = cluster_meta[cluster_meta.cluster_id==cluster_id]
            csids = this_cluster.index.unique()
            cell_roi_ids = cells_table[cells_table.cell_specimen_id.isin(csids)].index.values

            max_cluster_exp_var = results[results.cell_roi_id.isin(cell_roi_ids)].sort_values(by='adj_variance_explained_full', ascending=False)
            max_roi_ids = max_cluster_exp_var.cell_roi_id.values[:50]
            max_csids = cells_table.loc[max_roi_ids].cell_specimen_id.unique()
            for cell_specimen_id in max_csids:

                plot_matched_cell_rois(cluster_id, cell_specimen_id, cells_table, save_dir=save_dir, folder=folder, ax=None)



    ### heatmaps of single cell responses per cluster  ###
    
    import visual_behavior_analysis.visualization.ophys.platform_paper_plots as ppf

    # images
    if 'cluster_id' not in image_mdf:
           image_mdf = image_mdf.merge(cluster_meta.reset_index()[['cell_specimen_id', 'cluster_id']], on='cell_specimen_id')

    xlim_seconds = [-0.5, 0.7]
    event_type = 'all'
    timestamps = image_mdf.trace_timestamps.values[0]
    row_condition = 'experience_level'
    col_condition = 'cluster_id'
    cols_to_sort_by = ['mean_response']

    for cre_line in image_mdf.cre_line.unique():
        df = image_mdf[image_mdf.cre_line == cre_line]
        ppf.plot_response_heatmaps_for_conditions(df, timestamps, data_type, event_type,
                                                  row_condition, col_condition,
                                                  cols_to_sort_by=cols_to_sort_by, suptitle=cre_line,
                                                  microscope=None, vmax=None, xlim_seconds=xlim_seconds,
                                                  match_cells=False, cbar=False,
                                                  save_dir=os.path.join(save_dir, 'heatmaps'), folder=folder, suffix='', ax=None)

    # omissions
    if 'cluster_id' not in omission_mdf:
        omission_mdf = omission_mdf.merge(cluster_meta.reset_index()[['cell_specimen_id', 'cluster_id']], on='cell_specimen_id')

    xlim_seconds = [-1, 1.5]
    event_type = 'all'
    timestamps = omission_mdf.trace_timestamps.values[0]
    row_condition = 'experience_level'
    col_condition = 'cluster_id'
    cols_to_sort_by = ['mean_response']

    for cre_line in omission_mdf.cre_line.unique():
        df = omission_mdf[omission_mdf.cre_line == cre_line]
        ppf.plot_response_heatmaps_for_conditions(df, timestamps, data_type, event_type,
                                                  row_condition, col_condition,
                                                  cols_to_sort_by=cols_to_sort_by, suptitle=cre_line,
                                                  microscope=None, vmax=None, xlim_seconds=xlim_seconds,
                                                  match_cells=False, cbar=False,
                                                  save_dir=os.path.join(save_dir, 'heatmaps'), folder=folder, suffix='', ax=None)





    ### TBD ###


    ### image tuning curves for individual cells ###

    ### image tuning curve heatmap for all cells each cluster ###


