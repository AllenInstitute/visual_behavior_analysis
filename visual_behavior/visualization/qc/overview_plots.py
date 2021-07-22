import numpy as np
import matplotlib.pyplot as plt

from visual_behavior.visualization import utils as ut
from visual_behavior.data_access import loading as data_loading
# from visual_behavior.visualization.qc import session_plots as sp
# from visual_behavior.visualization.qc import plotting_utils as pu
# from visual_behavior.data import processing as data_processing
# from visual_behavior.visualization.qc import experiment_plots as ep


def plot_expts_for_container(experiments_table, ophys_container_id, experiment_ids_to_highlight=None, max_n_expts=None,
                             ax=None):
    expts = experiments_table[experiments_table.ophys_container_id == ophys_container_id]
    if experiment_ids_to_highlight is None:
        experiment_ids_to_highlight = expts.ophys_experiment_id.unique()
    order = np.argsort(expts.date_of_acquisition.values)
    experiment_ids = expts.ophys_experiment_id.values[order]
    session_type_color_map = ut.get_session_type_color_map()

    if max_n_expts is None:
        n_expts = len(expts)
    else:
        n_expts = max_n_expts
    img = np.empty((1, n_expts, 3))
    img[:] = 256
    fail_x = []
    for i, expt_id in enumerate(experiment_ids):
        this_expt = expts[expts.ophys_experiment_id == expt_id]
        if expt_id in experiment_ids_to_highlight:
            img[0, i, :] = session_type_color_map[this_expt['session_type'].values[0]]
        else:
            color = session_type_color_map[this_expt['session_type'].values[0]]
            color = ut.make_color_transparent(color, alpha=0.05)
            img[0, i, :] = color
        if this_expt['experiment_workflow_state'].values[0] == 'failed':
            fail_x.append(i)

    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 1))

    # label with container info
    project_code = expts.project_code.values[0]
    super_ophys_container_id = expts.super_ophys_container_id.values[0]
    ax.text(x=-2.5, y=0, s=str(super_ophys_container_id) + '_' + str(ophys_container_id), ha='center', va='bottom', fontsize=14)
    location = expts.location.values[0]
    color = ut.get_location_color(location, project_code)
    ax.text(x=-2.5, y=0, s=location, ha='center', va='top', fontsize=14, color=color)

    # plot session colors and fails
    ax.imshow(img.astype(int))
    ax.axis('off')
    for x in fail_x:
        ax.text(x=x, y=0, s='X', ha='center', va='center', fontsize=20)

    return ax


def plot_container_overview(experiments_table, project_code):
    project_expts = experiments_table[experiments_table.project_code == project_code]
    ophys_container_ids = np.sort(project_expts.ophys_container_id.unique())
    max_n_expts = project_expts.groupby('ophys_container_id').count().ophys_experiment_id.max()

    figsize = (15, len(ophys_container_ids))
    fig, ax = plt.subplots(len(ophys_container_ids), 1, figsize=figsize, gridspec_kw={'wspace': 0, 'hspace': 0})
    for i, ophys_container_id in enumerate(ophys_container_ids):
        ax[i] = plot_expts_for_container(ophys_container_id, project_expts, max_n_expts=max_n_expts, ax=ax[i])
    plt.suptitle('project code: ' + project_code, x=0.5, y=0.89, fontsize=20, horizontalalignment='center')

    save_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots'
    ut.save_figure(fig, figsize, save_dir, 'overview_plots', project_code + '_containers_chronological')


def plot_chronological_datacube_summary(project_experiments_table, experiment_ids_to_highlight, what_is_highlighted_string, save_dir=None):
    expts = project_experiments_table.copy()
    project_code = expts.project_code.unique()[0]
    max_n_expts = expts.groupby(['super_ophys_container_id', 'ophys_container_id']).count().ophys_session_id.max()
    super_ophys_container_ids = np.sort(expts.super_ophys_container_id.unique())
    n_ophys_container_ids = len(expts.ophys_container_id.unique())

    figsize = (15, n_ophys_container_ids)
    fig, ax = plt.subplots(n_ophys_container_ids, 1, figsize=figsize, gridspec_kw={'wspace': 0, 'hspace': 0})
    i = 0
    for x, super_ophys_container_id in enumerate(super_ophys_container_ids):
        super_container_expts = expts[expts.super_ophys_container_id == super_ophys_container_id]
        ophys_container_ids = super_container_expts.ophys_container_id.unique()
        for y, ophys_container_id in enumerate(ophys_container_ids):
            ax[i] = plot_expts_for_container(super_container_expts, ophys_container_id, experiment_ids_to_highlight, max_n_expts=max_n_expts, ax=ax[i])
            i += 1
    plt.suptitle('project code: ' + project_code + ' - ' + what_is_highlighted_string, x=0.3, y=0.9, fontsize=20, horizontalalignment='center')
    save_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\qc_plots'
    if save_dir:
        ut.save_figure(fig, figsize, save_dir, 'overview_plots', project_code + '_containers_chronological_' + what_is_highlighted_string)


def plot_sorted_datacube_summary(project_experiments_table, experiment_ids_to_highlight, what_is_highlighted_string, save_dir=None):
    expts = project_experiments_table.copy()
    project_code = expts.project_code.unique()[0]
    max_n_expts = expts.groupby(['super_ophys_container_id', 'ophys_container_id']).count().ophys_session_id.max()
    sorted_super_ophys_container_ids = expts.groupby(['cre_line', 'targeted_structure', 'date_of_acquisition', 'super_ophys_container_id']).count().reset_index().super_ophys_container_id.unique()
    super_ophys_container_ids = sorted_super_ophys_container_ids
    n_ophys_container_ids = len(expts.ophys_container_id.unique())

    figsize = (15, n_ophys_container_ids)
    fig, ax = plt.subplots(n_ophys_container_ids, 1, figsize=figsize, gridspec_kw={'wspace': 0, 'hspace': 0})
    i = 0
    for x, super_ophys_container_id in enumerate(super_ophys_container_ids):
        super_container_expts = expts[expts.super_ophys_container_id == super_ophys_container_id]
        ophys_container_ids = super_container_expts.ophys_container_id.unique()
        for y, ophys_container_id in enumerate(ophys_container_ids):
            ax[i] = plot_expts_for_container(super_container_expts, ophys_container_id, experiment_ids_to_highlight, max_n_expts=max_n_expts, ax=ax[i])
            i += 1
    plt.suptitle('project code: ' + project_code + ' - ' + what_is_highlighted_string, x=0.3, y=0.9, fontsize=20, horizontalalignment='center')
    fig.subplots_adjust(left=0.2)
    save_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\qc_plots'
    if save_dir:
        ut.save_figure(fig, figsize, save_dir, 'overview_plots', project_code + '_containers_sorted_' + what_is_highlighted_string)


if __name__ == "__main__":

    import visual_behavior.visualization.qc.data_filtering as df

    project_code = 'VisualBehaviorTask1B'
    # project_code = 'VisualBehaviorMultiscope4areasx2d'
    # project_code = 'VisualBehaviorMultiscope'

    experiments = data_loading.get_filtered_ophys_experiment_table(include_failed_data=True)
    project_expts = experiments[experiments.project_code == project_code]

    list_of_filters = ['passing_expts',
                       'first_novel_image_exposure_passing',
                       'first_novel_image_exposure_including_failures',
                       'first_omission_exposure_passing',
                       'first_omission_exposure_including_failures']

    for what_is_highlighted_string in list_of_filters:

        if what_is_highlighted_string == 'passing_expts':
            experiment_ids_to_highlight = df.get_experiment_ids_that_pass_qc(project_expts)
        if what_is_highlighted_string == 'first_novel_image_exposure_passing':
            experiment_ids_to_highlight = df.get_first_passing_novel_image_exposure_experiment_ids(project_expts)
        if what_is_highlighted_string == 'first_novel_image_exposure_including_failures':
            experiment_ids_to_highlight = df.get_first_novel_image_exposure_experiment_ids(project_expts)
        if what_is_highlighted_string == 'first_omission_exposure_passing':
            experiment_ids_to_highlight = df.get_first_passing_omission_exposure_experiment_ids(project_expts)
        if what_is_highlighted_string == 'first_omission_exposure_including_failures':
            experiment_ids_to_highlight = df.get_first_omission_exposure_experiment_ids(project_expts)

        plot_sorted_datacube_summary(project_expts, experiment_ids_to_highlight, what_is_highlighted_string)
