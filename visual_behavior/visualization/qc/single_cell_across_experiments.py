import mindscope_utilities
import mindscope_utilities.visual_behavior_ophys as ophys
import visual_behavior_glm.GLM_analysis_tools as gat
import visual_behavior_glm.GLM_visualization_tools as gvt
import allensdk.brain_observatory.behavior.behavior_project_cache as bpc
import visual_behavior.plotting as vbp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from visual_behavior.data_access import from_lims
import visual_behavior.database as db
from visual_behavior.data_access import loading
import argparse

from tqdm import tqdm
import pandas as pd
import seaborn as sns
import numpy as np
import os
import glob

import warnings
warnings.filterwarnings("ignore")


def is_cell_in_experiment(cell_specimen_id, experiment):
    '''
    checks to determine if cell is in experiment
    '''
    return cell_specimen_id in experiment.cell_specimen_table.index


def get_text(cell_specimen_id, experiment, glm_version):
    '''
    gets text for plot title
    '''

    search_dict = {'glm_version': glm_version, 'ophys_experiment_id': experiment.ophys_experiment_id, 'cell_specimen_id': cell_specimen_id}
    glm_results = gat.retrieve_results(search_dict, results_type='full')
    if len(glm_results) > 0:
        glm_var_explained = glm_results.iloc[0]['Full__avg_cv_var_test']
    else:
        glm_var_explained = np.nan

    metadata = experiment.metadata

    text = 'ophys_experiment_id = {}, session_type = {}, date = {}, equipment = {}  | cell in experiment? {}\nFull model var explained = {:0.3f}'.format(
        experiment.ophys_experiment_id,
        metadata['session_type'],
        experiment.metadata['date_of_acquisition'].strftime('%m-%d-%Y'),
        metadata['equipment_name'],
        is_cell_in_experiment(cell_specimen_id, experiment),
        glm_var_explained
    )
    return text


def get_experiments(ophys_experiment_ids, disable_progress_bar=False):
    '''
    gets experiment objects for a list of experiment IDs
    '''

    cache = bpc.VisualBehaviorOphysProjectCache.from_lims()

    experiments = {}
    for ophys_experiment_id in tqdm(ophys_experiment_ids, desc='loading experiments', disable=disable_progress_bar):
        experiments[ophys_experiment_id] = cache.get_behavior_ophys_experiment(ophys_experiment_id)
    return experiments


def get_container_id_for_cell_id(cell_specimen_id):
    '''
    gets container ID associated with a given cell specimen ID
    '''

    query_string = '''
        select * from cell_rois
        where cell_specimen_id = {}
    '''

    single_oeid = db.lims_query(query_string.format(cell_specimen_id)).iloc[0]['ophys_experiment_id']
    all_ids = from_lims.get_all_ids_for_ophys_experiment_id(single_oeid)
    return all_ids.iloc[0]['ophys_container_id']


def get_all_experiments_ids_for_cell(cell_specimen_id):
    '''
    gets all associated experiment_ids for a given cell
    this will include experiments where the cell should have been found, but was not
    Steps:
        1) get one experiment ID associated with the cell ID
        2) get the container ID for that experiment ID
        3) get all experiment IDs for the given container
    '''
    cache = bpc.VisualBehaviorOphysProjectCache.from_lims()
    experiment_table = cache.get_ophys_experiment_table()

    container_id = get_container_id_for_cell_id(cell_specimen_id)
    ophys_experiment_ids = from_lims.get_ophys_experiment_ids_for_ophys_container_id(container_id)['ophys_experiment_id']
    ophys_experiment_ids = [ophys_experiment_id for ophys_experiment_id in ophys_experiment_ids if ophys_experiment_id in experiment_table.index and experiment_table.loc[ophys_experiment_id]['experiment_workflow_state'] == 'passed']

    return ophys_experiment_ids


def get_bounding_box(cell_specimen_id, buffer=60):
    '''
    defines a bounding box to draw around a given cell specimen ID
    uses cell position in all experiments
    attempts to draw a box that will surround the position of the cell in every experiment
    '''
    query_string = '''
        select * from cell_rois
        where cell_specimen_id = {}
    '''
    cell_df = db.lims_query(query_string.format(cell_specimen_id))

    left = cell_df['x'].min() - buffer
    width = (cell_df['x'].max() + buffer) - left
    top = cell_df['y'].min() - buffer
    height = (cell_df['y'].max() + buffer) - top

    return left, top, width, height


def add_bounding_box(left, top, width, height, ax):
    '''
    adds bounding box to max projection plot
    '''
    # Create a Rectangle patch
    rect = patches.Rectangle((left, top), width, height, linewidth=1, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)


def add_tidy_neural_data(experiment):
    '''
    adds an attribute to the experiment object that contains cell activity in tidy format
    '''
    if not hasattr(experiment, 'event_triggered_responses'):
        experiment.tidy_neural_data = ophys.build_tidy_cell_df(experiment, exclude_invalid_rois=False)


def add_event_triggered_averages(experiment, cell_specimen_id):
    '''
    adds an attribute to the experiment object that contains event triggered responses for all images + omissions
    '''
    stim_table = experiment.stimulus_presentations
    image_indices = np.sort(stim_table['image_index'].unique())

    if not hasattr(experiment, 'event_triggered_responses'):
        experiment.event_triggered_responses = pd.DataFrame({'cell_specimen_id': [], 'image_index': []})

    for image_index in image_indices:
        # avoid adding the etr for this cell/event twice
        if len(experiment.event_triggered_responses.query('cell_specimen_id == @cell_specimen_id and image_index == @image_index')) == 0:
            if experiment.stimulus_presentations.query('image_index == @image_index')['image_name'].iloc[0] == 'omitted':
                # get all omission times
                event_query = 'image_index == @image_index'
            else:
                # get only change times for other images
                event_query = 'image_index == @image_index and is_change'
            this_etr = mindscope_utilities.event_triggered_response(
                data=experiment.tidy_neural_data.query('cell_specimen_id == @cell_specimen_id'),
                t='timestamps',
                y='dff',
                event_times=stim_table.query(event_query)['start_time'],
                t_before=1.5,
                t_after=3,
            )
            this_etr['cell_specimen_id'] = cell_specimen_id
            this_etr['image_index'] = image_index

            experiment.event_triggered_responses = pd.concat((
                experiment.event_triggered_responses,
                this_etr
            ))


def show_dropout_summary(ophys_experiment_id, cell_specimen_id, glm_version, ax):
    '''
    adds dropout summary plot to the given axis
    '''
    search_dict = {
        'cell_specimen_id': cell_specimen_id,
        'ophys_experiment_id': ophys_experiment_id,
        'glm_version': glm_version,
    }
    this_cell_results_summary = gat.retrieve_results(search_dict, results_type='summary')

    dropouts_to_show = [
        'all-images',
        'omissions',
        'behavioral',
        'cognitive',
        'expectation',
        'licks',
        'pupil',
        'running',
        'pupil_and_omissions',
        'pupil_and_running',
        'hits'
    ]
    dropouts_to_show += ['single-' + dropout for dropout in dropouts_to_show]

    bp = gvt.plot_dropout_summary(
        results_summary=this_cell_results_summary.query('dropout in @dropouts_to_show'),
        cell_specimen_id=cell_specimen_id,
        ax=ax,
        dropouts_to_show=None,
        dropouts_to_plot='both',
        dropouts_to_exclude=[],
        ylabel_fontsize=8,
        ticklabel_fontsize=8,
        title_fontsize=8,
        legend_fontsize=6
    )

    return bp


def append_event_triggered_averages_to_experiments(experiments, cell_specimen_id, disable_progress_bar=False):
    '''
    adds tidy neural data and event triggered average attributes to each experiment
    experiments should be a dictionary with keys: experiment_ids and values: experiment objects
    '''

    cache = bpc.VisualBehaviorOphysProjectCache.from_lims()
    experiment_table = cache.get_ophys_experiment_table()

    ophys_experiment_ids = list(experiments.keys())
    for ophys_experiment_id in tqdm(ophys_experiment_ids, desc='adding tidy neural data and event triggered averages to each experiment', disable=disable_progress_bar):
        experiment = experiments[ophys_experiment_id]
        if cell_specimen_id in experiment.cell_specimen_table.index and ophys_experiment_id in experiment_table.index and experiment_table.loc[ophys_experiment_id]['experiment_workflow_state'] == 'passed':
            add_tidy_neural_data(experiment)
            add_event_triggered_averages(experiment, cell_specimen_id)


def assemble_plot(experiments, cell_specimen_id, glm_version, disable_progress_bar=False):
    row_buffer = 0.025

    ophys_experiment_ids = list(experiments.keys())

    n_rows = len(ophys_experiment_ids)

    cell_session_plot = plt.figure(figsize=(16, 4 * len(ophys_experiment_ids)))
    axes = {}
    for row, ophys_experiment_id in tqdm(enumerate(ophys_experiment_ids), total=len(ophys_experiment_ids), desc='populating plot axes for each experiment', disable=disable_progress_bar):

        experiment = experiments[ophys_experiment_id]

        row_start = row / n_rows
        row_end = (row + 1) / n_rows

        axes[ophys_experiment_id] = {
            'text': vbp.placeAxesOnGrid(cell_session_plot, xspan=[0, 1], yspan=[row_start, row_start + row_buffer]),
            'mask': vbp.placeAxesOnGrid(cell_session_plot, xspan=[0, 0.1], yspan=[row_start + row_buffer, row_end]),
            'zoomed_mask': vbp.placeAxesOnGrid(cell_session_plot, xspan=[0.125, 0.225], yspan=[row_start + row_buffer, row_end]),
            'visual_responses': vbp.placeAxesOnGrid(cell_session_plot, xspan=[0.275, 0.8], yspan=[row_start + 1.5 * row_buffer, row_end - row_buffer], dim=[1, 9], sharey=True),
            'model_dropout_summary': vbp.placeAxesOnGrid(cell_session_plot, xspan=[0.9, 1], yspan=[row_start + 1.5 * row_buffer, row_end - row_buffer],)
        }

        axes[ophys_experiment_id]['text'].text(0, 0, '{}'.format(get_text(cell_specimen_id, experiment, glm_version)), ha='left', va='bottom')
        axes[ophys_experiment_id]['text'].axis('off')

        axes[ophys_experiment_id]['mask'].imshow(experiment.max_projection, cmap='gray')
        axes[ophys_experiment_id]['mask'].axis('off')
        axes[ophys_experiment_id]['mask'].set_title('full max proj.', fontsize=8)

        left, top, width, height = get_bounding_box(cell_specimen_id)
        add_bounding_box(left, top, width, height, axes[ophys_experiment_id]['mask'])

        axes[ophys_experiment_id]['zoomed_mask'].imshow(experiment.max_projection, cmap='gray')
        axes[ophys_experiment_id]['zoomed_mask'].axis('off')
        axes[ophys_experiment_id]['zoomed_mask'].set_title('zoomed max proj.', fontsize=8)
        add_bounding_box(left, top, width, height, axes[ophys_experiment_id]['zoomed_mask'])

        axes[ophys_experiment_id]['zoomed_mask'].set_xlim(left - 10, left + width + 10)
        axes[ophys_experiment_id]['zoomed_mask'].set_ylim(top + height + 10, top - 10, )

        if is_cell_in_experiment(cell_specimen_id, experiment):
            mask = experiments[ophys_experiment_id].cell_specimen_table.loc[cell_specimen_id]['roi_mask'].astype(float)
            mask[np.where(mask == 0)] = np.nan
            if experiment.cell_specimen_table.loc[cell_specimen_id]['valid_roi']:
                cmap = 'Greens_r'
            else:
                cmap = 'Reds_r'
            axes[ophys_experiment_id]['zoomed_mask'].imshow(mask, cmap=cmap, alpha=0.75)

        if hasattr(experiment, 'event_triggered_responses') and cell_specimen_id in experiment.cell_specimen_table.index:
            stim_table = experiment.stimulus_presentations
            for col, image_index in enumerate(np.sort(stim_table['image_index'].unique())):
                image_name = stim_table.query('image_index == @image_index')['image_name'].iloc[0]
                sns.lineplot(
                    data=experiment.event_triggered_responses.query('cell_specimen_id == @cell_specimen_id and image_index == @image_index'),
                    x='time',
                    y='dff',
                    ax=axes[ophys_experiment_id]['visual_responses'][col]
                )
                vbp.designate_flashes(
                    axes[ophys_experiment_id]['visual_responses'][col],
                    omit=0 if image_name == 'omitted' else None,
                    pre_color='gray',
                    post_color='blue'
                )
                axes[ophys_experiment_id]['visual_responses'][col].set_title(image_name)
                axes[ophys_experiment_id]['visual_responses'][col].set_xlim(-1.5, 3)
        else:
            axes[ophys_experiment_id]['visual_responses'][0].text(0, 0, 'ROI is not in experiment', ha='left', va='bottom')
            for col in range(9):
                axes[ophys_experiment_id]['visual_responses'][col].axis('off')

        if is_cell_in_experiment(cell_specimen_id, experiment):
            if experiment.cell_specimen_table.loc[cell_specimen_id]['valid_roi']:
                show_dropout_summary(ophys_experiment_id, cell_specimen_id, glm_version, axes[ophys_experiment_id]['model_dropout_summary'])
            else:
                axes[ophys_experiment_id]['model_dropout_summary'].text(0, 0, 'ROI is invalid\nNo GLM results', ha='left', va='bottom')
                axes[ophys_experiment_id]['model_dropout_summary'].axis('off')
        else:
            axes[ophys_experiment_id]['model_dropout_summary'].axis('off')

    # get ylims extremes across all event triggered plots
    ylim_extrema = [np.inf, -np.inf]
    for ophys_experiment_id in ophys_experiment_ids:
        for ii in range(9):
            ylims = axes[ophys_experiment_id]['visual_responses'][ii].get_ylim()
            ylim_extrema = [func((ylim_extrema[i], ylims[i])) for i, func in zip(range(2), [np.min, np.max])]

    # apply ylims to all plots
    for ophys_experiment_id in ophys_experiment_ids:
        for ii in range(9):
            axes[ophys_experiment_id]['visual_responses'][ii].set_ylim(*ylim_extrema)

    cell_session_plot.suptitle('cell specimen ID = {}\ncontainer_id = {}\ngenotype = {}\nGLM Version = {}'.format(
        cell_specimen_id,
        get_container_id_for_cell_id(cell_specimen_id),
        experiment.metadata['cre_line'],
        glm_version
    ))
    cell_session_plot.tight_layout()

    return cell_session_plot, axes


def is_csid_in_folder(csid, folder):
    '''
    checks to see if cell_specimen_id exists in specified folder
    '''
    csids = []
    for fn in glob.glob(os.path.join(folder, 'csid*.png')):
        csids.append(int(fn.split('csid=')[1].split('_')[0]))
    return csid in csids


def roi_has_dff(cell_roi_id):
    '''
    checks to see if a given ROI has deltaF/F (or is NaN)
    '''
    dff = loading.get_dff_traces_for_roi(cell_roi_id)
    return np.all(pd.notnull(dff))


def make_single_cell_across_experiment_plot(cell_specimen_id, glm_version, disable_progress_bars=False, saveloc='', return_fig=True):
    '''
    performs all steps to build the plot for a single cell
    '''
    if not is_csid_in_folder(cell_specimen_id, saveloc):
        print('making plot for cell_specimen_id = {}'.format(cell_specimen_id))
        cell_specimen_id = int(cell_specimen_id)
        ophys_experiment_ids = get_all_experiments_ids_for_cell(cell_specimen_id)
        experiments = get_experiments(ophys_experiment_ids, disable_progress_bar=disable_progress_bars)

        append_event_triggered_averages_to_experiments(experiments, cell_specimen_id, disable_progress_bar=disable_progress_bars)

        fig, ax = assemble_plot(experiments, cell_specimen_id, glm_version, disable_progress_bar=disable_progress_bars)

        if saveloc != '':

            fn = 'csid={}_container={}_cre_line={}_glm_version={}.png'.format(
                cell_specimen_id,
                get_container_id_for_cell_id(cell_specimen_id),
                experiments[ophys_experiment_ids[0]].metadata['cre_line'],
                glm_version,
            )
            fig.savefig(os.path.join(saveloc, fn))

        if return_fig:
            return fig, ax

    else:
        print('plot for csid {} already exists in {}'.format(cell_specimen_id, saveloc))


def is_valid(cell_specimen_id, experiment):
    if is_cell_in_experiment(cell_specimen_id, experiment):
        return experiment.cell_specimen_table.loc[cell_specimen_id]['valid_roi']
    else:
        return False


def make_cell_matching_across_experiment_plot(cell_specimen_id, experiment_id_to_highlight=None, disable_progress_bar=False, saveloc='', return_fig=True):
    '''
    makes a set of plots showing only the fields of view, highlighting the cell ROI when found
    '''
    print('making plot for cell_specimen_id = {}, experiment_id = {}'.format(cell_specimen_id, experiment_id_to_highlight))
    cell_specimen_id = int(cell_specimen_id)
    ophys_experiment_ids = get_all_experiments_ids_for_cell(cell_specimen_id)
    experiments = get_experiments(ophys_experiment_ids, disable_progress_bar=disable_progress_bar)

    row_buffer = 0.025

    ophys_experiment_ids = list(experiments.keys())

    n_rows = len(ophys_experiment_ids)

    cell_session_plot = plt.figure(figsize=(5, 4 * len(ophys_experiment_ids)))
    axes = {}
    for row, ophys_experiment_id in tqdm(enumerate(ophys_experiment_ids), total=len(ophys_experiment_ids), desc='populating plot axes for each experiment', disable=disable_progress_bar):

        experiment = experiments[ophys_experiment_id]

        row_start = row / n_rows
        row_end = (row + 1) / n_rows

        axes[ophys_experiment_id] = {
            'text': vbp.placeAxesOnGrid(cell_session_plot, xspan=[0, 1], yspan=[row_start, row_start + row_buffer]),
            'mask': vbp.placeAxesOnGrid(cell_session_plot, xspan=[0, 0.45], yspan=[row_start + row_buffer, row_end]),
            'zoomed_mask': vbp.placeAxesOnGrid(cell_session_plot, xspan=[0.55, 1], yspan=[row_start + row_buffer, row_end]),
        }

        in_exp = is_cell_in_experiment(cell_specimen_id, experiment)
        text = 'experiment_id = {}, cell in experiment? {}, valid = {}'.format(
            ophys_experiment_id,
            in_exp,
            is_valid(cell_specimen_id, experiment)
        )
        axes[ophys_experiment_id]['text'].text(0, 0, text, ha='left', va='bottom')
        axes[ophys_experiment_id]['text'].axis('off')

        axes[ophys_experiment_id]['mask'].imshow(experiment.max_projection, cmap='gray')
        axes[ophys_experiment_id]['mask'].set_xticks([])
        axes[ophys_experiment_id]['mask'].set_yticks([])
        axes[ophys_experiment_id]['mask'].set_title('full max proj.', fontsize=8)

        left, top, width, height = get_bounding_box(cell_specimen_id)
        add_bounding_box(left, top, width, height, axes[ophys_experiment_id]['mask'])

        axes[ophys_experiment_id]['zoomed_mask'].imshow(experiment.max_projection, cmap='gray')
        axes[ophys_experiment_id]['zoomed_mask'].set_xticks([])
        axes[ophys_experiment_id]['zoomed_mask'].set_yticks([])
        axes[ophys_experiment_id]['zoomed_mask'].set_title('zoomed max proj.', fontsize=8)
        add_bounding_box(left, top, width, height, axes[ophys_experiment_id]['zoomed_mask'])

        axes[ophys_experiment_id]['zoomed_mask'].set_xlim(left - 10, left + width + 10)
        axes[ophys_experiment_id]['zoomed_mask'].set_ylim(top + height + 10, top - 10, )

        if ophys_experiment_id == experiment_id_to_highlight:
            for ax_label in ['mask', 'zoomed_mask']:
                for spine in axes[ophys_experiment_id][ax_label].spines.values():
                    spine.set_edgecolor('green')
                    spine.set_linewidth(5)

        if is_cell_in_experiment(cell_specimen_id, experiment):
            mask = experiments[ophys_experiment_id].cell_specimen_table.loc[cell_specimen_id]['roi_mask'].astype(float)
            mask[np.where(mask == 0)] = np.nan
            if experiment.cell_specimen_table.loc[cell_specimen_id]['valid_roi']:
                cmap = 'Greens_r'
            else:
                cmap = 'Reds_r'
            axes[ophys_experiment_id]['zoomed_mask'].imshow(mask, cmap=cmap, alpha=0.75)

    experiment_count = len(experiments)
    # print('about to calculate found_count')
    # print(len(experiments))
    # print(cell_specimen_id)
    # print([cell_specimen_id for experiment in experiments])
    # print([experiment for experiment in experiments])
    found_count = sum([is_cell_in_experiment(cell_specimen_id, experiments[oeid]) for oeid in ophys_experiment_ids])
    valid_count = sum([is_valid(cell_specimen_id, experiments[oeid]) for oeid in ophys_experiment_ids])
    cell_session_plot.suptitle('cre_line = {}\ncell_specimen_id = {}\nfound_count = {}/{}\nvalid_count = {}/{}'.format(
        experiment.metadata['cre_line'],
        cell_specimen_id,
        found_count,
        experiment_count,
        valid_count,
        found_count
    ))
    cell_session_plot.tight_layout()

    return cell_session_plot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='make a single cell plot')
    parser.add_argument('--csid', type=int, default=0, metavar='cell_specimen_id')
    parser.add_argument('--glm-version', type=str, default='', metavar='glm_version')
    parser.add_argument('--saveloc', type=str, default='', metavar='path in which to save file')
    parser.add_argument('--disable-progress-bars', dest='disable_progress_bars', action='store_true')
    args = parser.parse_args()

    print(args.csid, args.glm_version, args.suppress_progressbar)

    make_single_cell_across_experiment_plot(
        args.csid,
        args.glm_version,
        disable_progress_bars=args.disable_progress_bars,
        saveloc=args.saveloc
    )
