import mindscope_utilities
import mindscope_utilities.ophys
import visual_behavior_glm.GLM_analysis_tools as gat
import visual_behavior_glm.GLM_visualization_tools as gvt
import allensdk.brain_observatory.behavior.behavior_project_cache as bpc
import visual_behavior.plotting as vbp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from visual_behavior.data_access import from_lims
import visual_behavior.database as db
import argparse

from tqdm import tqdm
import pandas as pd
import seaborn as sns
import numpy as np

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
        metadata['date_of_acquisition'],
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

    query_string = '''
        select * from cell_rois
        where cell_specimen_id = {}
    '''
    single_oeid = db.lims_query(query_string.format(cell_specimen_id)).iloc[0]['ophys_experiment_id']
    all_ids = from_lims.get_all_ids_for_ophys_experiment_id(single_oeid)
    ophys_experiment_ids = from_lims.get_ophys_experiment_ids_for_ophys_container_id(all_ids.iloc[0]['ophys_container_id'])['ophys_experiment_id']
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
        experiment.tidy_neural_data = mindscope_utilities.ophys.build_tidy_cell_df(experiment, exclude_invalid_rois=False)


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
            this_etr = mindscope_utilities.event_triggered_response(
                data=experiment.tidy_neural_data.query('cell_specimen_id == @cell_specimen_id'),
                t='timestamps',
                y='dff',
                event_times=stim_table.query('image_index == @image_index')['start_time'],
                t_before=2,
                t_after=2,
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
                    pre_color='blue',
                    post_color='blue'
                )
                axes[ophys_experiment_id]['visual_responses'][col].set_title(image_name)
                axes[ophys_experiment_id]['visual_responses'][col].set_xlim(-2, 2)
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

    cell_session_plot.suptitle('cell specimen ID = {}\ngenotype = {}\nGLM Version = {}'.format(
        cell_specimen_id,
        experiment.metadata['cre_line'],
        glm_version
    ))
    cell_session_plot.tight_layout()

    return cell_session_plot, axes


def make_single_cell_across_experiment_plot(cell_specimen_id, glm_version, disable_progress_bars=False):
    '''
    performs all steps to build the plot for a single cell
    '''
    ophys_experiment_ids = get_all_experiments_ids_for_cell(cell_specimen_id)
    experiments = get_experiments(ophys_experiment_ids, disable_progress_bar=disable_progress_bars)

    append_event_triggered_averages_to_experiments(experiments, cell_specimen_id, disable_progress_bar=disable_progress_bars)

    fig, ax = assemble_plot(experiments, cell_specimen_id, glm_version, disable_progress_bar=disable_progress_bars)

    return fig, ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='make a single cell plot')
    parser.add_argument('--csid', type=int, default=0, metavar='cell_specimen_id')
    parser.add_argument('--glm-version', type=str, default='', metavar='glm_version')
    parser.add_argument('--save-loc', type=str, default='', metavar='path in which to save file')
    parser.add_argument('--disable-progress-bars', dest='suppress_progressbar', action='store_true')
    args = parser.parse_args()

    print(args.csid, args.glm_version, args.suppress_progressbar)

    make_single_cell_across_experiment_plot(
        args.csid,
        args.glm_version,
        disable_progress_bars=args.suppress_progressbar,
        saveloc=args.save_loc
    )
