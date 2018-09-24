from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis
import visual_behavior.ophys.response_analysis.utilities as ut

import matplotlib
import logging

matplotlib.use('Agg')

logger = logging.getLogger(__name__)


def create_analysis_files(experiment_id, cache_dir, overwrite_analysis_files=True):
    logger.info(experiment_id)
    logger.info('saving ', str(experiment_id), 'to', cache_dir)
    dataset = VisualBehaviorOphysDataset(experiment_id, cache_dir)
    analysis = ResponseAnalysis(dataset, overwrite_analysis_files)

    logger.info('plotting experiment summary figure')
    from visual_behavior.ophys.plotting import experiment_summary_figures as esf
    esf.plot_experiment_summary_figure(analysis, save_dir=cache_dir)
    esf.plot_experiment_summary_figure(analysis, save_dir=dataset.analysis_dir)
    esf.plot_mean_first_flash_response_by_image_block(analysis, save_dir=cache_dir, ax=None)

    analysis.flash_response_df = ut.annotate_flash_response_df_with_block_set(analysis.flash_response_df)
    fdf = analysis.flash_response_df.copy()
    data = ut.add_early_late_block_ratio_for_fdf(fdf)
    esf.plot_mean_response_across_image_block_sets(data, analysis.dataset.analysis_folder, save_dir=cache_dir, ax=None)

    logger.info('plotting cell responses')
    from visual_behavior.ophys.plotting import summary_figures as sf
    for cell in dataset.get_cell_indices():
        sf.plot_image_response_for_trial_types(analysis, cell, save_dir=analysis.dataset.analysis_dir)
        sf.plot_image_response_for_trial_types(analysis, cell, save_dir=cache_dir)

        # sf.plot_mean_response_by_repeat(analysis, cell, save_dir=analysis.dataset.analysis_dir)
        # sf.plot_mean_response_by_image_block(analysis, cell, save_dir=analysis.dataset.analysis_dir)
        sf.plot_mean_response_by_repeat(analysis, cell, save_dir=cache_dir)
        sf.plot_mean_response_by_image_block(analysis, cell, save_dir=cache_dir)

    if dataset.events != None:
        sf.plot_event_detection(dataset.dff_traces, dataset.events, dataset.analysis_dir)
    logger.info('done')


if __name__ == '__main__':
    import sys

    experiment_id = sys.argv[1]
    cache_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_pilot_analysis'
    create_analysis_files(experiment_id, cache_dir, overwrite_analysis_files=True)

    # experiment_id = 673139359 #673139359
    # cache_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\visual_behavior_pilot_analysis'
    # create_analysis_files(experiment_id, cache_dir, overwrite_analysis_files=True)
