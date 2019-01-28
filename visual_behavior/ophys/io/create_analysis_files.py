"""
@author: marinag
"""

import numpy as np

from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis
import visual_behavior.ophys.response_analysis.utilities as ut

from visual_behavior.visualization.ophys import experiment_summary_figures as esf
from visual_behavior.visualization.ophys import summary_figures as sf


# import logging

# logger = logging.getLogger(__name__)


def create_analysis_files(experiment_id, cache_dir, overwrite_analysis_files=True):
    # logger.info(experiment_id)
    print(experiment_id)
    print('saving ', str(experiment_id), 'to', cache_dir)
    dataset = VisualBehaviorOphysDataset(experiment_id, cache_dir)

    use_events = False

    # print('plotting example traces')
    # active_cell_indices = ut.get_active_cell_indices(dataset.dff_traces)
    # length_mins = 1
    # for xmin_seconds in np.arange(0, 5000, length_mins * 60):
    #     sf.plot_example_traces_and_behavior(dataset, active_cell_indices, xmin_seconds, length_mins, save=True,
    #                                         cell_label=False, include_running=True, use_events=use_events)

    analysis = ResponseAnalysis(dataset, overwrite_analysis_files, use_events=use_events)
    # pairwise_correlations_df = analysis.get_pairwise_correlations_df()  # flake8: noqa: F841

    # print('plotting experiment summary figure')
    # esf.plot_experiment_summary_figure(analysis, save_dir=cache_dir)
    # esf.plot_experiment_summary_figure(analysis, save_dir=dataset.analysis_dir)
    # esf.plot_roi_masks(dataset, save=True)
    #
    # print('plotting cell responses')
    # for cell in dataset.get_cell_indices():
    #     sf.plot_image_response_for_trial_types(analysis, cell, save=True)
    #     sf.plot_cell_summary_figure(analysis, cell, save=True, show=False, cache_dir=cache_dir)

    if dataset.events is not None:
        use_events = True
        analysis = ResponseAnalysis(dataset, overwrite_analysis_files, use_events=use_events)
        # pairwise_correlations_df = analysis.get_pairwise_correlations_df()
        #
        # print('plotting experiment summary figure')
        # esf.plot_experiment_summary_figure(analysis, save_dir=cache_dir)
        # esf.plot_experiment_summary_figure(analysis, save_dir=dataset.analysis_dir)
        #
        # print('plotting example traces')
        # for xmin_seconds in np.arange(0, 3000, length_mins * 60):
        #     sf.plot_example_traces_and_behavior(dataset, active_cell_indices, xmin_seconds, length_mins, save=True,
        #                                         cell_label=False, include_running=True, use_events=use_events)
        #
        # print('plotting cell responses')
        # for cell in dataset.get_cell_indices():
        #     sf.plot_image_response_for_trial_types(analysis, cell, save=True)
        #     sf.plot_cell_summary_figure(analysis, cell, save=True, show=False, cache_dir=cache_dir)
    else:
        print('no events for', experiment_id)


if __name__ == '__main__':
    experiment_ids = [775614751, 778644591, 787461073, 782675436, 783928214, 783927872,
                      787501821, 787498309, 788490510, 788488596, 788489531, 789359614,
                      790149413, 790709081, 791119849, 791453282, 791980891, 792813858,
                      792812544, 792816531, 792815735, 794381992, 794378505, 795076128,
                      795073741, 795952471, 795952488, 795953296, 795948257, 796106850,
                      796106321, 796108483, 796105823, 796308505, 797255551, 795075034,
                      798403387, 798404219, 799366517, 799368904, 799368262, 803736273,
                      805100431, 805784331, 805784313, 806456687, 806455766, 806989729,
                      807753318, 807752719, 807753334, 807753920, 796105304, 784482326,
                      779335436, 782675457, 791974731, 791979236,
                      800034837, 802649986, 806990245, 808621958,
                      808619526, 808619543, 808621034, 808621015]

    cache_dir = r'\\allen/programs\braintv\workgroups\nc-ophys\visual_behavior\visual_behavior_production_analysis'
    for experiment_id in experiment_ids:
        create_analysis_files(experiment_id, cache_dir, overwrite_analysis_files=True)
