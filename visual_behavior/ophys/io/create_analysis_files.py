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
    esf.plot_experiment_summary_figure(analysis, save_dir=cache_dir)
    esf.plot_experiment_summary_figure(analysis, save_dir=dataset.analysis_dir)
    esf.plot_roi_masks(dataset, save=True)
    # esf.plot_average_flash_response_example_cells(analysis, save_figures=True, save_dir=cache_dir, folder='mean_flash_response_average')
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
    # #VisualBehaviorDevelopment - complete dataset as of 11/15/18
    experiment_ids = [639253368, 639438856, 639769395, 639932228, 644942849, 645035903,
                      645086795, 645362806, 646922970, 647108734, 647551128, 647887770,
                      648647430, 649118720, 649318212, 661423848, 663771245, 663773621,
                      664886336, 665285900, 665286182, 670396087, 671152642, 672185644,
                      672584839, 673139359, 673460976, 685744008, 686726085, 692342909,
                      692841424, 693272975, 693862238, 695471168, 696136550, 698244621,
                      698724265, 700914412, 701325132, 702134928, 702723649, 712178916,
                      712860764, 713525580, 714126693, 715161256, 715228642, 715887471,
                      715887497, 716327871, 716337289, 716600289, 716602547, 719321260,
                      719996589, 720001924, 720793118, 723037901, 723064523, 723748162,
                      723750115, 729951441, 730863840, 731936595, 732911072, 733691636,
                      736490031, 736927574, 737471012, 745353761, 745637183, 747248249,
                      750469573, 751935154, 752966796, 753931104, 754552635, 754566180,
                      754943841, 756715598, 758274779, 760003838, 760400119, 760696146,
                      760986090, 761861597, 762214438, 762214650, 766779984, 767424894,
                      768223868, 768224465, 768225217, 768865460, 768871217, 769514560,
                      770094844, 771381093, 771427955, 772131949, 772696884, 772735942,
                      773816712, 773843260, 774370025, 774379465, 775011398, 775429615,
                      776042634]
    import os
    cache_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\visual_behavior_pilot_analysis'
    for experiment_id in experiment_ids:
        folder = [folder for folder in os.listdir(cache_dir) if str(experiment_id) in folder]
        if len(folder) > 0:
            file = [file for file in os.listdir(os.path.join(cache_dir, folder[0])) if 'flash_response_df' in file]
            if len(file) != 0:
                # print('no flash events for',experiment_id,'- generating')
                create_analysis_files(experiment_id, cache_dir, overwrite_analysis_files=False)
