"""
@author: marinag
"""

# import numpy as np

from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis
# import visual_behavior.ophys.response_analysis.utilities as ut

from visual_behavior.visualization.ophys import experiment_summary_figures as esf
from visual_behavior.visualization.ophys import summary_figures as sf


# import logging

# logger = logging.getLogger(__name__)


def create_analysis_files(experiment_id, cache_dir, overwrite_analysis_files=True, turn_off_plotting=False):

    from visual_behavior.ophys.io.convert_level_1_to_level_2 import convert_level_1_to_level_2
    ophys_data = convert_level_1_to_level_2(experiment_id, cache_dir, plot_roi_validation=False);

    print(experiment_id)
    print('saving ' + str(experiment_id) + ' to ' + cache_dir)
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

    print('plotting experiment summary figure')
    esf.plot_experiment_summary_figure(analysis, save_dir=cache_dir)
    esf.plot_experiment_summary_figure(analysis, save_dir=dataset.analysis_dir)
    esf.plot_roi_masks(dataset, save=True)
    # esf.plot_average_flash_response_example_cells(analysis, save_figures=True, save_dir=cache_dir, folder='mean_flash_response_average')
    # #
    # if not(turn_off_plotting):
    #     print('plotting cell responses')
    #     for cell in dataset.get_cell_indices():
    #         # sf.plot_image_response_for_trial_types(analysis, cell, save=True)
    #         sf.plot_cell_summary_figure(analysis, cell, save=True, show=False, cache_dir=cache_dir)
    #
    #     print('plotting example traces')
    #     active_cell_indices = ut.get_active_cell_indices(dataset.dff_traces)
    #     length_mins = 1
    #     for xmin_seconds in np.arange(0, 5000, length_mins * 60):
    #         sf.plot_example_traces_and_behavior(dataset, active_cell_indices, xmin_seconds, length_mins, save=True,
    #                                             cell_label=False, include_running=True, use_events=use_events)
    if dataset.events is not None:
        use_events = True
        analysis = ResponseAnalysis(dataset, overwrite_analysis_files, use_events=use_events)
        # pairwise_correlations_df = analysis.get_pairwise_correlations_df()
        #
        print('plotting experiment summary figure')
        esf.plot_experiment_summary_figure(analysis, save_dir=cache_dir)
        esf.plot_experiment_summary_figure(analysis, save_dir=dataset.analysis_dir)
        #
        # print('plotting example traces')
        # for xmin_seconds in np.arange(0, 3000, length_mins * 60):
        #     sf.plot_example_traces_and_behavior(dataset, active_cell_indices, xmin_seconds, length_mins, save=True,
        #                                         cell_label=False, include_running=True, use_events=use_events)
        #
        # print('plotting cell responses')
        for cell in dataset.get_cell_indices():
            # sf.plot_image_response_for_trial_types(analysis, cell, save=True)
            sf.plot_cell_summary_figure(analysis, cell, save=True, show=False, cache_dir=cache_dir)
    else:
        print('no events for', experiment_id)


if __name__ == '__main__':
    # VisualBehavior production as of 3/20/19
    experiment_ids = [775614751, 778644591, 782675436, 783927872, 783928214, 784482326,
                      787461073, 787498309, 787501821, 788488596, 788489531, 788490510,
                      789359614, 790149413, 790709081, 791119849, 791453282,
                      791980891, 792812544, 792813858, 792815735, 792816531, 794378505,
                      794381992, 795073741, 795075034, 795076128, 795948257, 795952471,
                      795952488, 795953296, 796105304, 796105823, 796106321, 796106850,
                      796108483, 796306417, 796308505, 797255551, 798392580, 798403387,
                      798404219, 799366517, 799368262, 799368904, 803736273, 805100431,
                      805784313, 805784331, 806455766, 806456687, 806989729, 807752719,
                      807753318, 807753334, 807753920, 808619543, 808621015, 808621034,
                      808621958, 809497730, 809501118, 811456530, 811458048, 813083478,
                      814610580, 815097949, 815652334, 817267785, 817267860, 818073631,
                      819432482, 819434449, 820307518, 822024770,
                      822028017, 822028587, 822641265, 822647116, 822647135, 822656725,
                      823392290, 823396897, 823401226, 824333777, 825120601, 825130141,
                      825623170, 826583436, 826585773, 826587940, 830093338, 830700781,
                      830700800, 831330404, 832117336, 833629926, 833629942, 833631914,
                      834275020, 834275038, 834279496, 836258936, 836258957, 836260147,
                      836910438, 836911939, 837296345, 837729902, 838849930]

    # mouse 458155
    experiment_ids = [893832486,  # 1_images_G
                    894726047,  # 1_images_E
                    895421150,  # 1_images_G, fail z-drift
                    896164962,  # 3_images_G, fail low d prime
                    897385282] # 3_images_G
    # # # mouse 451790
    # experiment_id = [897385227,  #1_images_E
    #                 897766332,  #1_images_G
    #                 898747809,  #3_images_E, fail low d prime
    #                 899085549, #3_images_E, failed dropped frames
    #                 90055097]  #1_3images_E
    # import os
    cache_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\visual_behavior_production_analysis'
    for experiment_id in experiment_ids:
        create_analysis_files(experiment_id, cache_dir, overwrite_analysis_files=False)
