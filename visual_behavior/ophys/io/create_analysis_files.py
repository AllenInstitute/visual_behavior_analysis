"""
@author: marinag
"""

# import numpy as np
from visual_behavior.ophys.io.convert_level_1_to_level_2 import convert_level_1_to_level_2
# from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis
import visual_behavior.data_access.loading as loading
#
# from visual_behavior.visualization.ophys import experiment_summary_figures as esf
# from visual_behavior.visualization.ophys import summary_figures as sf


def create_analysis_files(experiment_id, cache_dir, overwrite_analysis_files=True):
    use_events = False
    print(experiment_id)
    dataset = loading.get_ophys_dataset(experiment_id)
    analysis_dir = dataset.analysis_dir
    if len(analysis_dir) == 0:
        try:
            _ = convert_level_1_to_level_2(experiment_id, cache_dir, plot_roi_validation=False)
        except:  # NOQA E722
            print('could not convert', experiment_id)
    analysis = ResponseAnalysis(dataset, use_events=use_events, overwrite_analysis_files=overwrite_analysis_files)

    _ = analysis.trials_response_df
    _ = analysis.omission_response_df
    _ = analysis.stimulus_response_df
    # tmp = analysis.omission_run_speed_df
    # tmp = analysis.trials_run_speed_df
    # tmp = analysis.stimulus_run_speed_df

    # example_cells = ut.get_active_cell_indices(dataset.dff_traces_array)
    # sf.plot_example_traces_and_behavior(dataset, example_cells, xmin_seconds=600, length_mins=1.5, dff_max=4,
    #                                     include_running=False, cell_label=False, use_events=use_events,
    #                                     save_figures=True, save_dir=dataset.analysis_dir,
    #                                     folder='experiment_summary_figures')
    # sf.plot_average_flash_response_example_cells(analysis, example_cells, include_changes=False,
    #                                              save_figures=True, save_dir=dataset.analysis_dir,
    #                                              folder='experiment_summary_figures')
    # sf.plot_max_projection_image(dataset, save_dir=dataset.analysis_dir, folder='experiment_summary_figures')
    # esf.plot_experiment_summary_figure(analysis, save_dir=dataset.cache_dir)
    # esf.plot_experiment_summary_figure(analysis, save_dir=dataset.analysis_dir)
    #
    #
    #
    # print('plotting experiment summary figure')
    # esf.plot_experiment_summary_figure(analysis, save_dir=cache_dir)
    # esf.plot_experiment_summary_figure(analysis, save_dir=dataset.analysis_dir)
    # esf.plot_roi_masks(dataset, save=True)
    # esf.plot_average_flash_response_example_cells(analysis, save_figures=True, save_dir=cache_dir, folder='mean_flash_response_average')
    # #
    # if not(turn_off_plotting):
    # print('plotting cell responses')
    # for cell in dataset.get_cell_indices():
    #     # sf.plot_image_response_for_trial_types(analysis, cell, save=True)
    #     sf.plot_cell_summary_figure(analysis, cell, save=True, show=False, cache_dir=cache_dir)
    #
    # print('plotting example traces')
    # active_cell_indices = ut.get_active_cell_indices(dataset.dff_traces)
    # length_mins = 1
    # for xmin_seconds in np.arange(0, 5000, length_mins * 60):
    #     sf.plot_example_traces_and_behavior(dataset, active_cell_indices, xmin_seconds, length_mins, save=True,
    #                                         cell_label=False, include_running=True, use_events=use_events)
    # if (use_events == True) and (dataset.events is not None):
    #     analysis = ResponseAnalysis(dataset, overwrite_analysis_files, use_events=use_events)
    #     # pairwise_correlations_df = analysis.get_pairwise_correlations_df()
    #     #
    #     print('plotting experiment summary figure')
    #     esf.plot_experiment_summary_figure(analysis, save_dir=cache_dir)
    #     esf.plot_experiment_summary_figure(analysis, save_dir=dataset.analysis_dir)
    #     #
    #     # print('plotting example traces')
    #     # for xmin_seconds in np.arange(0, 3000, length_mins * 60):
    #     #     sf.plot_example_traces_and_behavior(dataset, active_cell_indices, xmin_seconds, length_mins, save=True,
    #     #                                         cell_label=False, include_running=True, use_events=use_events)
    #     #
    #     # print('plotting cell responses')
    #     for cell in dataset.get_cell_indices():
    #         # sf.plot_image_response_for_trial_types(analysis, cell, save=True)
    #         sf.plot_cell_summary_figure(analysis, cell, save=True, show=False, cache_dir=cache_dir)
    # else:
    #     print('no events for', experiment_id)


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

    # import os
    cache_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\visual_behavior_production_analysis'
    for experiment_id in experiment_ids:
        create_analysis_files(experiment_id, cache_dir, overwrite_analysis_files=False)
