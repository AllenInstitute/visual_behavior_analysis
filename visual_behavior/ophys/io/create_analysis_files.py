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
    use_events = False
    print(experiment_id)
    dataset = VisualBehaviorOphysDataset(experiment_id, cache_dir=cache_dir)
    analysis = ResponseAnalysis(dataset, use_events=use_events, overwrite_analysis_files=overwrite_analysis_files)
    # tmp = analysis.trials_run_speed_df
    tmp = analysis.trials_response_df
    tmp = analysis.omission_response_df
    tmp = analysis.stimulus_response_df

    # example_cells = ut.get_active_cell_indices(dataset.dff_traces_array)
    # sf.plot_example_traces_and_behavior(dataset, example_cells, xmin_seconds=600, length_mins=1.5, dff_max=4,
    #                                     include_running=False, cell_label=False, use_events=use_events,
    #                                     save_figures=True, save_dir=dataset.analysis_dir,
    #                                     folder='experiment_summary_figures')
    # sf.plot_average_flash_response_example_cells(analysis, example_cells, include_changes=False,
    #                                              save_figures=True, save_dir=dataset.analysis_dir,
    #                                              folder='experiment_summary_figures')
    sf.plot_max_projection_image(dataset, save_dir=dataset.analysis_dir, folder='experiment_summary_figures')
    esf.plot_experiment_summary_figure(analysis, save_dir=dataset.cache_dir)
    esf.plot_experiment_summary_figure(analysis, save_dir=dataset.analysis_dir)



if __name__ == '__main__':

    # pilot study manuscript final expts
    experiment_ids = [644942849, 645035903, 645086795, 645362806, 646922970, 647108734,
                       647551128, 647887770, 639253368, 639438856, 639769395, 639932228,
                       661423848, 663771245, 663773621, 665286182, 670396087, 671152642,
                       672185644, 672584839, 695471168, 696136550, 698244621, 698724265,
                       700914412, 701325132, 702134928, 702723649, 692342909, 692841424,
                       693272975, 693862238, 712178916, 712860764, 713525580, 714126693,
                       715161256, 715887497, 716327871, 716600289, 729951441, 730863840,
                       736490031, 737471012, 715228642, 715887471, 716337289, 716602547,
                       720001924, 720793118, 723064523, 723750115, 719321260, 719996589,
                       723748162, 723037901, 731936595, 732911072, 733691636, 736927574,
                       745353761, 745637183, 747248249, 750469573, 754566180, 754943841,
                       756715598, 758274779, 751935154, 752966796, 753931104, 754552635,
                       766779984, 767424894, 768223868, 768865460, 771381093, 772696884,
                       773816712, 774370025, 771427955, 772131949, 772735942, 773843260,
                       768224465, 768871217, 769514560, 770094844, 760696146, 760986090,
                       762214438, 768225217, 774379465, 775011398, 775429615, 776042634,
                       648647430, 649118720, 649318212, 673139359, 673460976]

    import os
    cache_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_pilot_analysis/visual_behavior_pilot_manuscript_FINAL'
    for experiment_id in experiment_ids:
        create_analysis_files(experiment_id, cache_dir, overwrite_analysis_files=False)
