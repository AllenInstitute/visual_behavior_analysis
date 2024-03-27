#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')

from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis

from visual_behavior.visualization.ophys.experiment_summary_figures import plot_experiment_summary_figure


if __name__ == '__main__':


    # formatting
    import seaborn as sns
    sns.set_style('white')
    sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})
    sns.set_palette('deep')

    # experiment_id = 723037901
    # experiment_id = 712860764
    # cache_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\visual_behavior_pilot_analysis'
    # dataset = VisualBehaviorOphysDataset(experiment_id, cache_dir=cache_dir)
    # analysis = ResponseAnalysis(dataset)
    # plot_experiment_summary_figure(analysis, save_dir=cache_dir)
    #
    lims_ids = [644942849, 645035903, 645086795, 645362806, 646922970, 647108734,
                647551128, 647887770, 648647430, 649118720, 649318212, 652844352,
                653053906, 653123781, 639253368, 639438856, 639769395, 639932228,
                661423848, 663771245, 663773621, 665286182, 670396087, 671152642,
                672185644, 672584839, 685744008, 686726085, 695471168, 696136550,
                698244621, 698724265, 700914412, 701325132, 702134928, 702723649,
                692342909, 692841424, 693272975, 693862238, 712178916, 712860764,
                713525580, 714126693, 715161256, 715887497, 716327871, 716600289,
                715228642, 715887471, 716337289, 716602547, 720001924, 720793118,
                723064523, 723750115, 719321260, 719996589, 723748162, 723037901]

    cache_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\visual_behavior_pilot_analysis'

    for lims_id in lims_ids[10:]:
        print(lims_id)
        dataset = VisualBehaviorOphysDataset(lims_id, cache_dir=cache_dir)
        analysis = ResponseAnalysis(dataset)
        plot_experiment_summary_figure(analysis, save_dir=cache_dir)
        print('done plotting figures')
