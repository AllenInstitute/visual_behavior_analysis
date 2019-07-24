#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')

from visual_behavior.ophys.io.convert_level_1_to_level_2 import convert_level_1_to_level_2


if __name__ == '__main__':
    import sys
    experiment_id = sys.argv[1]
    cache_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_production_analysis'
    ophys_data = convert_level_1_to_level_2(experiment_id, cache_dir)

    # #multiscope viral pilot
    # experiment_ids = [895448196, 895448198, 895448200, 895448202, 895448204, 895448206, 895448208, 895448210,
    #                   895450112, 895450116, 895450120, 895450122, 895450124, 895450126, 895450128, 895450130,
    #                   894750259, 894750261, 894750263, 894750265, 894750267, 894750269, 894750272, 894750274,
    #                   881955118, 881955120, 881955122, 881955124, 881955128, 881955132, 881955134, 881955136]
    #
    # cache_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\visual_behavior_production_analysis'
    # for experiment_id in experiment_ids:
    #     try:
    #         ophys_data = convert_level_1_to_level_2(experiment_id, cache_dir)
    #     except:
    #         print('problem with', experiment_id)
