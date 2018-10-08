
from visual_behavior.ophys.io.convert_level_1_to_level_2 import convert_level_1_to_level_2


if __name__ == '__main__':
    import sys
    import matplotlib
    matplotlib.use('Agg')

    experiment_id = sys.argv[1]
    # experiment_id = 736490031
    # cache_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\visual_behavior_pilot_analysis'
    cache_dir = r'/allen/programs/braintv/workgroups/ophysdev/OPhysCore/Analysis/2018-08 - Behavior Integration test'
    ophys_data = convert_level_1_to_level_2(experiment_id, cache_dir)
