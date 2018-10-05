
from visual_behavior.ophys.io.convert_level_1_to_level_2 import convert_level_1_to_level_2


if __name__ == '__main__':

    import matplotlib
    matplotlib.use('Agg')

    cache_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\visual_behavior_pilot_analysis'
    # cache_dir = r'\\allen\programs\braintv\workgroups\ophysdev\OPhysCore\Analysis\2018-08 - Behavior Integration test'
    experiment_id = 736490031
    ophys_data = convert_level_1_to_level_2(experiment_id, cache_dir)
