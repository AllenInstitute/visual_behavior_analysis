import os
import numpy as np
import pandas as pd
import argparse

import visual_behavior.data_access.loading as loading
import visual_behavior.visualization.utils as utils

from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis

from visual_behavior.ophys.response_analysis import cell_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ophys_experiment_id", type=int,
                        help="Experiment ID to process")
    args = parser.parse_args()
    ophys_experiment_id = args.ophys_experiment_id

    from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache as bpc
    cache_dir = loading.get_platform_analysis_cache_dir()
    print(cache_dir)
    cache = bpc.from_s3_cache(cache_dir=cache_dir)
    ophys_experiment_table = cache.get_ophys_experiment_table()

    # import platform
    # if platform.system() == 'Linux':
    #     save_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/platform_paper_cache/single_cell_metrics'
    # else:
    #     save_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\platform_paper_cache\single_cell_metrics'

    save_dir = loading.get_platform_analysis_cache_dir()
    
    # use filtered events when use_events = True
    filter_events = True

    ### trace metrics ###
    for use_events in [True, False]:

        try:
            trace_metrics = cell_metrics.get_trace_metrics_table(ophys_experiment_id,
                                                                 ophys_experiment_table,
                                                                 use_events=use_events, filter_events=filter_events)
            filename = cell_metrics.get_metrics_df_filename(ophys_experiment_id, 'traces', 'none', 'full_session', use_events, filter_events)
            filepath = os.path.join(save_dir, 'cell_metrics', filename + '.h5')
            if os.path.exists(filepath):
                os.remove(filepath)
                print('h5 file exists for', ophys_experiment_id, ' - overwriting')
            trace_metrics.to_hdf(filepath, key='df')
            print('trace metrics saved for', ophys_experiment_id)
        except Exception as e:
            print('metrics not generated for trace_metrics for experiment', ophys_experiment_id)
            print(e)

    ### event locked response metrics ###
    conditions = ['changes', 'omissions', 'images']
    stimuli = ['all_images', 'pref_image']
    session_subsets = ['engaged', 'disengaged', 'full_session']

    metrics_df = pd.DataFrame()
    for condition in conditions:
        for stimulus in stimuli:
            for session_subset in session_subsets:
                for use_events in [True, False]:
                    try:  # code will not always run, such as in the case of passive sessions (no trials that are 'engaged')
                        metrics_df = cell_metrics.generate_metrics_table(ophys_experiment_id, ophys_experiment_table, use_events=use_events, filter_events=filter_events,
                                                                         condition=condition, session_subset=session_subset, stimuli=stimulus)

                        filename = cell_metrics.get_metrics_df_filename(ophys_experiment_id, condition, stimulus, session_subset, use_events, filter_events)
                        filepath = os.path.join(save_dir, 'cell_metrics', filename + '.h5')
                        if os.path.exists(filepath):
                            os.remove(filepath)
                            print('h5 file exists for', ophys_experiment_id, ' - overwriting')
                        metrics_df.to_hdf(filepath, key='df')

                    except Exception as e:
                        print('metrics not generated for', condition, stimulus, session_subset, 'events', use_events, filter_events, ophys_experiment_id)
                        print(e)

