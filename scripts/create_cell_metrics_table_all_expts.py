import os
import numpy as np
import pandas as pd

import visual_behavior.data_access.loading as loading
import visual_behavior.visualization.utils as utils

from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis

from visual_behavior.ophys.response_analysis import cell_metrics


if __name__ == '__main__':

    ophys_experiment_table = loading.get_filtered_ophys_experiment_table(release_data_only=True)
    ophys_experiment_ids = ophys_experiment_table.index.values

    cache_dir = loading.get_platform_analysis_cache_dir()
    print(cache_dir)

    ### trace metrics ###
    for use_events in [False]:
        trace_metrics = pd.DataFrame()
        for ophys_experiment_id in ophys_experiment_ids:
            try:
                tmp = cell_metrics.get_trace_metrics_table(ophys_experiment_id,
                                                                 ophys_experiment_table,
                                                                 use_events=use_events)
                trace_metrics = pd.concat([trace_metrics, tmp])
                print(ophys_experiment_id)
            except Exception as e:
                print('trace metrics not generated for', ophys_experiment_id)
                print(e)
        filename = cell_metrics.get_metrics_df_filename('all_experiments', 'traces', 'none', 'full_session', use_events)
        filepath = os.path.join(cache_dir, ,'cell_metrics', filename + '.h5')
        if os.path.exists(filepath):
            os.remove(filepath)
            print('h5 file exists for', ophys_experiment_id, ' - overwriting')
        trace_metrics.to_hdf(filepath, key='df')
        print('all experiments trace file saved')


    ### event locked response metrics ###
    conditions = ['changes', 'omissions', 'images']
    stimuli = ['all_images', 'pref_image']
    session_subsets = ['engaged', 'disengaged', 'full_session']

    for condition in conditions:
        for stimulus in stimuli:
            for session_subset in session_subsets:
                for use_events in [True, False]:
                    metrics_df = pd.DataFrame()
                    print(condition, stimulus, session_subset, use_events)
                    for ophys_experiment_id in ophys_experiment_ids:
                        try: # code will not always run, such as in the case of passive sessions (no trials that are 'engaged')
                            tmp = cell_metrics.generate_metrics_table(ophys_experiment_id, ophys_experiment_table, use_events=use_events,
                                                         condition=condition, session_subset=session_subset, stimuli=stimulus)
                            metrics_df = pd.concat([metrics_df, tmp])
                        except Exception as e:
                            print('metrics not generated for', ophys_experiment_id)
                            print(e)

                    filename = cell_metrics.get_metrics_df_filename('all_experiments', condition, stimulus, session_subset, use_events)
                    filepath = os.path.join(save_dir, 'cell_metrics', filename+'.h5')
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        print('h5 file exists for', ophys_experiment_id, ' - overwriting')
                    metrics_df.to_hdf(filepath, key='df')
                    print('all experiments file saved')



