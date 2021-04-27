import os
import numpy as np
import pandas as pd

import visual_behavior.data_access.loading as loading
import visual_behavior.visualization.utils as utils

from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis

from visual_behavior.ophys.response_analysis import cell_metrics


if __name__ == '__main__':
    import sys
    ophys_experiment_id = int(sys.argv[1])

    ophys_experiment_table = loading.get_filtered_ophys_experiment_table(release_data_only=True)

    import platform

    if platform.system() == 'Linux':
        save_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/single_cell_metrics'
    else:
        save_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\single_cell_metrics'

    ### trace metrics ###
    for use_events in [True, False]:
        trace_metrics = cell_metrics.get_trace_metrics_table(ophys_experiment_id,
                                                         ophys_experiment_table,
                                                         use_events=use_events)
        filename = cell_metrics.get_metrics_df_filename(ophys_experiment_id, 'traces', 'none', 'full_session', use_events)
        filepath = os.path.join(save_dir, 'cell_metrics', filename + '.h5')
        if os.path.exists(filepath):
            os.remove(filepath)
            print('h5 file exists for', ophys_experiment_id, ' - overwriting')
        trace_metrics.to_hdf(filepath, key='df')


    ### event locked response metrics ###
    conditions = ['changes', 'omissions', 'images']
    stimuli = ['all_images', 'pref_image']
    session_subsets = ['engaged', 'disengaged', 'full_session']

    metrics_df = pd.DataFrame()
    for condition in conditions:
        for stimulus in stimuli:
            for session_subset in session_subsets:
                for use_events in [True, False]:
                    try: # code will not always run, such as in the case of passive sessions (no trials that are 'engaged')
                        metrics_df = cell_metrics.generate_metrics_table(ophys_experiment_id, ophys_experiment_table, use_events=use_events,
                                                     condition=condition, session_subset=session_subset, stimuli=stimulus)

                        filename = cell_metrics.get_metrics_df_filename(ophys_experiment_id, condition, stimulus, session_subset, use_events)
                        filepath = os.path.join(save_dir, 'cell_metrics', filename+'.h5')
                        if os.path.exists(filepath):
                            os.remove(filepath)
                            print('h5 file exists for', ophys_experiment_id, ' - overwriting')
                        metrics_df.to_hdf(filepath, key='df')

                    except Exception as e:
                        print('metrics not generated for', condition, stimulus, session_subset, 'events', use_events)
                        print(e)


                        # trace_metrics = cell_metrics.get_trace_metrics_table(ophys_experiment_id, ophys_experiment_table, use_events=use_events)
    #
    # metrics_df = pd.concat([metrics_df, trace_metrics])
    #
    # import platform
    # if platform.system() == 'Linux':
    #     save_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/single_cell_metrics'
    # else:
    #     save_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\single_cell_metrics'
    #
    # # metrics_df.to_csv(os.path.join(save_dir, 'cell_metrics', 'experiment_id_' + str(ophys_experiment_id) + '.csv'))
    # filepath = os.path.join(save_dir, 'cell_metrics', 'experiment_id_' + str(ophys_experiment_id) + '.h5')
    # if os.path.exists(filepath):
    #     os.remove(filepath)
    #     print('h5 file exists for', ophys_experiment_id, ' - overwriting')
    # metrics_df.to_hdf(filepath, key='df')


