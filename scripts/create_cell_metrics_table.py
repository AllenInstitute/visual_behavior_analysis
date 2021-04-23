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

    conditions = ['changes', 'omissions', 'images']
    stimuli = ['all_images', 'pref_image']
    session_subsets = ['engaged', 'disengaged', 'full_session']

    metrics_df = pd.DataFrame()
    for condition in conditions:
        for stimulus in stimuli:
            for session_subset in session_subsets:
                for use_events in [True, False]:
                    tmp = cell_metrics.generate_metrics_table(ophys_experiment_id, ophys_experiment_table, use_events=use_events,
                                                 condition=condition, session_subset=session_subset, stimuli=stimulus)
                    tmp = tmp.reset_index()
                    metrics_df = pd.concat([metrics_df, tmp])

    trace_metrics = cell_metrics.get_trace_metrics_table(ophys_experiment_id, ophys_experiment_table, use_events=use_events)

    metrics_df = pd.concat([metrics_df, trace_metrics])

    import platform
    if platform.system() == 'Linux':
        save_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/single_cell_metrics'
    else:
        save_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\single_cell_metrics'

    metrics_df.to_csv(os.path.join(save_dir, 'cell_metrics', 'experiment_id_' + str(ophys_experiment_id) + '.csv'))


