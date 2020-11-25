#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import visual_behavior.data_access.loading as loading
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis


if __name__ == '__main__':
    import sys
    project_code = sys.argv[1][1:-1]
    cre_line = sys.argv[2][:-1]
    session_number = sys.argv[3][:-1]
    session_number = int(session_number)
    print(project_code, cre_line, session_number)
    save_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/decoding/data'

    experiments_table = loading.get_filtered_ophys_experiment_table()
    experiments = experiments_table[(experiments_table.project_code == project_code) &
                                    (experiments_table.session_number == session_number)&
                                    (experiments_table.cre_line == cre_line)].copy()

    response_df = pd.DataFrame()
    for ophys_experiment_id in experiments.index.values:
        try:
            print(ophys_experiment_id, '-', np.where(experiments.index.values == ophys_experiment_id)[0][0],
                  'out of ', len(experiments.index.values))
            dataset = loading.get_ophys_dataset(ophys_experiment_id)
            analysis = ResponseAnalysis(dataset, use_extended_stimulus_presentations=False)

            stim_response_df = analysis.get_response_df(df_name='stimulus_response_df')
            sdf = stim_response_df.copy()
            sdf['ophys_experiment_id'] = ophys_experiment_id
            sdf['ophys_session_id'] = loading.get_ophys_session_id_for_ophys_experiment_id(ophys_experiment_id)
            sdf['ophys_frame_rate'] = np.round(analysis.ophys_frame_rate, 2)
            sdf = sdf[['ophys_experiment_id', 'ophys_session_id','stimulus_presentations_id', 'cell_specimen_id', 'trace',
                       'trace_timestamps', 'mean_response', 'baseline_response', 'ophys_frame_rate']]
            response_df = pd.concat([response_df, sdf])
        except Exception as e:
            print('problem for ophys_experiment_id:', ophys_experiment_id)
            print(e)
        response_df.to_hdf(os.path.join(save_dir, 'stimulus_response_dfs_'+project_code+'_'+cre_line+'_session_'+str(session_number)+'.h5'), key='df')
    print('done')

