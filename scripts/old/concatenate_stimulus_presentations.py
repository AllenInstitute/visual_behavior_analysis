#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import visual_behavior.data_access.loading as loading


if __name__ == '__main__':
    import sys
    project_code = sys.argv[1][1:-1]
    session_number = int(sys.argv[2][:-1])
    print(project_code, session_number)
    print(type(project_code), type(session_number))
    save_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/decoding/data'

    experiments_table = loading.get_filtered_ophys_experiment_table()
    experiments = experiments_table[(experiments_table.project_code == project_code) &
                                    (experiments_table.session_number == session_number)].copy()

    stim_df = pd.DataFrame()
    for ophys_session_id in experiments.ophys_session_id.unique():
        try:
            print(ophys_session_id, '-', np.where(experiments.ophys_session_id.unique()==ophys_session_id)[0][0],
                 'out of ', len(experiments.ophys_session_id.unique()))
            ophys_experiment_id = loading.get_ophys_experiment_id_for_ophys_session_id(ophys_session_id)
            dataset = loading.get_ophys_dataset(ophys_experiment_id)
            stim = dataset.extended_stimulus_presentations.copy()
            stim = stim.reset_index()
            stim['ophys_session_id'] = ophys_session_id
            if 'hit_fraction' not in stim.keys():
                stim['reward_rate'] = None
                stim['hit_fraction'] = None
                stim['engagement_state'] = None
                stim['lick_on_next_flash'] = None
            stim['rewarded'] = [True if len(rewards)>0 else False for rewards in stim.rewards.values]
            stim = stim[['ophys_session_id', 'stimulus_presentations_id', 'image_index', 'image_name',
                         'image_name_next_flash', 'image_index_next_flash', 'image_name_previous_flash', 'image_index_previous_flash',
                         'omitted', 'change', 'pre_change', 'mean_running_speed', 'licked', 'rewarded',
                         'reward_rate', 'hit_fraction', 'engagement_state', 'lick_on_next_flash']]
            stim_df = pd.concat([stim_df, stim])
        except Exception as e:
            print('problem for ophys_session_id:', ophys_session_id)
            print(e)
    stim_df.to_hdf(os.path.join(save_dir, 'stimulus_presentations_'+project_code+'_session_'+str(session_number)+'.h5'), key='df')
    print('done')