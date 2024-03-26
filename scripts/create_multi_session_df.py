#!/usr/bin/env python

import os
import argparse
import numpy as np
import pandas as pd
import visual_behavior.data_access.loading as loading
import visual_behavior.ophys.io.create_multi_session_df as io


if __name__ == '__main__':
    # define args
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_code', type=str, help='project code to use')
    parser.add_argument('--session_number', type=str, help='session number to use')
    args = parser.parse_args()
    project_code = args.project_code
    session_number = args.session_number


    print(project_code, session_number)

    # params for stim response df creation
    time_window = [-2, 2.1]
    interpolate = True
    output_sampling_rate = 30
    use_extended_stimulus_presentations = False
    epoch_duration_mins = 5

    # set up conditions to make multi session dfs for
    physio_data_types = ['events', ] #'filtered_events', 'dff']
    behavior_data_types = ['pupil_width', 'running_speed', 'lick_rate']

    physio_conditions = [['cell_specimen_id', 'is_change', 'omitted', 'epoch']]
    # physio_conditions = [['cell_specimen_id', 'is_change'], # all stim presentations, change vs. not a change
    #                      ['cell_specimen_id', 'omitted'], # all omissions
    #                      ['cell_specimen_id', 'pre_change'], # all omissions
    #                      ['cell_specimen_id', 'is_change', 'epoch'], # all stim presentations, change vs no change, x min epochs
    #                      ['cell_specimen_id', 'omitted', 'epoch'], # omissions in x min epochs
    #                      ['cell_specimen_id', 'pre_change', 'epoch'],  # all stim presentations, pre-change, x min epochs
    #                      ['cell_specimen_id', 'image_name', 'epoch'], # each image, x min epochs
    #                      ['cell_specimen_id', 'is_change', 'omitted', 'epoch'],  # each image, x min epochs
    #                      ['cell_specimen_id', 'is_change', 'image_name'], # all stim presentations, change vs no change, each image identity
    #                      ['cell_specimen_id', 'is_change', 'hit'],  # only changes, hit vs. miss
    #                      ['cell_specimen_id', 'is_change', 'hit', 'epoch'], # only changes, hit vs. miss, x min epochs
    #                      ['cell_specimen_id', 'omitted', 'pre_omitted'], # all stim presentations, omission or not, pre-omitted or not
    #                      ]

    # event types corresponding to the above physio conditions - must be in same sequential order
    physio_event_types_for_conditions = ['all']
    # physio_event_types_for_conditions = ['changes', 'omissions', 'all', 'changes', 'omissions',
    #                                      'all', 'all', 'all', 'changes', 'changes', 'changes', 'all']


    behavior_conditions = [['ophys_experiment_id', 'omitted'],
                            ['ophys_experiment_id', 'omitted', 'epoch'],
                            ['ophys_experiment_id', 'is_change'],
                            ['ophys_experiment_id', 'is_change', 'epoch'],
                            ['ophys_experiment_id', 'is_change', 'image_name'],
                            ['ophys_experiment_id', 'is_change', 'hit', 'epoch'],
                            ['ophys_experiment_id', 'is_change', 'hit'],
                            ['ophys_experiment_id', 'pre_change'],
                            ['ophys_experiment_id', 'pre_change', 'epoch'],
                            ['ophys_experiment_id', 'omitted', 'pre_omitted'],]

    behavior_event_types_for_conditions = ['omissions', 'omissions', 'changes', 'changes', 'changes',
                                           'changes','changes', 'all', 'all', 'all']


    # add engagement state to all conditions
    # for i in range(len(physio_conditions)):
    #     physio_conditions[i].insert(1, 'engagement_state')
    #     behavior_conditions[i].insert(1, 'engagement_state')

    # create dfs for all data types and conditions for physio data
    # data_type = 'dff'
    # conditions = ['cell_specimen_id', 'omitted', 'epoch']
    for data_type in physio_data_types:
        for i, conditions in enumerate(physio_conditions):
            print(conditions)
            event_type = physio_event_types_for_conditions[i]
            # event_type = 'all'
            print(event_type)
            if 'omitted' in conditions:
                response_window_duration = 0.75
            else:
                response_window_duration = 0.5
            print('creating multi_session_df for', data_type, event_type, conditions)
            try: # use try except so that it skips over any conditions that fail to generate for some reason
                df = io.get_multi_session_df(project_code, session_number, conditions, data_type, event_type,
                                             time_window=time_window, interpolate=interpolate, output_sampling_rate=output_sampling_rate,
                                             response_window_duration=response_window_duration,
                                             use_extended_stimulus_presentations=use_extended_stimulus_presentations,
                                             epoch_duration_mins=epoch_duration_mins, overwrite=True)
            except Exception as e:
                print('failed to create multi_session_df for', data_type, event_type, conditions)
                print(e)


    # # create dfs for all data types and conditions for behavior data
    # for data_type in behavior_data_types:
    #     for i, conditions in enumerate(behavior_conditions):
    #         print(conditions)
    #         event_type = behavior_event_types_for_conditions[i]
    #         # event_type = 'all'
    #         print(event_type)
    #         if 'omitted' in conditions:
    #             response_window_duration = 0.75
    #         else:
    #             response_window_duration = 0.5
    #         print('creating multi_session_df for', data_type, event_type, conditions)
    #         try: # use try except so that it skips over any conditions that fail to generate for some reason
    #             df = io.get_multi_session_df(project_code, session_number, conditions, data_type, event_type,
    #                                          time_window=time_window, interpolate=interpolate,
    #                                          output_sampling_rate=output_sampling_rate,
    #                                          response_window_duration=response_window_duration,
    #                                          use_extended_stimulus_presentations=use_extended_stimulus_presentations,
    #                                          overwrite=True)
    #         except Exception as e:
    #             print('failed to create multi_session_df for', data_type, event_type, conditions)
    #             print(e)
