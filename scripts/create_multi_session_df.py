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
    # session_number = float(args.session_number)
    session_number = args.session_number


    # project_code = 'VisualBehaviorMultiscope'
    # session_number = 3

    print(project_code, session_number)

    # params for stim response df creation
    time_window = [-2, 2.1]
    interpolate = True
    output_sampling_rate = 30
    # response_window_duration_seconds = 0.5
    use_extended_stimulus_presentations = False

    # set up conditions to make multi session dfs for
    physio_data_types = ['filtered_events', 'events', 'dff']
    behavior_data_types = ['pupil_width', 'running_speed', 'lick_rate']

    # epoch duration that will be used is coded in io\create_multi_session_df.py

    physio_conditions = [#'cell_specimen_id', 'is_change'], # all stim presentations, change vs. not a change
                         ['cell_specimen_id', 'is_change', 'epoch'], # all stim presentations, change vs no change, x min epochs
                         # ['cell_specimen_id', 'is_change', 'image_name'], # all stim presentations, change vs no change, each image identity
                         # ['cell_specimen_id', 'omitted'], # all omissions
                         ['cell_specimen_id', 'omitted', 'epoch'], # omissions in x min epochs
                         # ['cell_specimen_id', 'is_change'], # only changes
                         ['cell_specimen_id', 'image_name', 'epoch'], # each image, x min epochs
                         # ['cell_specimen_id', 'is_change', 'image_name'], # only changes, each image identity
                         ['cell_specimen_id', 'is_change', 'hit', 'epoch'], # only changes, hit vs. miss, x min epochs
                         # ['cell_specimen_id', 'is_change', 'hit'], # only changes, hit vs. miss
                         ['cell_specimen_id', 'pre_change', 'epoch'], # all stim presentations, pre-change, x min epochs
                         # ['cell_specimen_id', 'omitted', 'pre_omitted'], # all stim presentations, omission or not, pre-omitted or not
                            ]


    behavior_conditions = [#['ophys_experiment_id', 'is_change'],
                            ['ophys_experiment_id', 'is_change', 'epoch'],
                            # ['ophys_experiment_id', 'is_change', 'image_name'],
                            # ['ophys_experiment_id', 'omitted'],
                            ['ophys_experiment_id', 'omitted', 'epoch'],
                            # ['ophys_experiment_id', 'is_change'],
                            ['ophys_experiment_id', 'is_change', 'epoch'],
                            # ['ophys_experiment_id', 'is_change', 'image_name'],
                            ['ophys_experiment_id', 'is_change', 'hit', 'epoch'],
                            # ['ophys_experiment_id', 'is_change', 'hit'],
                            ['ophys_experiment_id', 'pre_change', 'epoch'],
                            # ['ophys_experiment_id', 'omitted', 'pre_omitted'],
                                ]


    # event types corresponding to the above physio and behavior conditions - must be in same sequential order
    event_types_for_conditions = ['all', 'all', 'all', 'all', 'all']
                                  # 'omissions', 'omissions',
                                  # 'changes', 'changes', 'changes',
                                  # 'changes', 'changes',
                                  # 'all', 'all']

    # add engagement state to all conditions
    # for i in range(len(physio_conditions)):
    #     physio_conditions[i].insert(1, 'engagement_state')
    #     behavior_conditions[i].insert(1, 'engagement_state')

    # create dfs for all data types and conditions for physio data
    for data_type in physio_data_types:
    # data_type = 'events'
    #     conditions = ['cell_specimen_id', 'omitted', 'epoch']
        for i, conditions in enumerate(physio_conditions):
            print(conditions)
            event_type = event_types_for_conditions[i]
            print(event_type)
            # event_type = 'all'
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
                                             overwrite=True)
            except Exception as e:
                print('failed to create multi_session_df for', data_type, event_type, conditions)
                print(e)


    # # create dfs for all data types and conditions for behavior data
    # for data_type in behavior_data_types:
    #     for i, conditions in enumerate(behavior_conditions):
    #         event_type = event_types_for_conditions[i]
    #         if event_type == 'omissions':
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
