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
    session_number = float(args.session_number)

    print(project_code, session_number)

    # omissions

    # df_name = 'omission_response_df'
    # conditions = ['cell_specimen_id']
    #
    # print('creating multi_session_df for', df_name, ', ', project_code, ', session number', session_number, conditions)
    # df = io.get_multi_session_df(project_code, session_number, df_name, conditions, use_events=False, filter_events=False,
    #                              use_extended_stimulus_presentations=False)
    #
    # df_name = 'omission_response_df'
    # conditions = ['cell_specimen_id', 'epoch']
    #
    # print('creating multi_session_df for', df_name, ', ', project_code, ', session number', session_number, conditions)
    # df = io.get_multi_session_df(project_code, session_number, df_name, conditions, use_events=False, filter_events=False,
    #                              use_extended_stimulus_presentations=True)

    # trials

    # df_name = 'trials_response_df'
    # conditions = ['cell_specimen_id', 'stimulus_change']
    #
    # print('creating multi_session_df for', df_name, ', ', project_code, ', session number', session_number, conditions)
    # df = io.get_multi_session_df(project_code, session_number, df_name, conditions, use_events=True, filter_events=False,
    #                              use_extended_stimulus_presentations=False)
    #
    # df_name = 'trials_response_df'
    # conditions = ['cell_specimen_id', 'stimulus_change', 'epoch']
    #
    # print('creating multi_session_df for', df_name, ', ', project_code, ', session number', session_number, conditions)
    # df = io.get_multi_session_df(project_code, session_number, df_name, conditions, use_events=True, filter_events=False,
    #                              use_extended_stimulus_presentations=True)
    #
    # df_name = 'trials_response_df'
    # conditions = ['cell_specimen_id', 'stimulus_change', 'change_image_name', 'epoch']
    #
    # print('creating multi_session_df for', df_name, ', ', project_code, ', session number', session_number, conditions)
    # df = io.get_multi_session_df(project_code, session_number, df_name, conditions, use_events=True, filter_events=False,
    #                              use_extended_stimulus_presentations=True)

    # print('done')

    # stim response df

    df_name = 'stimulus_response_df'
    conditions = ['cell_specimen_id', 'pre_change', 'image_name', 'epoch']

    print('creating multi_session_df for', df_name, ', ', project_code, ', session number', session_number, conditions)
    df = io.get_multi_session_df(project_code, session_number, df_name, conditions, use_events=True,
                                 filter_events=False,
                                 use_extended_stimulus_presentations=True)

    df_name = 'stimulus_response_df'
    conditions = ['cell_specimen_id', 'is_change',  'image_name', 'epoch']

    print('creating multi_session_df for', df_name, ', ', project_code, ', session number', session_number, conditions)
    df = io.get_multi_session_df(project_code, session_number, df_name, conditions, use_events=True, filter_events=False,
                                 use_extended_stimulus_presentations=True)


    df_name = 'stimulus_response_df'
    conditions = ['cell_specimen_id', 'is_change', 'epoch']

    print('creating multi_session_df for', df_name, ', ', project_code, ', session number', session_number, conditions)
    df = io.get_multi_session_df(project_code, session_number, df_name, conditions, use_events=True,
                                 filter_events=False,
                                 use_extended_stimulus_presentations=True)


    #
    # # pupil ###
    #
    # # omissions
    #
    # df_name = 'omission_pupil_area_df'
    # conditions = ['ophys_experiment_id']
    #
    # print('creating multi_session_df for', df_name, ', ', project_code, ', session number', session_number, conditions)
    # df = io.get_multi_session_df(project_code, session_number, df_name, conditions, use_events=False,
    #                              filter_events=False,
    #                              use_extended_stimulus_presentations=False)
    #
    # df_name = 'omission_pupil_area_df'
    # conditions = ['ophys_experiment_id', 'epoch']
    #
    # print('creating multi_session_df for', df_name, ', ', project_code, ', session number', session_number, conditions)
    # df = io.get_multi_session_df(project_code, session_number, df_name, conditions, use_events=False, filter_events=False,
    #                              use_extended_stimulus_presentations=True)
    #
    # # trials
    #
    # df_name = 'trials_pupil_area_df'
    # conditions = ['ophys_experiment_id', 'stimulus_change']
    #
    # print('creating multi_session_df for', df_name, ', ', project_code, ', session number', session_number, conditions)
    # df = io.get_multi_session_df(project_code, session_number, df_name, conditions, use_events=False,
    #                              filter_events=False,
    #                              use_extended_stimulus_presentations=False)
    #
    # df_name = 'trials_pupil_area_df'
    # conditions = ['ophys_experiment_id', 'stimulus_change', 'epoch']
    #
    # print('creating multi_session_df for', df_name, ', ', project_code, ', session number', session_number, conditions)
    # df = io.get_multi_session_df(project_code, session_number, df_name, conditions, use_events=False, filter_events=False,
    #                              use_extended_stimulus_presentations=True)
    #
    # df_name = 'trials_pupil_area_df'
    # conditions = ['ophys_experiment_id', 'stimulus_change', 'change_image_name']
    #
    # print('creating multi_session_df for', df_name, ', ', project_code, ', session number', session_number, conditions)
    # df = io.get_multi_session_df(project_code, session_number, df_name, conditions, use_events=False, filter_events=False,
    #                              use_extended_stimulus_presentations=False)
    #
    # print('done')

