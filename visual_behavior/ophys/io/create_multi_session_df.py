import os
import pandas as pd

# from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache
import visual_behavior.ophys.response_analysis.utilities as ut
from visual_behavior.data_access import loading


def get_multi_session_df(project_code, session_number, conditions, data_type, event_type,
                         time_window=[-3, 3.1], interpolate=True, output_sampling_rate=30,
                         response_window_duration=0.5, epoch_duration_mins=5,
                         use_extended_stimulus_presentations=False, overwrite=True):
    """

    For a given session_number (i.e. 1 for OPHYS_1, 2 for OPHYS_2) within a given project_code, loop through all ophys_experiment_ids, load the SDK dataset object,
    create stimulus_response_df with event aligned traces for provided data_type (ex: 'dff', 'events', 'pupil_width', etc),
    then average across a given set of conditions to get a trial averaged trace for those conditions.

    Ex: if data_type = 'dff', event_type = 'changes', and conditions = ['cell_specimen_id', 'image_name'], this function
    will compute the average change aligned dF/F trace for each 'image_name' for each 'cell_specimen_id'.

    For non-neural timeseries, including data_type = 'pupil_width, 'running_speed', or 'lick_rate', conditions should include
    'ophys_experiment_id' to use as index instead of 'cell_specimen_id'

    trial averaged multi_session_dfs are saved to the directory defined by loading.get_multi_session_df_dir()
    Will overwrite existing dfs if overwrite=True, otherwise will only save the df if the file corresponding to the provided
    project_code and mouse_id does not exist.

    Function can be run for multiple mouse_ids and/or project_codes using /scripts/run_create_multi_session_df.py

    Currently does not work for behavior training sessions - switch to learning_mFISH branch to use with behavior training + ophys


    :param project_code: lims project code to use when identifying what experiment_ids to include in the multi_session_df
    :param session_number: session_number in ophys_experiments_table used to identify the OPHYS session of interest to
                            aggregate and average responses over. must be one of [1, 2, 3, 4, 5, 6].
    :param conditions: columns in stimulus_response_df to group by when averaging across trials / stimulus presentations
                        if use_extended_stimulus_presentations is True, columns available include the set of columns provided in that table (ex: engagement_state)
    :param data_type: which timeseries in dataset object to get event triggered responses for
                        options: 'filtered_events', 'events', 'dff', 'running_speed', 'pupil_diameter', 'lick_rate'
    :param event_type: how to filter stimulus presentations when creating table with loading.get_stimulus_response_df()
                        options: 'all', 'omissions', 'changes'
                        filtering for just changes or just omissions makes loading of stim_response_df much faster than using 'all'
    :param time_window: window over which to extract the event triggered response around each stimulus presentation time
    :param interpolate: Boolean, whether or not to interpolate traces
    :param output_sampling_rate: sampling rate for interpolation, only used if interpolate is True
    :param response_window_duration: window of time, in seconds, relative to the stimulus start_time, over which to compute the mean response
                                        (ex: if response_window_duration = 0.5, the mean cell (or pupil or running) trace in a 500ms window will be computed).
                                        Creates a column called 'mean_response' in the multi_session_df containing this value.
                                        The same window will be applied to the pre-stimulus response period to create a column called 'baseline_response' in the multi_session-df
    :param use_extended_stimulus_presentations: Boolean, whether or not to call loading.extended_stimulus_presentations_table() when loading the dataset object,
                                        setting to True will result in many additional columns being added to the stimulus_presentations_table that can be used as
                                        conditions to group by when computing averaged responses, such as engagement state, time from last lick / change / omission
                                        If False, will include the set of additional stimulus presentation columns that comes by default from
                                        mindscope_utilities.visual_behavior_ophys.data_formatting.get_annotated_stimulus_presentations(dataset, epoch_duration_mins=epoch_duration_mins)
                                        which includes epoch, hit, miss, time_from_last_change etc.
    :param epoch_duration_mins: int, period of time, in minutes, for which to split up session when creating 'epoch' column of stimulus_response_df,
                                        which is an integer value indicating the epoch within session each stimulus presentation belongs to
    :param overwrite: Boolean, if False, will search for existing files for the provided project_code and mouse_id and
                            will not save output if file exists. If True, will overwrite any existing files.

    :return: multi_session_df: dataframe containing trial averaged event triggered responses for a given set of conditions,
                                concatenated over all ophys_experiment_ids for the given mouse_id and project_code

    """

    # cant get prefered stimulus if images are not in the set of conditions
    if ('image_name' in conditions) or ('change_image_name' in conditions):
        get_pref_stim = True
    else:
        get_pref_stim = False
    print('get_pref_stim', get_pref_stim)

    # cache_dir = loading.get_platform_analysis_cache_dir()
    # cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=cache_dir)
    # print(cache_dir)
    # experiments_table = cache.get_ophys_experiment_table()
    # dont include Ai94 experiments because they makes things too slow
    # experiments_table = experiments_table[(experiments_table.reporter_line != 'Ai94(TITL-GCaMP6s)')]

    # limit to platform paper experiments
    experiments_table = loading.get_platform_paper_experiment_table(add_extra_columns=True,
                                                                    limit_to_closest_active=True,
                                                                    include_4x2_data=False)

    print(len(experiments_table), 'platform expts')

    print(experiments_table.session_number.unique(), 'session numbers')
    # session_number = int(session_number)
    session_number = float(session_number)
    experiments = experiments_table[(experiments_table.project_code == project_code) &
                                    (experiments_table.session_number == session_number)].copy()

    print(len(experiments), 'expts after filtering for project code and session number')

    print('session_number:', experiments.session_number.unique(),
          'session_types:', experiments.session_type.unique(),
          ' - there should only be one session_type per session_number')
    session_type = experiments.session_type.unique()[0]

    if 'epoch' not in conditions:  # if epoch isnt one of the conditions, dont put epoch duration into filename
        filename = loading.get_file_name_for_multi_session_df(data_type, event_type, project_code, session_type, conditions,
                                                              epoch_duration_mins=None)
    else:  # otherwise, include it
        filename = loading.get_file_name_for_multi_session_df(data_type, event_type, project_code, session_type,
                                                              conditions, epoch_duration_mins=epoch_duration_mins)
        
    mega_mdf_write_dir = loading.get_multi_session_df_dir(interpolate=interpolate, output_sampling_rate=output_sampling_rate,
                                                          event_type=event_type)
    filepath = os.path.join(mega_mdf_write_dir, filename)

    if not overwrite:  # if we dont want to overwrite
        if os.path.exists(filepath):  # and file exists, dont regenerate
            print('multi_session_df exists for', filepath)
            print('not regenerating')
            process_data = False
        else:  # if file doesnt exist, create it
            print('creating multi session mean df for', filename)
            process_data = True
    else:  # if we do want to overwrite
        process_data = True  # regenerate and save
        print('creating multi session mean df for', filename)

    if process_data:
        mega_mdf = pd.DataFrame()
        for experiment_id in experiments.index.unique():
            try:
                print(experiment_id)
                # get dataset
                dataset = loading.get_ophys_dataset(experiment_id,
                                                    get_extended_stimulus_presentations=use_extended_stimulus_presentations)
                # get stimulus_response_df
                df = loading.get_stimulus_response_df(dataset, data_type=data_type, event_type=event_type, time_window=time_window,
                                                      interpolate=interpolate, output_sampling_rate=output_sampling_rate,
                                                      load_from_file=True, epoch_duration_mins=epoch_duration_mins)
                print(len(df), 'length of stimulus_response_df')
                # use response_window duration from stim response df if it exists
                if response_window_duration in df.keys():
                    response_window_duration = df.response_window_duration.values[0]
                df['ophys_experiment_id'] = str(experiment_id)
                # if using omissions, only include omissions where time from last change is more than 3 seconds
                # if event_type == 'omissions':
                #     df = df[df.time_from_last_change > 3]
                # modify columns for specific conditions
                if 'passive' in dataset.metadata['session_type']:
                    df['lick_on_next_flash'] = False
                    df['engaged'] = False
                    df['engagement_state'] = 'disengaged'
                if 'running_state' in conditions:  # create 'running_state' Boolean column based on threshold on mean_running_speed
                    df['running'] = [True if mean_running_speed > 2 else False for mean_running_speed in
                                     df.mean_running_speed.values]
                if 'pupil_state' in conditions:  # create 'pupil_state' Boolean column based on threshold on mean_pupil_
                    if 'mean_pupil_area' in df.keys():
                        df = df[df.mean_pupil_area.isnull() == False]
                        if len(df) > 100:
                            median_pupil_area = df.mean_pupil_area.median()
                            df['large_pupil'] = [True if mean_pupil_area > median_pupil_area else False for mean_pupil_area in
                                                 df.mean_pupil_area.values]
                if 'pre_change' in conditions:
                    df = df[df.pre_change.isnull() == False]
                print(len(mdf), 'length of stimulus_response_df after filtering')
                # get params for mean df creation from stimulus_response_df
                output_sampling_rate = df.output_sampling_rate.unique()[0]
                print('generating mean response df')
                mdf = ut.get_mean_df(df, conditions=conditions, frame_rate=output_sampling_rate,
                                     window_around_timepoint_seconds=time_window,
                                     response_window_duration_seconds=response_window_duration,
                                     get_pref_stim=get_pref_stim, exclude_omitted_from_pref_stim=True)
                print(len(mdf), 'length of multi session df')
                if 'correlation_values' in mdf.keys():
                    mdf = mdf.drop(columns=['correlation_values'])
                mdf['ophys_experiment_id'] = int(experiment_id)
                print('mean df created for', experiment_id)
                print(len(mega_mdf),'length of mega mdf')
                mega_mdf = pd.concat([mega_mdf, mdf])
                print(len(mega_mdf), 'length of mega mdf after merge')
            except Exception as e:  # flake8: noqa: E722
                print(e)
                print('problem for', experiment_id)

        if 'level_0' in mega_mdf.keys():
            mega_mdf = mega_mdf.drop(columns='level_0')
        if 'index' in mega_mdf.keys():
            mega_mdf = mega_mdf.drop(columns='index')

        # if file of the same name exists, delete & overwrite to prevent files from getting huge
        if os.path.exists(filepath):
            os.remove(filepath)
        print('saving multi session mean df as ', filename)
        mega_mdf.to_hdf(filepath, key='df')
        print('saved to', mega_mdf_write_dir)

        return mega_mdf

    else:
        print('multi_session_df not created')
