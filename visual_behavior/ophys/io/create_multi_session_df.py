import os
import pandas as pd

from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache
import visual_behavior.ophys.response_analysis.utilities as ut
from visual_behavior.data_access import loading


def get_multi_session_df(project_code, mouse_id, conditions, data_type, event_type,
                         time_window=[-3, 3.1], interpolate=True, output_sampling_rate=30,
                         response_window_duration=0.5, use_extended_stimulus_presentations=False, overwrite=False):
    """

    :param project_code:
    :param mouse_id:
    :param conditions:
    :param data_type:
    :param event_type:
    :param time_window:
    :param interpolate:
    :param output_sampling_rate:
    :param response_window_duration:
    :param use_extended_stimulus_presentations:
    :return:
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
    # # dont include Ai94 experiments because they makes things too slow
    # experiments_table = experiments_table[(experiments_table.reporter_line != 'Ai94(TITL-GCaMP6s)')]

    experiments_table = loading.get_filtered_ophys_experiment_table()
    experiments_table = experiments_table[experiments_table.project_code == 'LearningmFISHTask1A']
    print(len(experiments_table), 'expts in experiments table')

    # session_type = float(session_type)
    experiments = experiments_table[(experiments_table.project_code == project_code) &
                                    (experiments_table.mouse_id == str(mouse_id))].copy()

    mouse_id = experiments.mouse_id.unique()[0]

    filename = loading.get_file_name_for_multi_session_df(data_type, event_type, project_code, mouse_id, conditions)
    mega_mdf_write_dir = loading.get_multi_session_df_dir(interpolate=interpolate, output_sampling_rate=output_sampling_rate,
                                                          event_type=event_type)
    filepath = os.path.join(mega_mdf_write_dir, filename)

    if not overwrite: # if we dont want to overwrite
        if os.path.exists(filepath): # and file exists, dont regenerate
            print('multi_session_df exists for', filepath)
            print('not regenerating')
            process_data = False
        else: # if file doesnt exist, create it
            print('creating multi session mean df for', filename)
            process_data = True
    else: # if we do want to overwrite
        process_data = True # regenerate and save
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
                                                      load_from_file=True)
                # use response_window duration from stim response df if it exists
                if response_window_duration in df.keys():
                    response_window_duration = df.response_window_duration.values[0]
                df['ophys_experiment_id'] = experiment_id
                # if using omissions, only include omissions where time from last change is more than 3 seconds
                # if event_type == 'omissions':
                #     df = df[df.time_from_last_change>3]
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
                # get params for mean df creation from stimulus_response_df
                output_sampling_rate = df.frame_rate.unique()[0]
                timestamps = df.trace_timestamps.values[0]
                window_around_timepoint_seconds = [timestamps[0], timestamps[-1]]

                mdf = ut.get_mean_df(df, conditions=conditions, frame_rate=output_sampling_rate,
                                     window_around_timepoint_seconds=time_window,
                                     response_window_duration_seconds=response_window_duration,
                                     get_pref_stim=get_pref_stim, exclude_omitted_from_pref_stim=True)
                if 'correlation_values' in mdf.keys():
                    mdf = mdf.drop(columns=['correlation_values'])
                mdf['ophys_experiment_id'] = experiment_id
                print('mean df created for', experiment_id)
                mega_mdf = pd.concat([mega_mdf, mdf])
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



