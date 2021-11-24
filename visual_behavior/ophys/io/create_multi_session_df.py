import os
import pandas as pd

from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache
import visual_behavior.ophys.response_analysis.utilities as ut
from visual_behavior.data_access import loading


def get_multi_session_df(project_code, session_number, conditions, data_type, event_type,
                         time_window=[-3, 3.1], interpolate=True, output_sampling_rate=30,
                         response_window_duration_seconds=0.5, use_extended_stimulus_presentations=False):
    """

    :param project_code:
    :param session_number:
    :param conditions:
    :param data_type:
    :param event_type:
    :param time_window:
    :param interpolate:
    :param output_sampling_rate:
    :param response_window_duration_seconds:
    :param use_extended_stimulus_presentations:
    :return:
    """
    # cant get prefered stimulus if images are not in the set of conditions
    if ('image_name' not in conditions) or ('change_image_name' not in conditions):
        get_pref_stim = False
    else:
        get_pref_stim = True
    # use extended stim presentations when conditions require it (this is not an exhaustive list of conditions)
    if ('engaged' in conditions) or ('engagement_state' in conditions) or ('epoch' in conditions):
        use_extended_stimulus_presentations = True

    cache_dir = loading.get_platform_analysis_cache_dir()
    cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=cache_dir)
    print(cache_dir)
    experiments_table = cache.get_ophys_experiment_table()

    session_number = float(session_number)
    experiments = experiments_table[(experiments_table.project_code == project_code) &
                                    (experiments_table.session_number == session_number)].copy()
    print('session_types:', experiments.session_type.unique(),
          ' - there should only be one session_type per session_number')
    session_type = experiments.session_type.unique()[0]

    filename = loading.get_file_name_for_multi_session_df(data_type, event_type, project_code, session_type, conditions)
    print('creating multi session mean df for', filename)

    mega_mdf = pd.DataFrame()
    for experiment_id in experiments.index:
        try:
            print(experiment_id)
            # get dataset
            dataset = loading.get_ophys_dataset(experiment_id,
                                                get_extended_stimulus_presentations=use_extended_stimulus_presentations)
            # get stimulus_response_df
            df = loading.get_stimulus_response_df(dataset, data_type=data_type, event_type=event_type, time_window=time_window,
                                                  interpolate=interpolate, output_sampling_rate=output_sampling_rate,
                                                  load_from_file=True)
            df['ophys_experiment_id'] = experiment_id
            # if using omissions, only include omissions where time from last change is more than 3 seconds
            if event_type == 'omissions':
                df = df[df.time_from_last_change>3]
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
                                 response_window_duration_seconds=response_window_duration_seconds,
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

    filename = loading.get_file_name_for_multi_session_df(data_type, event_type, project_code, session_type, conditions)
    mega_mdf_write_dir = loading.get_multi_session_df_df_dir(interpolate=interpolate,
                                                             output_sampling_rate=output_sampling_rate)
    print('saving multi session mean df as ', filename)
    mega_mdf.to_hdf(os.path.join(mega_mdf_write_dir, filename), key='df')
    print('saved to', mega_mdf_write_dir)

    return mega_mdf
