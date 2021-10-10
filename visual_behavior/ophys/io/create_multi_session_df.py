import os
import pandas as pd

from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache as bpc
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis
import visual_behavior.ophys.response_analysis.utilities as ut
from visual_behavior.data_access import loading


def get_multi_session_df(project_code, session_number, df_name, conditions, use_events=False, filter_events=False,
                         use_extended_stimulus_presentations=True):
    if 'stimulus' in df_name:
        flashes = True
        omitted = False
        get_pref_stim = True
    elif 'omission' in df_name:
        flashes = False
        omitted = True
        get_pref_stim = False
    elif 'trials' in df_name:
        flashes = False
        omitted = False
        get_pref_stim = True
    else:
        print('unable to set params for', df_name)
    if ('run_speed' in df_name) or ('pupil_area' in df_name):
        get_pref_stim = False
    if ('engaged' in conditions) or ('engagement_state' in conditions):
        use_extended_stimulus_presentations = True

    # experiments_table = loading.get_filtered_ophys_experiment_table(release_data_only=True)

    cache_dir = loading.get_platform_analysis_cache_dir()
    cache = bpc.from_s3_cache(cache_dir=cache_dir)
    print(cache_dir)
    experiments_table = cache.get_ophys_experiment_table()

    session_number = float(session_number)
    experiments = experiments_table[(experiments_table.project_code == project_code) &
                                    (experiments_table.session_number == session_number)].copy()
    print('session_types:', experiments.session_type.unique(), ' - there should only be one!')
    session_type = experiments.session_type.unique()[0]

    mega_mdf = pd.DataFrame()
    for experiment_id in experiments.index:
        try:
            print(experiment_id)
            dataset = loading.get_ophys_dataset(experiment_id)
            analysis = ResponseAnalysis(dataset, use_events=use_events, filter_events=filter_events,
                                        use_extended_stimulus_presentations=use_extended_stimulus_presentations)
            df = analysis.get_response_df(df_name)
            df['ophys_experiment_id'] = experiment_id
            if 'passive' in dataset.metadata['session_type']:
                df['lick_on_next_flash'] = False
                df['engaged'] = False
                df['engagement_state'] = 'disengaged'
            if 'running' in conditions:
                df['running'] = [True if mean_running_speed > 2 else False for mean_running_speed in df.mean_running_speed.values]
            if 'large_pupil' in conditions:
                if 'mean_pupil_area' in df.keys():
                    df = df[df.mean_pupil_area.isnull() == False]
                    if len(df) > 100:
                        median_pupil_area = df.mean_pupil_area.median()
                        df['large_pupil'] = [True if mean_pupil_area > median_pupil_area else False for mean_pupil_area in
                                             df.mean_pupil_area.values]
            if 'pre_change' in conditions:
                df = df[df.pre_change.isnull()==False]
            mdf = ut.get_mean_df(df, analysis, conditions=conditions, get_pref_stim=get_pref_stim,
                                 flashes=flashes, omitted=omitted, get_reliability=True,
                                 exclude_omitted_from_pref_stim=False)
            if 'correlation_values' in mdf.keys():
                mdf = mdf.drop(columns=['correlation_values'])
            mdf['ophys_experiment_id'] = experiment_id
            # mdf = ut.add_metadata_to_mean_df(mdf, dataset.metadata)
            print('mean df created for', experiment_id)
            mega_mdf = pd.concat([mega_mdf, mdf])
        except Exception as e:  # flake8: noqa: E722
            print(e)
            print('problem for', experiment_id)

    if 'level_0' in mega_mdf.keys():
        mega_mdf = mega_mdf.drop(columns='level_0')
    if 'index' in mega_mdf.keys():
        mega_mdf = mega_mdf.drop(columns='index')

    filename = loading.get_file_name_for_multi_session_df(df_name, project_code, session_type, conditions, use_events, filter_events)
    mega_mdf_write_dir = os.path.join(loading.get_platform_analysis_cache_dir(), 'multi_session_summary_dfs')
    print('saving multi session mean df as ', filename)
    mega_mdf.to_hdf(os.path.join(mega_mdf_write_dir, filename), key='df')
    print('saved to', mega_mdf_write_dir)
    return mega_mdf
