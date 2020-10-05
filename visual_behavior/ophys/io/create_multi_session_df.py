import os
import pandas as pd

from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis
import visual_behavior.ophys.response_analysis.utilities as ut
from visual_behavior.data_access import loading


def get_multi_session_df(project_code, session_number, df_name, conditions, use_events=False):
    if df_name == 'stimulus_response_df':
        flashes = True
        omitted = False
        get_pref_stim = True
    elif df_name == 'omission_response_df':
        flashes = False
        omitted = True
        get_pref_stim = False
    elif df_name == 'trials_response_df':
        flashes = False
        omitted = False
        get_pref_stim = True
    else:
        print('multi_session_df not yet supported for', df_name)

    experiments_table = loading.get_filtered_ophys_experiment_table()
    experiments = experiments_table[(experiments_table.project_code == project_code) &
                                    (experiments_table.session_number == session_number)].copy()
    print('session_types:',experiments.session_type.unique(),' - there should only be one!')
    session_type = experiments.session_type.unique()[0]

    mega_mdf = pd.DataFrame()
    for experiment_id in experiments.index:
        try:
            print(experiment_id)
            dataset = loading.get_ophys_dataset(experiment_id)
            analysis = ResponseAnalysis(dataset, use_events=use_events, use_extended_stimulus_presentations=True)
            df = analysis.get_response_df(df_name)
            if 'passive' in dataset.metadata['session_type']:
                df['lick_on_next_flash'] = False
                df['engagement_state'] = 'disengaged'
            df['ophys_experiment_id'] = experiment_id
            if 'engaged' in conditions:
                df['engaged'] = [True if reward_rate > 2 else False for reward_rate in df.reward_rate.values]
            if 'running' in conditions:
                df['running'] = [True if mean_running_speed > 2 else False for mean_running_speed in df.mean_running_speed.values]
            if 'large_pupil' in conditions:
                if 'mean_pupil_area' in df.keys():
                    df = df[df.mean_pupil_area.isnull() == False]
                    if len(df) > 100:
                        median_pupil_area = df.mean_pupil_area.median()
                        df['large_pupil'] = [True if mean_pupil_area > median_pupil_area else False for mean_pupil_area in
                                             df.mean_pupil_area.values]
            mdf = ut.get_mean_df(df, analysis, conditions=conditions, get_pref_stim=get_pref_stim,
                                 flashes=flashes, omitted=omitted, get_reliability=True,
                                 exclude_omitted_from_pref_stim=True)
            if 'correlation_values' in mdf.keys():
                mdf = mdf.drop(columns=['correlation_values'])
            mdf['ophys_experiment_id'] = experiment_id
            # mdf = ut.add_metadata_to_mean_df(mdf, dataset.metadata)
            print('mean df created for', experiment_id)
        except Exception as e:  # flake8: noqa: E722
            print(e)
            print('problem for', experiment_id)
    mega_mdf = pd.concat([mega_mdf, mdf])

    if use_events:
        suffix = '_events'
    else:
        suffix = ''
    if 'level_0' in mega_mdf.keys():
        mega_mdf = mega_mdf.drop(columns='level_0')
    if 'index' in mega_mdf.keys():
        mega_mdf = mega_mdf.drop(columns='index')

    filename = loading.get_file_name_for_multi_session_df(df_name, project_code, session_type, conditions, use_events)

    print('saving multi session mean df to ', filename)
    mega_mdf.to_hdf(os.path.join(mega_mdf_write_dir, filename), key='df')
    print('saved')
    return mega_mdf


