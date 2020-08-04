import os
import pandas as pd

from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis
import visual_behavior.ophys.response_analysis.utilities as ut
from visual_behavior.data_access import loading


def get_multi_session_mean_df(experiment_ids, analysis_cache_dir, df_name,
                              conditions=['cell_specimen_id', 'change_image_name', 'behavioral_response_type'],
                              flashes=False, use_events=False, omitted=False, get_reliability=False, get_pref_stim=True,
                              exclude_omitted_from_pref_stim=True, use_sdk_dataset=True):
    experiments_table = loading.get_filtered_ophys_experiment_table()
    mega_mdf = pd.DataFrame()
    for experiment_id in experiment_ids:
        print(experiment_id)
        # try:
        if use_sdk_dataset:
            dataset = loading.get_ophys_dataset(experiment_id)
        else:
            dataset = VisualBehaviorOphysDataset(experiment_id, analysis_cache_dir)
        analysis = ResponseAnalysis(dataset, use_events=use_events, overwrite_analysis_files=False, use_extended_stimulus_presentations=True)
        df = analysis.get_response_df(df_name)
        df['ophys_experiment_id'] = dataset.ophys_experiment_id
        df['project_code'] = experiments_table.loc[experiment_id].project_code
        df['session_type'] = experiments_table.loc[experiment_id].session_type
        # if 'engaged' in conditions:
        #     df['engaged'] = [True if reward_rate > 2 else False for reward_rate in df.reward_rate.values]
        if 'running' in conditions:
            df['running'] = [True if window_running_speed > 5 else False for window_running_speed in
                             df.window_running_speed.values]
        # if 'large_pupil' in conditions:
        #     if 'mean_pupil_area' in df.keys():
        #         df = df[df.mean_pupil_area.isnull() == False]
        #         if len(df) > 100:
        #             median_pupil_area = df.mean_pupil_area.median()
        #             df['large_pupil'] = [True if mean_pupil_area > median_pupil_area else False for mean_pupil_area in
        #                                  df.mean_pupil_area.values]
        mdf = ut.get_mean_df(df, analysis, conditions=conditions, get_pref_stim=get_pref_stim,
                             flashes=flashes, omitted=omitted, get_reliability=get_reliability,
                             exclude_omitted_from_pref_stim=exclude_omitted_from_pref_stim)
        mdf['ophys_experiment_id'] = dataset.ophys_experiment_id
        dataset.metadata['reporter_line'] = dataset.metadata['reporter_line'][0]
        dataset.metadata['driver_line'] = dataset.metadata['driver_line'][0]
        metadata = pd.DataFrame(dataset.metadata, index=[experiment_id])
        mdf = ut.add_metadata_to_mean_df(mdf, metadata)
        mega_mdf = pd.concat([mega_mdf, mdf])
        # except Exception as e:  # flake8: noqa: E722
        #     print(e)
        #     print('problem for', experiment_id)
    if use_events:
        suffix = '_events'
    else:
        suffix = ''
    if 'level_0' in mega_mdf.keys():
        mega_mdf = mega_mdf.drop(columns='level_0')
    if 'index' in mega_mdf.keys():
        mega_mdf = mega_mdf.drop(columns='index')

    mega_mdf_write_dir = os.path.join(analysis_cache_dir, 'multi_session_summary_dfs')
    if not os.path.exists(mega_mdf_write_dir):
        os.makedirs(mega_mdf_write_dir)

    if len(conditions) == 5:
        filename = 'mean_' + df_name + '_' + conditions[1] + '_' + conditions[2] + '_' + conditions[3] + '_' + \
                   conditions[4] + suffix + '.h5'
    elif len(conditions) == 4:
        filename = 'mean_' + df_name + '_' + conditions[1] + '_' + conditions[2] + '_' + conditions[
            3] + suffix + '.h5'
    elif len(conditions) == 3:
        filename = 'mean_' + df_name + '_' + conditions[1] + '_' + conditions[2] + suffix + '.h5'
    elif len(conditions) == 2:
        filename = 'mean_' + df_name + '_' + conditions[1] + suffix + '.h5'
    elif len(conditions) == 1:
        filename = 'mean_' + df_name + '_' + conditions[0] + suffix + '.h5'

    print('saving multi session mean df to ', filename)
    mega_mdf.to_hdf(
        os.path.join(mega_mdf_write_dir, filename),
        key='df')
    print('saved')
    return mega_mdf


if __name__ == '__main__':
    cache_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\visual_behavior_production_analysis'

    # VisualBehaviorTask1B 7/29/19
    # experiment_ids = [899085531, 901559828, 901560861, 902474078, 902487182, 902487200,
    #                902490609, 903396212, 903403819, 903405627, 904155140, 904155155,
    #                904161089, 905495248, 908350502, 908350518, 909090479, 910213154,
    #                910600099, 910608808, 911146705, 911149119, 911149136, 911642167,
    #                911642184]

    # platform = 'scientifica'

    # get_multi_session_mean_df(experiment_ids, cache_dir, platform=platform,
    #                           conditions=['cell_specimen_id', 'image_name'], flashes=True, omitted=False)
    # get_multi_session_mean_df(experiment_ids, cache_dir, platform=platform,
    #                           conditions=['cell_specimen_id', 'change_image_name', 'trial_type'])
    # get_multi_session_mean_df(experiment_ids, cache_dir, platform=platform,
    #                           conditions=['cell_specimen_id', 'image_name'], flashes=True, omitted=True)
    # get_multi_session_mean_df(experiment_ids, cache_dir, platform=platform,
    #                           conditions=['cell_specimen_id', 'image_name', 'repeat'], flashes=True)
    # get_multi_session_mean_df(experiment_ids, cache_dir, platform=platform,
    #                           conditions=['cell_specimen_id', 'image_name', 'engaged', 'repeat'], flashes=True)
    # get_multi_session_mean_df(experiment_ids, cache_dir, platform=platform,
    #                           conditions=['cell_specimen_id', 'change_image_name', 'behavioral_response_type'])
    #
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                           conditions=['cell_specimen_id', 'change_image_name', 'trial_type'], use_events=True)
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                           conditions=['cell_specimen_id', 'image_name', 'repeat'], flashes=True, use_events=True)
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                           conditions=['cell_specimen_id', 'image_name', 'engaged', 'repeat'], flashes=True,
    #                           use_events=True)
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                           conditions=['cell_specimen_id', 'change_image_name', 'behavioral_response_type'],
    #                           use_events=True)
