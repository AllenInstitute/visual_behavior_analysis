import os
import pandas as pd

from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis
import visual_behavior.ophys.response_analysis.utilities as ut


# import logging
# logger = logging.getLogger(__name__)


def get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'change_image_name', 'behavioral_response_type'],
                              flashes=False, use_events=False):
    mega_mdf = pd.DataFrame()
    for experiment_id in experiment_ids:
        print(experiment_id)
        dataset = VisualBehaviorOphysDataset(experiment_id, cache_dir=cache_dir)
        analysis = ResponseAnalysis(dataset, use_events=use_events)
        try:
            if flashes:
                # if analysis.get_flash_response_df_path().split('\\')[-1] in os.listdir(dataset.analysis_dir):
                if 'repeat' in conditions:
                    flash_response_df = analysis.flash_response_df.copy()
                    flash_response_df = flash_response_df[flash_response_df.repeat.isin([1, 5, 10, 15])]
                else:
                    flash_response_df = analysis.flash_response_df.copy()
                flash_response_df['engaged'] = [True if reward_rate > 2 else False for reward_rate in
                                                flash_response_df.reward_rate.values]
                mdf = ut.get_mean_df(flash_response_df, analysis,
                                     conditions=conditions, flashes=True)
                mdf['experiment_id'] = dataset.experiment_id
                mdf = ut.add_metadata_to_mean_df(mdf, dataset.metadata)
                mega_mdf = pd.concat([mega_mdf, mdf])
                # else:
                #     print('problem with',analysis.get_flash_response_df_path().split('\\')[-1],'for',experiment_id)
            else:
                # if analysis.get_trial_response_df_path().split('\\')[-1] in os.listdir(dataset.analysis_dir):
                mdf = ut.get_mean_df(analysis.trial_response_df, analysis,
                                     conditions=conditions)
                mdf['experiment_id'] = dataset.experiment_id
                mdf = ut.add_metadata_to_mean_df(mdf, dataset.metadata)
                mega_mdf = pd.concat([mega_mdf, mdf])
                # else:
                #     print('problem with',analysis.get_trial_response_df_path().split('\\')[-1],'for',experiment_id)
        except:
            print('problem for', experiment_id)
    if flashes:
        type = '_flashes_'
    else:
        type = '_trials_'
    if use_events:
        suffix = '_events'
    else:
        suffix = ''
    if 'level_0' in mega_mdf.keys():
        mega_mdf = mega_mdf.drop(columns='level_0')
    if 'index' in mega_mdf.keys():
        mega_mdf = mega_mdf.drop(columns='index')

    mega_mdf.to_hdf(
        os.path.join(cache_dir, 'multi_session_summary_dfs', 'mean' + type + conditions[2] + suffix + '_df.h5'),
        key='df',
        format='fixed')


if __name__ == '__main__':
    cache_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\visual_behavior_pilot_analysis'
    # manifest = pd.read_csv(os.path.join(cache_dir, 'visual_behavior_data_manifest.csv'))
    # experiment_ids = manifest.experiment_id.values
    experiment_ids = [639253368, 639438856, 639769395, 639932228, 644942849, 645035903]

    get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'change_image_name', 'trial_type'])
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                           conditions=['cell_specimen_id', 'change_image_name', 'behavioral_response_type'])
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                           conditions=['cell_specimen_id', 'change_image_name', 'trial_type'], use_events=True)
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                           conditions=['cell_specimen_id', 'change_image_name', 'behavioral_response_type'],
    #                           use_events=True)
    #
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                           conditions=['cell_specimen_id', 'image_name', 'repeat'], flashes=True)
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                           conditions=['cell_specimen_id', 'image_name', 'repeat'], flashes=True, use_events=True)
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                           conditions=['cell_specimen_id', 'image_name', 'engaged', 'repeat'], flashes=True)
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                           conditions=['cell_specimen_id', 'image_name', 'engaged', 'repeat'], flashes=True,
    #                           use_events=True)
