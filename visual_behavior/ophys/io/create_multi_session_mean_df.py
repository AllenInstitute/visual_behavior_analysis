import os
import pandas as pd

from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysSession
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis
import visual_behavior.ophys.response_analysis.utilities as ut


# import logging
# logger = logging.getLogger(__name__)


def get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_roi_id', 'change_image_name', 'behavioral_response_type'],
                              flashes=False, use_events=False):
    mega_mdf = pd.DataFrame()
    for experiment_id in experiment_ids:
        print(experiment_id)
        dataset = VisualBehaviorOphysSession(experiment_id)
        analysis = ResponseAnalysis(dataset, use_events=use_events)
        try:
            if flashes:
                if 'repeat' in conditions:
                    flash_response_df = analysis.flash_response_df.copy()
                    repeats = [1, 5, 10, 15]
                    flash_response_df = flash_response_df[flash_response_df.repeat.isin(repeats)]
                else:
                    flash_response_df = analysis.flash_response_df.copy()
                flash_response_df['engaged'] = [True if reward_rate > 2 else False for reward_rate in
                                                flash_response_df.reward_rate.values]
                mdf = ut.get_mean_df(flash_response_df, analysis,
                                     conditions=conditions, flashes=True)
                mdf['ophys_experiment_id'] = dataset.ophys_experiment_id
                mdf = ut.add_metadata_to_mean_df(mdf, dataset.metadata)
                mega_mdf = pd.concat([mega_mdf, mdf])
            else:
                mdf = ut.get_mean_df(analysis.trial_response_df, analysis,
                                     conditions=conditions)
                mdf['ophys_experiment_id'] = dataset.ophys_experiment_id
                mdf = ut.add_metadata_to_mean_df(mdf, dataset.metadata)
                mega_mdf = pd.concat([mega_mdf, mdf])
        except Exception as e:  # flake8: noqa: E722
            print(e)
            print('problem for', experiment_id)
            raise
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

    mega_mdf_write_dir = os.path.join(cache_dir, 'multi_session_summary_dfs')
    if not os.path.exists(mega_mdf_write_dir):
        os.makedirs(mega_mdf_write_dir)

    mega_mdf.to_hdf(
        os.path.join(mega_mdf_write_dir, 'mean' + type + conditions[2] + suffix + '_df.h5'),
        key='df',
        format='fixed')


if __name__ == '__main__':
    cache_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\visual_behavior_production_analysis'

    # VisualBehavior production as of 1/15/19
    experiment_ids = [775614751, 778644591, 787461073, 782675436, 783928214, 783927872,
                      787501821, 787498309, 788490510, 788488596, 788489531, 789359614,
                      790149413, 790709081, 791119849, 791453282, 791980891, 792813858,
                      792812544, 792816531, 792815735, 794381992, 794378505, 795076128,
                      795073741, 795952471, 795952488, 795953296, 795948257, 796106850,
                      796106321, 796108483, 796105823, 796308505, 797255551, 795075034,
                      798403387, 798404219, 799366517, 799368904, 799368262, 803736273,
                      805100431, 805784331, 805784313, 806456687, 806455766, 806989729,
                      807753318, 807752719, 807753334, 807753920, 796105304, 784482326,
                      779335436, 782675457, 791974731, 791979236,
                      800034837, 802649986, 806990245, 808621958,
                      808619526, 808619543, 808621034, 808621015]

    get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'change_image_name', 'trial_type'])
    get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'image_name', 'repeat'], flashes=True)
    get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'image_name', 'engaged', 'repeat'], flashes=True)
    get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'change_image_name', 'behavioral_response_type'])

    get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'change_image_name', 'trial_type'], use_events=True)
    get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'image_name', 'repeat'], flashes=True, use_events=True)
    get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'image_name', 'engaged', 'repeat'], flashes=True,
                              use_events=True)
    get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'change_image_name', 'behavioral_response_type'],
                              use_events=True)
