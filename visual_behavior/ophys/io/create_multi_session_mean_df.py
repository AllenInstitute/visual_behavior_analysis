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
                if 'repeat' in conditions:
                    flash_response_df = analysis.flash_response_df.copy()
                    repeats = [1,5,10,15]
                    flash_response_df = flash_response_df[flash_response_df.repeat.isin(repeats)]
                else:
                    flash_response_df = analysis.flash_response_df.copy()
                flash_response_df['engaged'] = [True if reward_rate > 2 else False for reward_rate in
                                                flash_response_df.reward_rate.values]
                mdf = ut.get_mean_df(flash_response_df, analysis,
                                     conditions=conditions, flashes=True)
                mdf['experiment_id'] = dataset.experiment_id
                mdf = ut.add_metadata_to_mean_df(mdf, dataset.metadata)
                mega_mdf = pd.concat([mega_mdf, mdf])
            else:
                mdf = ut.get_mean_df(analysis.trial_response_df, analysis,
                                     conditions=conditions)
                mdf['experiment_id'] = dataset.experiment_id
                mdf = ut.add_metadata_to_mean_df(mdf, dataset.metadata)
                mega_mdf = pd.concat([mega_mdf, mdf])
        except Exception as e:  # flake8: noqa: E722
            print(e)
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

    mega_mdf_write_dir = os.path.join(cache_dir, 'multi_session_summary_dfs')
    if not os.path.exists(mega_mdf_write_dir):
        os.makedirs(mega_mdf_write_dir)

    if len(conditions) == 2:
        filename = 'mean' + type + conditions[2] + suffix + '_df.h5'
    elif len(conditions) == 1:
        filename = 'mean' + type + conditions[1] + suffix + '_df.h5'

    mega_mdf.to_hdf(
        os.path.join(mega_mdf_write_dir, filename),
        key='df',
        format='fixed')


if __name__ == '__main__':
    cache_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\visual_behavior_pilot_analysis'

    # #VisualBehaviorDevelopment - complete dataset as of 11/15/18
    experiment_ids = [639253368, 639438856, 639769395, 639932228, 644942849, 645035903,
                      645086795, 645362806, 646922970, 647108734, 647551128, 647887770,
                      648647430, 649118720, 649318212, 661423848, 663771245, 663773621,
                      664886336, 665285900, 665286182, 670396087, 671152642, 672185644,
                      672584839, 673139359, 673460976, 685744008, 686726085, 692342909,
                      692841424, 693272975, 693862238, 695471168, 696136550, 698244621,
                      698724265, 700914412, 701325132, 702134928, 702723649, 712178916,
                      712860764, 713525580, 714126693, 715161256, 715228642, 715887471,
                      715887497, 716327871, 716337289, 716600289, 716602547, 719321260,
                      719996589, 720001924, 720793118, 723037901, 723064523, 723748162,
                      723750115, 729951441, 730863840, 731936595, 732911072, 733691636,
                      736490031, 736927574, 737471012, 745353761, 745637183, 747248249,
                      750469573, 751935154, 752966796, 753931104, 754552635, 754566180,
                      754943841, 756715598, 758274779, 760003838, 760400119, 760696146,
                      760986090, 761861597, 762214438, 762214650, 766779984, 767424894,
                      768223868, 768224465, 768225217, 768865460, 768871217, 769514560,
                      770094844, 771381093, 771427955, 772131949, 772696884, 772735942,
                      773816712, 773843260, 774370025, 774379465, 775011398, 775429615,
                      776042634, 756565411]

    get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'image_name'], flashes=True)
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                           conditions=['cell_specimen_id', 'change_image_name', 'trial_type'])
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                           conditions=['cell_specimen_id', 'image_name', 'repeat'], flashes=True)
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                           conditions=['cell_specimen_id', 'image_name', 'engaged', 'repeat'], flashes=True)
    # get_multi_session_mean_df(experiment_ids, cache_dir,
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
