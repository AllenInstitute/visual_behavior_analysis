import os
import pandas as pd

from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis
import visual_behavior.ophys.response_analysis.utilities as ut


# import logging
# logger = logging.getLogger(__name__)


# def get_multi_session_mean_df(experiment_ids, cache_dir,
#                               conditions=['cell_specimen_id', 'change_image_name', 'behavioral_response_type'],
#                               flashes=False, use_events=False, omitted=False, get_reliability=False):
#     mega_mdf = pd.DataFrame()
#     for experiment_id in experiment_ids:
#         print(experiment_id)
#         try:
#             dataset = VisualBehaviorOphysDataset(experiment_id, cache_dir=cache_dir)
#             analysis = ResponseAnalysis(dataset, use_events=use_events)
#             if flashes:
#                 if omitted:
#                     # print('using omitted flash response df')
#                     flash_response_df = analysis.get_omitted_flash_response_df()
#                     # print(len(flash_response_df))
#                 elif not omitted:
#                     if 'repeat' in conditions:
#                         flash_response_df = analysis.get_flash_response_df()
#                         repeats = [0,5,10,15]
#                         flash_response_df = flash_response_df[flash_response_df.repeat.isin(repeats)]
#                     else:
#                         flash_response_df = analysis.get_flash_response_df()
#                 if len(flash_response_df) > 0:
#                     flash_response_df['engaged'] = [True if reward_rate > 2 else False for reward_rate in
#                                                     flash_response_df.reward_rate.values]
#                     last_flash = flash_response_df.stimulus_presentations_id.unique()[-1]  # sometimes last flash is truncated
#                     flash_response_df = flash_response_df[flash_response_df.stimulus_presentations_id != last_flash]
#                     if 'index' in flash_response_df.keys():
#                         flash_response_df = flash_response_df.drop(columns=['index'])
#                     mdf = ut.get_mean_df(flash_response_df, analysis, conditions=conditions,
#                                          flashes=flashes, omitted=omitted, get_reliability=get_reliability)
#                     mdf['experiment_id'] = dataset.experiment_id
#                     mdf = ut.add_metadata_to_mean_df(mdf, dataset.metadata)
#                     mega_mdf = pd.concat([mega_mdf, mdf])
#                 else:
#                     print('no response_df for',experiment_id)
#                     pass
#             else:
#                 trial_response_df = analysis.get_trial_response_df()
#                 trial_response_df['engaged'] = [True if reward_rate > 2 else False for reward_rate in
#                                                 trial_response_df.reward_rate.values]
#                 mdf = ut.get_mean_df(trial_response_df, analysis, conditions=conditions,
#                                      flashes=flashes, omitted=omitted, get_reliability=get_reliability)
#                 mdf['experiment_id'] = dataset.experiment_id
#                 mdf = ut.add_metadata_to_mean_df(mdf, dataset.metadata)
#                 mega_mdf = pd.concat([mega_mdf, mdf])
#         except Exception as e:  # flake8: noqa: E722
#             print(e)
#             print('problem for', experiment_id)
#     if flashes:
#         if omitted:
#             type = '_omitted_flashes_'
#         else:
#             type = '_flashes_'
#     else:
#         type = '_trials_'



def get_multi_session_mean_df(experiment_ids, cache_dir, df_name,
                              conditions=['cell_specimen_id', 'change_image_name', 'behavioral_response_type'],
                              flashes=False, use_events=False, omitted=False, get_reliability=False, get_pref_stim=True):
    mega_mdf = pd.DataFrame()
    for experiment_id in experiment_ids:
        print(experiment_id)
        try:
            dataset = VisualBehaviorOphysDataset(experiment_id, cache_dir=cache_dir)
            analysis = ResponseAnalysis(dataset, use_events=use_events)
            df = analysis.get_response_df(df_name)
            df['experiment_id'] = dataset.experiment_id
            if 'engaged' in conditions:
                df['engaged'] = [True if reward_rate > 2 else False for reward_rate in df.reward_rate.values]
            if 'running' in conditions:
                df['running'] = [True if mean_running_speed > 5 else False for mean_running_speed in df.mean_running_speed.values]
            mdf = ut.get_mean_df(df, analysis, conditions=conditions, get_pref_stim=get_pref_stim,
                                 flashes=flashes, omitted=omitted, get_reliability=get_reliability)
            mdf['experiment_id'] = dataset.experiment_id
            mdf = ut.add_metadata_to_mean_df(mdf, dataset.metadata)
            mega_mdf = pd.concat([mega_mdf, mdf])
        except Exception as e:  # flake8: noqa: E722
            print(e)
            print('problem for', experiment_id)
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

    if len(conditions) == 4:
        filename = 'mean_' + df_name +'_'+ conditions[1] +'_'+ conditions[2] +'_'+ conditions[3] + suffix + '.h5'
    elif len(conditions) == 3:
        filename = 'mean_' + df_name +'_'+ conditions[1] +'_'+ conditions[2] + suffix + '.h5'
    elif len(conditions) == 2:
        filename = 'mean_' + df_name +'_'+ conditions[1] + suffix + '.h5'
    elif len(conditions) == 1:
        filename = 'mean_' + df_name +'_'+ conditions[0] + suffix + '.h5'

    print('saving multi session mean df to ',filename)
    mega_mdf.to_hdf(
        os.path.join(mega_mdf_write_dir, filename),
        key='df')
    print('saved')
    return mega_mdf


if __name__ == '__main__':
    cache_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_pilot_analysis/visual_behavior_pilot_manuscript_resubmission'

    # Visual Behavior pilot complete dataset
    # experiment_ids = [639253368, 639438856, 639769395, 639932228, 644942849, 645035903,
    #                   645086795, 645362806, 646922970, 647108734, 647551128, 647887770,
    #                   648647430, 649118720, 649318212, 661423848, 663771245, 663773621,
    #                   664886336, 665285900, 665286182, 670396087, 671152642, 672185644,
    #                   672584839, 673139359, 673460976, 685744008, 686726085, 692342909,
    #                   692841424, 693272975, 693862238, 695471168, 696136550, 698244621,
    #                   698724265, 700914412, 701325132, 702134928, 702723649, 712178916,
    #                   712860764, 713525580, 714126693, 715161256, 715228642, 715887471,
    #                   715887497, 716327871, 716337289, 716600289, 716602547, 719321260,
    #                   719996589, 720001924, 720793118, 723037901, 723064523, 723748162,
    #                   723750115, 729951441, 730863840, 731936595, 732911072, 733691636,
    #                   736490031, 736927574, 737471012, 745353761, 745637183, 747248249,
    #                   750469573, 751935154, 752966796, 753931104, 754552635, 754566180,
    #                   754943841, 756715598, 758274779, 760003838, 760400119, 760696146,
    #                   760986090, 761861597, 762214438, 762214650, 766779984, 767424894,
    #                   768223868, 768224465, 768225217, 768865460, 768871217, 769514560,
    #                   770094844, 771381093, 771427955, 772131949, 772696884, 772735942,
    #                   773816712, 773843260, 774370025, 774379465, 775011398, 775429615,
    #                   776042634]

    # pilot study manuscript final expts
    experiment_ids = [644942849, 645035903, 645086795, 645362806, 646922970, 647108734,
                      647551128, 647887770, 639253368, 639438856, 639769395, 639932228,
                      661423848, 663771245, 663773621, 665286182, 670396087, 671152642,
                      672185644, 672584839, 695471168, 696136550, 698244621, 698724265,
                      700914412, 701325132, 702134928, 702723649, 692342909, 692841424,
                      693272975, 693862238, 712178916, 712860764, 713525580, 714126693,
                      715161256, 715887497, 716327871, 716600289, 729951441, 730863840,
                      736490031, 737471012, 715228642, 715887471, 716337289, 716602547,
                      720001924, 720793118, 723064523, 723750115, 719321260, 719996589,
                      723748162, 723037901, 731936595, 732911072, 733691636, 736927574,
                      745353761, 745637183, 747248249, 750469573, 754566180, 754943841,
                      756715598, 758274779, 751935154, 752966796, 753931104, 754552635,
                      766779984, 767424894, 768223868, 768865460, 771381093, 772696884,
                      773816712, 774370025, 771427955, 772131949, 772735942, 773843260,
                      768224465, 768871217, 769514560, 770094844, 760696146, 760986090,
                      762214438, 768225217, 774379465, 775011398, 775429615, 776042634,
                      648647430, 649118720, 649318212, 673139359, 673460976]

    get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id'], flashes=True, omitted=True, get_reliability=True)
    get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id','image_name'], flashes=True, omitted=True, get_reliability=True)
    get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'change_image_name', 'trial_type'])
    get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'image_name'], flashes=True, get_reliability=True)
    get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'image_name', 'repeat'], flashes=True, get_reliability=True)
    get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'image_name', 'engaged'], flashes=True)
    get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'image_name', 'engaged', 'repeat'], flashes=True)
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
