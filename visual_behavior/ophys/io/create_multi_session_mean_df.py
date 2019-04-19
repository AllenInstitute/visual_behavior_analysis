import os
import pandas as pd

from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis
import visual_behavior.ophys.response_analysis.utilities as ut


# import logging
# logger = logging.getLogger(__name__)


def get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'change_image_name', 'behavioral_response_type'],
                              flashes=False, use_events=False, omitted=False, get_reliability=False):
    mega_mdf = pd.DataFrame()
    for experiment_id in experiment_ids:
        print(experiment_id)
        try:
            dataset = VisualBehaviorOphysDataset(experiment_id, cache_dir=cache_dir)
            analysis = ResponseAnalysis(dataset, use_events=use_events)
            if flashes:
                if omitted:
                    print('using omitted flash response df')
                    flash_response_df = analysis.omitted_flash_response_df.copy()
                    print(len(flash_response_df))
                elif not omitted:
                    if 'repeat' in conditions:
                        flash_response_df = analysis.flash_response_df.copy()
                        repeats = [1, 5, 10, 15]
                        flash_response_df = flash_response_df[flash_response_df.repeat.isin(repeats)]
                    else:
                        flash_response_df = analysis.flash_response_df.copy()
                if len(flash_response_df) > 0:
                    flash_response_df['engaged'] = [True if reward_rate > 2 else False for reward_rate in
                                                    flash_response_df.reward_rate.values]
                    last_flash = flash_response_df.flash_number.unique()[-1]  # sometimes last flash is truncated
                    flash_response_df = flash_response_df[flash_response_df.flash_number != last_flash]
                    if 'index' in flash_response_df.keys():
                        flash_response_df = flash_response_df.drop(columns=['index'])
                    mdf = ut.get_mean_df(flash_response_df, analysis, conditions=conditions,
                                         flashes=flashes, omitted=omitted, get_reliability=get_reliability)
                    mdf['experiment_id'] = dataset.experiment_id
                    mdf = ut.add_metadata_to_mean_df(mdf, dataset.metadata)
                    mega_mdf = pd.concat([mega_mdf, mdf])
                else:
                    print('no omitted flashes for', experiment_id)
                    pass
            else:
                trial_response_df = analysis.trial_response_df.copy()
                trial_response_df['engaged'] = [True if reward_rate > 2 else False for reward_rate in
                                                trial_response_df.reward_rate.values]
                mdf = ut.get_mean_df(trial_response_df, analysis, conditions=conditions,
                                     flashes=flashes, omitted=omitted, get_reliability=get_reliability)
                mdf['experiment_id'] = dataset.experiment_id
                mdf = ut.add_metadata_to_mean_df(mdf, dataset.metadata)
                mega_mdf = pd.concat([mega_mdf, mdf])
        except Exception as e:  # flake8: noqa: E722
            print(e)
            print('problem for', experiment_id)
    if flashes:
        if omitted:
            type = '_omitted_flashes_'
        else:
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

    if len(conditions) == 4:
        filename = 'mean' + type + conditions[1] + '_' + conditions[2] + '_' + conditions[3] + suffix + '_df.h5'
    elif len(conditions) == 3:
        filename = 'mean' + type + conditions[1] + '_' + conditions[2] + suffix + '_df.h5'
    elif len(conditions) == 2:
        filename = 'mean' + type + conditions[1] + suffix + '_df.h5'

    print('saving multi session mean df to ', filename)
    mega_mdf.to_hdf(
        os.path.join(mega_mdf_write_dir, filename),
        key='df',
        format='fixed')
    print('saved')


if __name__ == '__main__':
    cache_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\visual_behavior_production_analysis'

    # VisualBehavior production as of 4/17/19
    experiment_ids = [775614751, 778644591, 783927872, 783928214, 784482326, 787498309,
                      787501821, 788488596, 788489531, 788490510, 789359614, 790149413,
                      790709081, 791119849, 791453282, 791453299, 792812544, 792813858,
                      792815735, 792816531, 794378505, 794381992, 795073741, 795075034,
                      795076128, 795948257, 795952471, 795952488, 795953296, 796105304,
                      796105823, 796106321, 796106850, 796108483, 796306417, 796308505,
                      797255551, 798392580, 798403387, 798404219, 799366517, 799368262,
                      799368904, 803736273, 805100431, 805784313, 805784331, 806455766,
                      806456687, 806989729, 807752719, 807753318, 807753334, 807753920,
                      808619543, 808621015, 808621034, 808621958, 809497730, 809501118,
                      811456530, 811458048, 813083478, 815097949, 815652334, 817267785,
                      817267860, 818073631, 819432482, 819434449, 820307042, 820307518,
                      821011078, 822024770, 822028017, 822028587, 822641265, 822647116,
                      822647135, 822656725, 823392290, 823396897, 823401226, 824333777,
                      825120601, 825130141, 825623170, 826583436, 826585773, 826587940,
                      830093338, 830700781, 830700800, 831330404, 832117336, 833629926,
                      833629942, 833631914, 834275020, 834275038, 834279496, 836258936,
                      836258957, 836260147, 836910438, 836911939, 837296345, 837729902,
                      838849930, 839387542, 840157581, 840702910, 840705705, 841601446,
                      841948542, 841951447, 842513687, 842513705, 842971471, 842973730,
                      842975542, 843518725, 843519218, 843519780, 843520488, 844392195,
                      844395446, 845037454, 845037476, 846490568, 847125577, 847128813,
                      847241639, 848007812, 848692970, 848694045, 848694639, 848697604,
                      848697625, 848698709, 849199228, 849203565, 849203586, 850479305,
                      850489605, 851056106, 851060467, 851932055]

    get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'change_image_name', 'trial_type'])
    get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'image_name', 'repeat'], flashes=True)
    get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'image_name'], flashes=True)
    get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'image_name'], flashes=True, omitted=True)

    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                           conditions=['cell_specimen_id', 'image_name'], flashes=True, omitted=False)
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                           conditions=['cell_specimen_id', 'change_image_name', 'trial_type', 'engaged'])
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                           conditions=['cell_specimen_id', 'image_name', 'engaged'], flashes=True)
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                           conditions=['cell_specimen_id', 'image_name', 'engaged', 'repeat'], flashes=True)
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
