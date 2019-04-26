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
                        repeats = [1,5,10,15]
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
                    print('no omitted flashes for',experiment_id)
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

    mega_mdf_write_dir = os.path.join(cache_dir, 'multi_session_summary_dfs_mesoscope')
    if not os.path.exists(mega_mdf_write_dir):
        os.makedirs(mega_mdf_write_dir)

    if len(conditions) == 4:
        filename = 'mean' + type + conditions[1] +'_'+ conditions[2] +'_'+ conditions[3] + suffix + '_df.h5'
    elif len(conditions) == 3:
        filename = 'mean' + type + conditions[1] +'_'+ conditions[2] + suffix + '_df.h5'
    elif len(conditions) == 2:
        filename = 'mean' + type + conditions[1] + suffix + '_df.h5'

    print('saving multi session mean df to ',filename)
    mega_mdf.to_hdf(
        os.path.join(mega_mdf_write_dir, filename),
        key='df',
        format='fixed')
    print('saved')


if __name__ == '__main__':
    cache_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\visual_behavior_production_analysis'

    # VisualBehavior production as of 3/20/19
    # experiment_ids = [775614751, 778644591, 782675436, 783927872, 783928214, 784482326,
    #                   787461073, 787498309, 787501821, 788488596, 788489531, 788490510,
    #                   789359614, 790149413, 790709081, 791119849, 791453282,
    #                   791980891, 792812544, 792813858, 792815735, 792816531, 794378505,
    #                   794381992, 795073741, 795075034, 795076128, 795948257, 795952471,
    #                   795952488, 795953296, 796105304, 796105823, 796106321, 796106850,
    #                   796108483, 796306417, 796308505, 797255551, 798392580, 798403387,
    #                   798404219, 799366517, 799368262, 799368904, 803736273, 805100431,
    #                   805784313, 805784331, 806455766, 806456687, 806989729, 807752719,
    #                   807753318, 807753334, 807753920, 808619543, 808621015, 808621034,
    #                   808621958, 809497730, 809501118, 811456530, 811458048, 813083478,
    #                   814610580, 815097949, 815652334, 817267785, 817267860, 818073631,
    #                   819432482, 819434449, 820307518, 822024770,
    #                   822028017, 822028587, 822641265, 822647116, 822647135, 822656725,
    #                   823392290, 823396897, 823401226, 824333777, 825120601, 825130141,
    #                   825623170, 826583436, 826585773, 826587940, 830093338, 830700781,
    #                   830700800, 831330404, 832117336, 833629926, 833629942, 833631914,
    #                   834275020, 834275038, 834279496, 836258936, 836258957, 836260147,
    #                   836910438, 836911939, 837296345, 837729902, 838849930]


    # Mesoscope production as of 4/25
    experiment_ids = [816242065, 816242073, 816242080, 816242091, 816242095, 816242105, 816242114, 816242121,
                      811911167, 811911171, 811911175, 811911179, 811911182, 811911184, 811911186, 811911188,
                      810623714, 810623716, 810623718, 810623720, 810623722, 810623724, 810623726, 810623728,
                      807432861, 807432867, 807432871, 807432876, 807432883, 807432891, 807432899, 807432903,
                      804835920, 804835922, 804835924, 804835926, 804835928, 804835930, 804835932, 804835934,
                      792694983, 792694987, 792694996, 792695013, 792695018, 792695021, 792695028, 792695031,
                      791748112, 791748114, 791748116, 791748118, 791748122, 791748124, 791748126, 791748128,
                      791262690, 791262693, 791262695, 791262698, 791262701, 791262705, 791262708, 791262710,
                      790261676, 790261687, 790261695, 790261701, 790261711, 790261714, 790261719, 790261723,
                      789989571, 789989573, 789989575, 789989578, 789989582, 789989586, 789989590, 789989594,
                      790002022, 790002024, 790002026, 790002030, 790002034, 790002038, 790002040, 790002044,
                      788325934, 788325938, 788325940, 788325944, 788325946, 788325948, 788325950, 788325953,
                      787282617, 787282625, 787282643, 787282662, 787282676, 787282685, 787282699, 787282708,
                      783477276, 783477281, 783477287, 783477293, 783477300, 783477307, 783477311, 783477325,
                      781340173, 781340175, 781340177, 781340179, 781340181, 781340186, 781340190, 781340192,
                      781354912, 781354915, 781354917, 781354920, 781354922, 781354924, 781354928, 781354932,
                      781316513, 781316515, 781316517, 781316519, 781316521, 781316523, 781316527, 781316532,
                      781304550, 781304553, 781304555, 781304557, 781304561, 781304563, 781304567, 781304571,
                      781304524, 781304528, 781304532, 781304534, 781304536, 781304540, 781304543, 781304547,
                      781304498, 781304500, 781304504, 781304508, 781304512, 781304514, 781304516, 781304520,
                      769724580, 769724588, 769724595, 769724602, 769724608, 769724615, 769724623, 769724630,
                      776865698, 776865704, 776865710, 776865717, 776865720, 776865723, 776865731, 776865738,
                      779319647, 779319650, 779319654, 779319658, 779319666, 779319669, 779319672, 779319675,
                      779798855, 779798862, 779798874, 779798893, 779798900, 779798907, 779798914, 779798923,
                      807353849, 807353859, 807353867, 807353869, 807353871, 807353874, 807353878, 807353880,
                      807310576, 807310578, 807310580, 807310582, 807310584, 807310587, 807310592, 807310594,
                      ]

    get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'image_name'], flashes=True, omitted=False)
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
