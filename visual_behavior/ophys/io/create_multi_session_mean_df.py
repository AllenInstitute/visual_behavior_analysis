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

    mega_mdf_write_dir = os.path.join(cache_dir, 'multi_session_summary_dfs_mesoscope')
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
    experiment_ids = [816242065, 816242073, 816242080, 816242091,
                      816242095, 816242105, 816242114, 816242121, 816571905, 816571911,
                      816571916, 816571919, 816571926, 816571929, 816571932, 816571934,
                      825516687, 825516689, 825516691, 825516694, 825516696, 825516700,
                      825516702, 825516706, 826334607, 826334610, 826334612, 826334616,
                      826334620, 826334623, 826334625, 826334627, 826794703, 826794707,
                      826794711, 826794714, 826794717, 826794723, 826794726, 826794730,
                      828144740, 828144742, 828144744, 828144746, 828144749, 828144751,
                      828144753, 828144755, 829269094, 829269096, 829269098, 829269100,
                      829269102, 829269104, 829269106, 829269108, 835653625, 835653627,
                      835653629, 835653631, 835653633, 835653635, 835653640, 835653642,
                      838486647, 838486649, 838486651, 838486653, 838486655, 838486657,
                      838486659, 838486661, 839716139, 839716141, 839716143, 839716145,
                      839716147, 839716149, 839716151, 839716153, 839716706, 839716708,
                      839716710, 839716712, 839716714, 839716716, 839716718, 839716720,
                      840460366, 840460368, 840460370, 840460372, 840460376, 840460378,
                      840460380, 840460383, 840717527, 840717529, 840717531, 840717534,
                      840717536, 840717538, 840717540, 840717542, 841624549, 841624552,
                      841624554, 841624556, 841624560, 841624564, 841624569, 841624576,
                      841968436, 841968438, 841968440, 841968442, 841968445, 841968447,
                      841968449, 841968452, 841969456, 841969458, 841969460, 841969462,
                      841969465, 841969467, 841969469, 841969471, 842545433, 842545435,
                      842545437, 842545439, 842545442, 842545444, 842545446, 842545448,
                      842545454, 842545456, 842545458, 842545462, 842545466, 842545468,
                      842545470, 842545472, 843007050, 843007052, 843007054, 843007056,
                      843007058, 843007061, 843007063, 843007065, 843534729, 843534731,
                      843534733, 843534736, 843534740, 843534742, 843534744, 843534746,
                      844420212, 844420214, 844420217, 844420220, 844420222, 844420224,
                      844420226, 844420229, 845070856, 845070858, 845070860, 845070862,
                      845070864, 845070866, 845070868, 845070870, 845777907, 845777909,
                      845777911, 845777913, 845777915, 845777918, 845777920, 845777922,
                      845783018, 845783021, 845783023, 845783025, 845783027, 845783030,
                      845783032, 845783034, 846546326, 846546328, 846546331, 846546333,
                      846546335, 846546337, 846546339, 846546341, 847267616, 847267618,
                      847267620, 847267622, 847267624, 847267626, 847267628, 847267630,
                      848039110, 848039113, 848039115, 848039117, 848039119, 848039121,
                      848039123, 848039125, 848760957, 848760959, 848760961, 848760963,
                      848760965, 848760967, 848760969, 848760971, 848760974, 848760977,
                      848760979, 848760981, 848760983, 848760985, 848760988, 848760990,
                      849233390, 849233392, 849233394, 849233396, 849233398, 849233400,
                      849233402, 849233404, 850517344, 850517346, 850517348, 850517350,
                      850517352, 850517354, 850517356, 850517358, 851085092, 851085095,
                      851085098, 851085100, 851085103, 851085105, 851085107, 851085109,
                      851093283, 851093285, 851093287, 851093289, 851093291, 851093296,
                      851093302, 851093306, 851958793, 851958795, 851958797, 851958800,
                      851958802, 851958805, 851958807, 851958809, 851959317, 851959320,
                      851959322, 851959324, 851959326, 851959329, 851959331, 851959333,
                      852730503, 852730505, 852730508, 852730510, 852730514, 852730516,
                      852730518, 852730520, 853362765, 853362767, 853362769, 853362771,
                      853362773, 853362775, 853362777, 853362780, 853363739, 853363743,
                      853363745, 853363747, 853363749, 853363751, 853363753, 853363756,
                      853988430, 853988435, 853988437, 853988444, 853988446, 853988448,
                      853988450, 853988454, 854759890, 854759894, 854759896, 854759898,
                      854759900, 854759903, 854759905, 854759907, 856123117, 856123119,
                      856123122, 856123124, 856123126, 856123130, 856123132, 856123134,
                      856967230, 856967232, 856967234, 856967237, 856967241, 856967243,
                      856967245, 856967247]
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
