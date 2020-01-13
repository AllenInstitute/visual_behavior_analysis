#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')

import pandas as pd
from visual_behavior.ophys.io.create_multi_session_mean_df import get_multi_session_mean_df

if __name__ == '__main__':

    cache_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_production_analysis'
    # manifest = pd.read_csv(os.path.join(cache_dir, 'visual_behavior_data_manifest.csv'))
    # experiment_ids = manifest.experiment_id.values

    #VisualBehavior production as of 7/23/19
    experiment_ids = [775614751, 778644591, 783927872, 783928214, 784482326, 787498309,
                     787501821, 788488596, 788489531, 788490510, 789359614, 790149413,
                     790709081, 791119849, 791453282, 791453299, 792812544, 792813858,
                     792815735, 792816531, 794378505, 794381992, 795073741, 795075034,
                     795076128, 795948257, 795952471, 795952488, 795953296, 796105304,
                     796105823, 796106321, 796106850, 796108483, 796306417, 796308505,
                     797255551, 798403387, 798404219, 799366517, 799368262, 799368904,
                     803736273, 805100431, 805784313, 805784331, 806455766, 806456687,
                     806989729, 807752719, 807753318, 807753334, 807753920, 808619543,
                     808621015, 808621034, 808621958, 809497730, 809501118, 811456530,
                     811458048, 813083478, 815652334, 817267785, 817267860, 820307042,
                     820307518, 821011078, 822024770, 822028017, 822641265, 822647116,
                     822647135, 822656725, 823392290, 823396897, 825120601, 825130141,
                     825623170, 826583436, 826585773, 826587940, 830093338, 830700781,
                     830700800, 831330404, 833629926, 833631914, 834279496, 836258936,
                     836258957, 836911939, 837296345, 837729902, 838849930, 840702910,
                     841948542, 842973730, 842975542, 843519218, 843520488, 844395446,
                     845037476, 846487947, 846490568, 847125577, 847241639, 848006184,
                     848692970, 848694045, 848694639, 848697604, 848697625, 848698709,
                     849199228, 849203565, 849203586, 850479305, 850489605, 851056106,
                     851060467, 851932055, 852691524, 853328115, 853962951, 853962969,
                     854703305, 855577488, 855582961, 855582981, 856096766, 859147033,
                     862023618, 862848066, 862848084, 863735602, 864370674, 866463736,
                     867337243, 868900909, 868911434, 869969393, 869969410, 869972431,
                     871159631, 872436745, 873968820, 873972085, 875045489, 877018118,
                     877022592, 877696762, 877697554, 878358326, 878363070, 878363088,
                     879331157, 879332693, 880374622, 880375092, 880961028, 882520593,
                     882935355, 884218326, 884221469, 885067826, 885067844, 885933191,
                     888666698, 888666715, 889771676, 889772922, 889775742, 889777243,
                     891052180, 891054695, 891067673, 891994418, 891996193, 892799212,
                     892805315, 893831526, 893832486, 894726001, 894726047, 894727297,
                     896160394, 898747791]

    platform = 'scientifica'

    # get_multi_session_mean_df(experiment_ids, cache_dir, platform=platform,
    #                           conditions=['cell_specimen_id', 'image_name', ], flashes=True)
    # get_multi_session_mean_df(experiment_ids, cache_dir, platform=platform,
    #                           conditions=['cell_specimen_id', 'image_name', ], flashes=True, omitted=True)
    # get_multi_session_mean_df(experiment_ids, cache_dir, platform=platform,
    #                           conditions=['cell_specimen_id', 'change_image_name', 'trial_type'])
    get_multi_session_mean_df(experiment_ids, cache_dir, platform=platform,
                                      conditions=['cell_specimen_id', 'image_name', 'repeat'], flashes=True)

    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                               conditions=['cell_specimen_id', 'change_image_name', 'trial_type'], use_events=True)
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                                       conditions=['cell_specimen_id', 'image_name', 'repeat'], flashes=True, use_events=True)
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                                       conditions=['cell_specimen_id', 'image_name', 'engaged', 'repeat'], flashes=True, use_events=True)
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                               conditions=['cell_specimen_id', 'change_image_name', 'behavioral_response_type'], use_events=True)
