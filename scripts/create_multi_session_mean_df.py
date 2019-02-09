#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')

import pandas as pd
from visual_behavior.ophys.io.create_multi_session_mean_df import get_multi_session_mean_df

if __name__ == '__main__':

    cache_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_pilot_analysis'
    # manifest = pd.read_csv(os.path.join(cache_dir, 'visual_behavior_data_manifest.csv'))
    # experiment_ids = manifest.experiment_id.values

    # VisualBehaviorDevelopment - complete dataset as of 11/15/18
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

    # VisualBehavior production as of 1/15/19
    # experiment_ids = [775614751, 778644591, 787461073, 782675436, 783928214, 783927872,
    #                   787501821, 787498309, 788490510, 788488596, 788489531, 789359614,
    #                   790149413, 790709081, 791119849, 791453282, 791980891, 792813858,
    #                   792812544, 792816531, 792815735, 794381992, 794378505, 795076128,
    #                   795073741, 795952471, 795952488, 795953296, 795948257, 796106850,
    #                   796106321, 796108483, 796105823, 796308505, 797255551, 795075034,
    #                   798403387, 798404219, 799366517, 799368904, 799368262, 803736273,
    #                   805100431, 805784331, 805784313, 806456687, 806455766, 806989729,
    #                   807753318, 807752719, 807753334, 807753920, 796105304, 784482326,
    #                   779335436, 782675457, 791974731, 791979236,
    #                   800034837, 802649986, 806990245, 808621958,
    #                   808619526, 808619543, 808621034, 808621015]


    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                           conditions=['cell_specimen_id', 'image_name'], flashes=True)
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                           conditions=['cell_specimen_id', 'change_image_name', 'trial_type'])
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                                 conditions=['cell_specimen_id', 'image_name', 'repeat'], flashes=True)
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                                   conditions=['cell_specimen_id', 'image_name', 'engaged', 'repeat'], flashes=True)
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                           conditions=['cell_specimen_id', 'change_image_name', 'behavioral_response_type'])
    #
    #
    get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'image_name'], flashes=True, use_events=True)
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                               conditions=['cell_specimen_id', 'change_image_name', 'trial_type'], use_events=True)
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                                       conditions=['cell_specimen_id', 'image_name', 'repeat'], flashes=True, use_events=True)
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                                       conditions=['cell_specimen_id', 'image_name', 'engaged', 'repeat'], flashes=True, use_events=True)
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                               conditions=['cell_specimen_id', 'change_image_name', 'behavioral_response_type'], use_events=True)
