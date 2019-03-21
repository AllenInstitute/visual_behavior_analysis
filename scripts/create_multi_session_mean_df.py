#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')

import pandas as pd
from visual_behavior.ophys.io.create_multi_session_mean_df import get_multi_session_mean_df

if __name__ == '__main__':

    cache_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_production_analysis'
    # manifest = pd.read_csv(os.path.join(cache_dir, 'visual_behavior_data_manifest.csv'))
    # experiment_ids = manifest.experiment_id.values

    # VisualBehavior production as of 3/20/19
    experiment_ids = [775614751, 778644591, 782675436, 783927872, 783928214, 784482326,
                     787461073, 787498309, 787501821, 788488596, 788489531, 788490510,
                     789359614, 790149413, 790709081, 791119849, 791453282,
                     791980891, 792812544, 792813858, 792815735, 792816531, 794378505,
                     794381992, 795073741, 795075034, 795076128, 795948257, 795952471,
                     795952488, 795953296, 796105304, 796105823, 796106321, 796106850,
                     796108483, 796306417, 796308505, 797255551, 798392580, 798403387,
                     798404219, 799366517, 799368262, 799368904, 803736273, 805100431,
                     805784313, 805784331, 806455766, 806456687, 806989729, 807752719,
                     807753318, 807753334, 807753920, 808619543, 808621015, 808621034,
                     808621958, 809497730, 809501118, 811456530, 811458048, 813083478,
                     814610580, 815097949, 815652334, 817267785, 817267860, 818073631,
                     819432482, 819434449, 820307518, 822024770,
                     822028017, 822028587, 822641265, 822647116, 822647135, 822656725,
                     823392290, 823396897, 823401226, 824333777, 825120601, 825130141,
                     825623170, 826583436, 826585773, 826587940, 830093338, 830700781,
                     830700800, 831330404, 832117336, 833629926, 833629942, 833631914,
                     834275020, 834275038, 834279496, 836258936, 836258957, 836260147,
                     836910438, 836911939, 837296345, 837729902, 838849930]

    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                           conditions=['cell_specimen_id', 'image_name'], flashes=True)
    get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'change_image_name', 'trial_type', 'engaged'])
    get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'image_name', 'engaged'], flashes=True)
    get_multi_session_mean_df(experiment_ids, cache_dir,
                                      conditions=['cell_specimen_id', 'image_name', 'engaged', 'repeat'], flashes=True)
    get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'image_name'], flashes=True, omitted=True)
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                           conditions=['cell_specimen_id', 'change_image_name', 'behavioral_response_type'])


    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                               conditions=['cell_specimen_id', 'change_image_name', 'trial_type'], use_events=True)
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                                       conditions=['cell_specimen_id', 'image_name', 'repeat'], flashes=True, use_events=True)
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                                       conditions=['cell_specimen_id', 'image_name', 'engaged', 'repeat'], flashes=True, use_events=True)
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                               conditions=['cell_specimen_id', 'change_image_name', 'behavioral_response_type'], use_events=True)
