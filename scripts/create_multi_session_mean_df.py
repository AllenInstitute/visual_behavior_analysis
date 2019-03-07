#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')

import pandas as pd
from visual_behavior.ophys.io.create_multi_session_mean_df import get_multi_session_mean_df

if __name__ == '__main__':

    cache_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_production_analysis'
    # manifest = pd.read_csv(os.path.join(cache_dir, 'visual_behavior_data_manifest.csv'))
    # experiment_ids = manifest.experiment_id.values

    # VisualBehavior production as of 2/4/19
    experiment_ids = [775614751, 778644591, 787461073, 788490510, 792812544,
       802649986, 794378505, 795075034, 795952488, 796106321, 798403387,
       788488596, 790149413, 791453282, 791980891, 792815735, 795073741,
       795953296, 796108483, 796308505, 798404219, 783928214, 787501821,
       787498309, 790709081, 791119849, 792816531, 792813858, 794381992,
       795076128, 795952471, 796105304, 797255551, 782675436, 783927872,
       784482326, 788489531, 789359614, 795948257, 799368904, 796106850,
       796105823, 799368262, 803736273, 805784331,
       807753318, 808621958, 809497730, 808619526,
       799366517, 805100431, 805784313, 807753920,
       808621015, 806456687, 807752719, 808619543, 811456530, 813083478,
       806455766, 806989729, 807753334, 808621034, 809501118, 811458048]


    get_multi_session_mean_df(experiment_ids, cache_dir,
                                    conditions=['cell_specimen_id', 'image_name', 'repeat'], flashes=True)
    get_multi_session_mean_df(experiment_ids, cache_dir,
                                      conditions=['cell_specimen_id', 'image_name', 'engaged', 'repeat'], flashes=True)
    get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'change_image_name', 'trial_type'])
    get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'change_image_name', 'behavioral_response_type'])


    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                               conditions=['cell_specimen_id', 'change_image_name', 'trial_type'], use_events=True)
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                                       conditions=['cell_specimen_id', 'image_name', 'repeat'], flashes=True, use_events=True)
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                                       conditions=['cell_specimen_id', 'image_name', 'engaged', 'repeat'], flashes=True, use_events=True)
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                               conditions=['cell_specimen_id', 'change_image_name', 'behavioral_response_type'], use_events=True)
