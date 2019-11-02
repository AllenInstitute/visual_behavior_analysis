#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')

import pandas as pd
from visual_behavior.ophys.io.create_multi_session_mean_df import get_multi_session_mean_df

if __name__ == '__main__':

    cache_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_pilot_analysis/visual_behavior_pilot_manuscript_resubmission'
    # manifest = pd.read_csv(os.path.join(cache_dir, 'visual_behavior_data_manifest.csv'))
    # experiment_ids = manifest.experiment_id.values

    # VisualBehaviorDevelopment - complete dataset as of 11/15/18
    # experiment_ids = [639253368, 639438856, 639769395, 639932228, 644942849, 645035903,
    #             645086795, 645362806, 646922970, 647108734, 647551128, 647887770,
    #             648647430, 649118720, 649318212, 661423848, 663771245, 663773621,
    #             664886336, 665285900, 665286182, 670396087, 671152642, 672185644,
    #             672584839, 673139359, 673460976, 685744008, 686726085, 692342909,
    #             692841424, 693272975, 693862238, 695471168, 696136550, 698244621,
    #             698724265, 700914412, 701325132, 702134928, 702723649, 712178916,
    #             712860764, 713525580, 714126693, 715161256, 715228642, 715887471,
    #             715887497, 716327871, 716337289, 716600289, 716602547, 719321260,
    #             719996589, 720001924, 720793118, 723037901, 723064523, 723748162,
    #             723750115, 729951441, 730863840, 731936595, 732911072, 733691636,
    #             736490031, 736927574, 737471012, 745353761, 745637183, 747248249,
    #             750469573, 751935154, 752966796, 753931104, 754552635, 754566180,
    #             754943841, 756715598, 758274779, 760003838, 760400119, 760696146,
    #             760986090, 761861597, 762214438, 762214650, 766779984, 767424894,
    #             768223868, 768224465, 768225217, 768865460, 768871217, 769514560,
    #             770094844, 771381093, 771427955, 772131949, 772696884, 772735942,
    #             773816712, 773843260, 774370025, 774379465, 775011398, 775429615,
    #             776042634]

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

    #
    get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'image_name'], flashes=True, omitted=True, get_reliability=True)
    get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'image_name'], flashes=True, get_reliability=False)
    get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'change_image_name', 'trial_type'], get_reliability=False)
    get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'image_name', 'repeat'], flashes=True, get_reliability=True)
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                           conditions=['cell_specimen_id', 'image_name', 'engaged', 'repeat'], flashes=True,
    #                           get_reliability=True)
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                           conditions=['cell_specimen_id', 'change_image_name', 'trial_type', 'engaged'], get_reliability=True)
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                           conditions=['cell_specimen_id', 'change_image_name', 'behavioral_response_type'], get_reliability=False)
    #

    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                           conditions=['cell_specimen_id', 'image_name'], flashes=True, omitted=True, use_events=True)
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                           conditions=['cell_specimen_id', 'image_name', 'engaged'], flashes=True, use_events=True)
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                               conditions=['cell_specimen_id', 'change_image_name', 'trial_type', 'engaged'], use_events=True, get_reliability=True)
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                                       conditions=['cell_specimen_id', 'image_name', 'repeat'], flashes=True, use_events=True, get_reliability=True)
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                                       conditions=['cell_specimen_id', 'image_name', 'engaged', 'repeat'], flashes=True, use_events=True, get_reliability=True)
    # get_multi_session_mean_df(experiment_ids, cache_dir,
    #                               conditions=['cell_specimen_id', 'change_image_name', 'behavioral_response_type'], use_events=True)
