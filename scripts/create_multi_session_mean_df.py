#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')

import pandas as pd
from visual_behavior.ophys.io.create_multi_session_mean_df import get_multi_session_mean_df

if __name__ == '__main__':

    cache_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_pilot_analysis'
    # cache_dir = r'/allen/programs/braintv/workgroups/ophysdev/OPhysCore/Analysis/2018-08 - Behavior Integration test'
    # manifest = pd.read_csv(os.path.join(cache_dir, 'visual_behavior_data_manifest.csv'))
    # experiment_ids = manifest.experiment_id.values
    experiment_ids = [644942849, 645035903, 645086795, 645362806, 646922970, 647108734,
       647551128, 647887770, 648647430, 649118720, 649318212, 639253368,
       639438856, 639769395, 639932228, 661423848, 663771245, 663773621,
       665286182, 670396087, 671152642, 672185644, 672584839, 695471168,
       696136550, 698244621, 698724265, 700914412, 701325132, 702134928,
       702723649, 692342909, 692841424, 693272975, 693862238, 712178916,
       712860764, 713525580, 714126693, 715161256, 715887497, 716327871,
       716600289, 715228642, 715887471, 716337289, 716602547, 720001924,
       720793118, 723064523, 723750115, 719321260, 719996589, 723748162,
       723037901, 729951441, 730863840, 736490031, 737471012, 731936595,
       732911072, 733691636, 736927574, 745353761, 745637183, 747248249,
       750469573, 751935154, 752966796, 753931104, 754552635, 754566180,
       754943841, 756715598, 758274779, 760696146, 760986090, 762214438,
       768225217, 768224465, 768871217, 769514560, 770094844, 766779984,
       767424894, 768223868, 768865460]

    get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'change_image_name', 'trial_type'])
    get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'change_image_name', 'behavioral_response_type'])

