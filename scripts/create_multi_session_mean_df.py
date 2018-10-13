#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')

import pandas as pd
from visual_behavior.ophys.io.create_multi_session_mean_df import get_multi_session_mean_df

if __name__ == '__main__':

    # cache_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_pilot_analysis'
    cache_dir = r'/allen/programs/braintv/workgroups/ophysdev/OPhysCore/Analysis/2018-08 - Behavior Integration test'
    # manifest = pd.read_csv(os.path.join(cache_dir, 'visual_behavior_data_manifest.csv'))
    # experiment_ids = manifest.experiment_id.values
    experiment_ids = [742828015, 743160774, 744540132, 744540163, 744540350, 746257362,
       746425960, 746426152, 746445059, 746445153, 747321339,
       747321353, 750536682, 750536778, 750850855, 750852080, 753089326,
       754056568, 754064001, 754065714, 754068181, 754091013, 755001578,
       755647750, 755647764, 755649776, 755649889, 755650903, 756119038,
       756119052, 756119239, 756812463, 756812477, 757628365, 757945399,
       758319777, 759037388, 759276375, 759283789, 759283945, 759580731,
       759841294, 760095646, 760097857, 760097947, 760098070, 760098185,
       760098363, 760598315, 760603437, 760603451, 760604597, 760604901,
       760604919, 760767168, 760767396, 760767562, 761058425, 761058739,
       761059374, 761059388, 761607566, 761867107, 762166213]

    # problem expts
    # 746425946,

    get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'change_image_name', 'trial_type'])
    get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'change_image_name', 'behavioral_response_type'])

