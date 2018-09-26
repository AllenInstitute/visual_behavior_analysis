from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis
import visual_behavior.ophys.response_analysis.utilities as ut
import pandas as pd
import matplotlib
import logging
import os

matplotlib.use('Agg')

logger = logging.getLogger(__name__)



def get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'change_image_name', 'behavioral_response_type']):
    mega_mdf = pd.DataFrame()
    for experiment_id in experiment_ids:
        print(experiment_id)
        dataset = VisualBehaviorOphysDataset(experiment_id, cache_dir=cache_dir)
        analysis = ResponseAnalysis(dataset)
        mdf = ut.get_mean_df(analysis.trial_response_df,
                             conditions=conditions)
        mdf['experiment_id'] = dataset.experiment_id
        mdf = ut.add_metadata_to_mean_df(mdf, dataset.metadata)

        mega_mdf = pd.concat([mega_mdf, mdf])
    mega_mdf.to_hdf(os.path.join(cache_dir, 'multi_session_summary_dfs', 'mean_'+conditions[3]+'_df.h5'), key='df',
                    format='fixed')

if __name__ == '__main__':
    import sys

    # cache_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_pilot_analysis'
    cache_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\visual_behavior_pilot_analysis'
    manifest = pd.read_csv(os.path.join(cache_dir, 'visual_behavior_data_manifest.csv'))
    experiment_ids = manifest.experiment_id.values
    get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'change_image_name', 'trial_type'])
