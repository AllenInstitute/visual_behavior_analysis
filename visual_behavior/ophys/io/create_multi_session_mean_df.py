from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis
import visual_behavior.ophys.response_analysis.utilities as ut
import pandas as pd
import logging
import os


logger = logging.getLogger(__name__)


def get_multi_session_mean_df(experiment_ids, cache_dir,
                              conditions=['cell_specimen_id', 'change_image_name', 'behavioral_response_type']):
    mega_mdf = pd.DataFrame()
    for experiment_id in experiment_ids:
        logger.info(experiment_id)
        print(experiment_id)
        dataset = VisualBehaviorOphysDataset(experiment_id, cache_dir=cache_dir)
        if 'trial_response_df.h5' in os.listdir(dataset.analysis_dir):
            analysis = ResponseAnalysis(dataset)
            mdf = ut.get_mean_df(analysis.trial_response_df,
                                 conditions=conditions)
            mdf['experiment_id'] = dataset.experiment_id
            mdf = ut.add_metadata_to_mean_df(mdf, dataset.metadata)

            mega_mdf = pd.concat([mega_mdf, mdf])
        else:
            print('problem for',experiment_id)

    mega_mdf.to_hdf(os.path.join(cache_dir, 'multi_session_summary_dfs', 'mean_' + conditions[2] + '_df.h5'), key='df',
                    format='fixed')
