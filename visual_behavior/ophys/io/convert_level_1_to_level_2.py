# """
# Created on Saturday July 14 2018
# 
# @author: nicholasc
# """

import os
from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysSession
from visual_behavior.ophys.io.filesystem_api import VisualBehaviorFileSystemAPI

def convert_level_1_to_level_2(ophys_experiment_id, cache_dir=None):

    analysis_dir = os.path.join(cache_dir, str(ophys_experiment_id))
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)

    data_set = VisualBehaviorOphysSession(ophys_experiment_id)
    api = VisualBehaviorFileSystemAPI(analysis_dir)
    api.save(data_set)

    return data_set

if __name__ == '__main__':
    convert_level_1_to_level_2(702134928, cache_dir='/allen/aibs/technology/nicholasc/pipeline/visual_behavior')
