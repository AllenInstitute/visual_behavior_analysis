# """
# Created on Saturday July 14 2018
# 
# @author: nicholasc
# """

import os
from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysSession
from visual_behavior.ophys.io.filesystem_api import VisualBehaviorFileSystemAPI
from visual_behavior.ophys.io.lims_api import VisualBehaviorLimsAPI_hackEvents

def convert_level_1_to_level_2(ophys_experiment_id, cache_dir=None, api=None):
    

    data_set = VisualBehaviorOphysSession(ophys_experiment_id, api=api)

    if cache_dir is not None:
        analysis_dir = os.path.join(cache_dir, str(ophys_experiment_id))
        if not os.path.exists(analysis_dir):
            os.mkdir(analysis_dir)
        VisualBehaviorFileSystemAPI(analysis_dir).save(data_set)

    return data_set

if __name__ == '__main__':
    event_cache_dir = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_pilot_analysis/events'
    api = VisualBehaviorLimsAPI_hackEvents(event_cache_dir = event_cache_dir)
    convert_level_1_to_level_2(702134928, cache_dir='/allen/aibs/technology/nicholasc/pipeline/visual_behavior')
