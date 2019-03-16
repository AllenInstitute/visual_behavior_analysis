import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset
from visual_behavior.visualization.ophys import summary_figures as sf
from visual_behavior.ophys.io.convert_level_1_to_level_2 import convert_level_1_to_level_2, get_analysis_dir, get_lims_data
from visual_behavior.ophys.io.create_analysis_files import create_analysis_files
from ophysextractor.utils import logger, util

cache_dir = r"\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\visual_behavior_production_analysis"

#VisualBehavior Mesoscope production as of 3/15/19
""" experiment_ids = [
       787282617, 787282625, 787282643, 787282662, 787282676, 787282685, 787282699, 787282708,
       788325934, 788325938, 788325940, 788325944, 788325946, 788325948, 788325950, 788325953,
       790002022, 790002024, 790002026, 790002030, 790002034, 790002038, 790002040, 790002044,
       789989571, 789989573, 789989575, 789989578, 789989582, 789989586, 789989590, 789989594,
       790261676, 790261687, 790261695, 790261701, 790261711, 790261714, 790261719, 790261723,
       791262690, 791262693, 791262695, 791262698, 791262701, 791262705, 791262708, 791262710,
       791748112, 791748114, 791748116, 791748118, 791748122, 791748124, 791748126, 791748128,
       792694983, 792694987, 792694996, 792695013, 792695018, 792695021, 792695028, 792695031
       ] """
 
experiment_ids = []
list_of_exp_cursor = util.mongo.db.ophys_experiment_log.find({"$and":[{'project_code': "VisualBehavior"},{"experiment_obj.status": "passed"}]})
for indiv_exp in list_of_exp_cursor:
    lims_id = indiv_exp['experiment_obj']['id']
    experiment_ids.append(lims_id)
    
for experiment_id in experiment_ids:   
    lims_data = get_lims_data(experiment_id) 
    tmp_folder = get_analysis_dir(lims_data, cache_dir=cache_dir, cache_on_lims_data=False)

    if  len(os.listdir(tmp_folder))<5:
        try:
            ophys_data = convert_level_1_to_level_2(experiment_id, cache_dir)

            # TODO: Need to consolidate event extraction code with its own class
            #lims_data = get_lims_data(lims_id)
            #exp_cach_folder = get_ophys_experiment_dir(lims_data)
            #events_dir = os.path.join(exp_cach_folder, 'events')
            #event_detection(lims_id,cache_dir=cache_dir,events_dir=events_dir)

            create_analysis_files(experiment_id, cache_dir, overwrite_analysis_files=False)
        except:
            print("issues with "+str(experiment_id))