import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset
from visual_behavior.visualization.ophys import summary_figures as sf
from visual_behavior.ophys.io.convert_level_1_to_level_2 import convert_level_1_to_level_2, get_analysis_dir, get_lims_data
from visual_behavior.ophys.io.create_analysis_files import create_analysis_files
from ophysextractor.utils import logger, util
from apscheduler.schedulers.blocking import BlockingScheduler

def run_analysis():
       cache_dir = r"//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_production_analysis"

       experiment_ids = []
       list_of_exp_cursor = util.mongo.db.ophys_experiment_log.find({"$and":[{'project_code': "VisualBehaviorMultiscope"},{"experiment_obj.status": "qc"}]}) # MesoscopeDevelopment
       for indiv_exp in list_of_exp_cursor:
              lims_id = indiv_exp['experiment_obj']['id']
              experiment_ids.append(lims_id)

       for experiment_id in experiment_ids:   
              lims_data = get_lims_data(experiment_id) 
              tmp_folder = get_analysis_dir(lims_data, cache_dir=cache_dir, cache_on_lims_data=False)

              try: 
                     if not('dff_traces.h5' in os.listdir(tmp_folder)):
                            ophys_data = convert_level_1_to_level_2(experiment_id, cache_dir,  plot_roi_validation=False)

                            # TODO: Need to consolidate event extraction code with its own class
                            #lims_data = get_lims_data(lims_id)
                            #exp_cach_folder = get_ophys_experiment_dir(lims_data)
                            #events_dir = os.path.join(exp_cach_folder, 'events')-
                            #event_detection(lims_id,cache_dir=cache_dir,events_dir=events_dir)
                     if not('omitted_flash_response_df.h5' in os.listdir(tmp_folder)):
                            create_analysis_files(experiment_id, cache_dir, overwrite_analysis_files=False, turn_off_plotting = True)
              except: 
                     print("issues with "+ str(experiment_id))
       
scheduler = BlockingScheduler()
job = scheduler.add_job(run_analysis, 'interval', minutes=2, id='batch_analysis_meso')
scheduler.start()
