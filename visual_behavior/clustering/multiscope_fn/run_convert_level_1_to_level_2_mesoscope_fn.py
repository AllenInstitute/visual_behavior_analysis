import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset
from visual_behavior.visualization.ophys import summary_figures as sf
from visual_behavior.ophys.io.convert_level_1_to_level_2 import convert_level_1_to_level_2, get_analysis_dir, get_lims_data
from visual_behavior.ophys.io.create_analysis_files import create_analysis_files
from ophysextractor.utils import logger, util

cache_dir = r"//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_production_analysis"

#%%
from ophysextractor.utils.util import mongo, get_psql_dict_cursor

def get_stage_mongo(session_id):

    DB = mongo.db.ophys_session_log 
    db_cursor_sess = DB.find({"id":session_id})

   
    ######### get stage from mongo
    a = list(db_cursor_sess[0]['name'])
    ai = (np.argwhere([a[i]=='_' for i in range(len(a))]).flatten()[-1]).astype(int)
    stage_mongo = str(db_cursor_sess[0]['name'][ai+1:])
    
    return stage_mongo


#%%
# dates2analyze = '20190602'
def setVars_convert_level_1_to_level_2_mesoscope_2(dates2analyze=''): 
    # dates2analyze='20190602' # means analyze all days recorded on this date and after.
        
    #%% Get the list of all visBehMult experiments that are in qc or passed
    
    # NOTE: I also added those in "processing" ... becaue I want to follow up with lims about them... 
    # you may want to get them separetly because the convert code will definitely fail for them...
    
    list_of_exp_cursor = util.mongo.db.ophys_experiment_log.find({"$and":[{'project_code': "VisualBehaviorMultiscope"}]}) #,\
#                                                                          {"experiment_obj.status": {"$in":["qc", "passed", "processing"]}}]})
#                                                                          {"experiment_obj.status": "qc"}]}) # MesoscopeDevelopment   {"experiment_obj.status": "qc"}]}
#    list_of_exp_cursor.count()
    
    experiment_ids = []
    experiment_dates = []       
    session_ids = []
    for indiv_exp in list_of_exp_cursor: # these are experiment_ids , not session_ids
        lims_id = indiv_exp['experiment_obj']['id']
        lims_date = indiv_exp['experiment_obj']['acquisition_date']
        experiment_ids.append(lims_id)  
        experiment_dates.append(lims_date.strftime("%Y%m%d"))
        a = indiv_exp['storage_directory'].find('ophys_session'); 
        b = indiv_exp['storage_directory'].find('/ophys_experiment'); 
        session_ids.append(int(indiv_exp['storage_directory'][a+14 : b]))
          
    
    #%% Extract experiments based on their acquision date
    
    if dates2analyze!='':
        experiment_ids = np.array(experiment_ids)[np.array(experiment_dates) > dates2analyze]
        session_ids = np.array(session_ids)[np.array(experiment_dates) > dates2analyze]
        experiment_dates = np.array(experiment_dates)[np.array(experiment_dates) > dates2analyze]

    experiment_ids0 = experiment_ids
    session_ids0 = session_ids
    experiment_dates0 = experiment_dates

    
    #%% Remove ophys7 sessions, because they cannot be analyzed by the convert code!

    print('Removing ophys7 sessions, because they cannot be analyzed by the convert code!')
    stage_mongo_all = [get_stage_mongo(int(session_id)) for session_id in np.unique(session_ids)]
    ophys7s = np.in1d(stage_mongo_all, ['Ophys7', 'ophys7', '7', '7RF', 'test'])
    print('%d sessions are ophys7 and test' %sum(ophys7s))
    
    experiment_ids = experiment_ids[~np.in1d(session_ids0, np.unique(session_ids0)[ophys7s])]
    experiment_dates = experiment_dates[~np.in1d(session_ids0, np.unique(session_ids0)[ophys7s])]
    session_ids = session_ids[~np.in1d(session_ids0, np.unique(session_ids0)[ophys7s])]
    
    
    #%% Print sessions to be assessed for analysis
    
    [u, iu] = np.unique(session_ids, return_index=True)
    print('============================================================== \nAssessing %d sessions, %d experiments\n==============================================================' %(len(u), len(experiment_ids)))
    print('expriments:\n%s' %experiment_ids)
    print('expriment dates:\n%s' %experiment_dates[iu])
    print('sessions:\n%s' %session_ids[iu])
    
    
    #%% Find the experiments without an analysis folder
    
    cnt = -1
    experiment_dates_2analyse = []
    session_ids_2analyse = []
    experiment_ids_2analyse = []
    
    for experiment_id in experiment_ids: # experiment_id = experiment_ids[0]
        experiment_id = int(experiment_id)
        lims_data = get_lims_data(experiment_id) 
        tmp_folder = get_analysis_dir(lims_data, cache_dir=cache_dir, cache_on_lims_data=False)
        cnt = cnt+1
        
        try: 
            if not('dff_traces.h5' in os.listdir(tmp_folder)):
#                print("==============================================================\ndate %s, session %d, experiment %d\n==============================================================" %(experiment_dates[cnt], session_ids[cnt], experiment_id))
                experiment_dates_2analyse.append(experiment_dates[cnt])
                session_ids_2analyse.append(session_ids[cnt])
                experiment_ids_2analyse.append(experiment_id)
                
#                ophys_data = convert_level_1_to_level_2(experiment_id, cache_dir,  plot_roi_validation=False)
                
                # TODO: Need to consolidate event extraction code with its own class
                #lims_data = get_lims_data(lims_id)
                #exp_cach_folder = get_ophys_experiment_dir(lims_data)
                #events_dir = os.path.join(exp_cach_folder, 'events')-
                #event_detection(lims_id,cache_dir=cache_dir,events_dir=events_dir)
           
#            if not('omitted_flash_response_df.h5' in os.listdir(tmp_folder)):
#                create_analysis_files(experiment_id, cache_dir, overwrite_analysis_files=False, turn_off_plotting = True)
        
        except: 
            print("issues with "+ str(experiment_id))
   
    experiment_ids_2analyse = np.array(experiment_ids_2analyse)
    experiment_dates_2analyse = np.array(experiment_dates_2analyse)
    session_ids_2analyse = np.array(session_ids_2analyse)
            
    [u, iu] = np.unique(session_ids_2analyse, return_index=True)
    print('============================================================== \nRunning convert code on %d sessions, %d experiments\n==============================================================' %(len(u), len(experiment_ids_2analyse)))
    print('expriments:\n%s' %experiment_ids_2analyse)
    print('expriment dates:\n%s' %experiment_dates_2analyse[iu])
    print('sessions:\n%s' %session_ids_2analyse[iu])
        
    
    return experiment_ids_2analyse, experiment_dates_2analyse, session_ids_2analyse


#%%
############################################################
############### Here is the part where you actually run the convert code, and it will be a bit time consuming. Upto here we set the experiments to be analyzed! ###############
############################################################

def run_convert_level_1_to_level_2_mesoscope_2(experiment_ids_2analyse):
    #%% Run the convert code if the analysis folder does not exist.   

    cnt = -1
    for experiment_id in experiment_ids_2analyse: # experiment_id = experiment_ids_2analyse[-1]
        cnt = cnt+1
        try: 
            print("==============================================================\ndate %s, session %d, experiment %d (#%d out of %d)\n" %(experiment_dates_2analyse[cnt], session_ids_2analyse[cnt], experiment_id, cnt, len(experiment_ids_2analyse)))
            
            ophys_data = convert_level_1_to_level_2(int(experiment_id), cache_dir,  plot_roi_validation=False)

        except Exception as e:
            print(e)
#             print("issues with "+ str(experiment_id))

    
    return ophys_data



#%%
########################################################################################################################
########################################################################################################################
########################################################################################################################

#%% Call the functions

dates2analyze = '20190326' #'20191201' # dates2analyze='20191125'
# Set experiment_ids to be analyzed
experiment_ids_2analyse, experiment_dates_2analyse, session_ids_2analyse = setVars_convert_level_1_to_level_2_mesoscope_2(dates2analyze) 

# Run the convert code
ophys_data = run_convert_level_1_to_level_2_mesoscope_2(experiment_ids_2analyse)


#%%
'''
07/12: 
Running convert code on 3 sessions, 13 experiments
==============================================================
expriments:
[901665958 901665969 901665971 901665963 901665956 901665961 901665954
 901665967 904352694 904352686 904363930 904363932 904363928]
expriment dates:
['20190708' '20190711' '20190711']
sessions:
[901149889 903621170 903813946]
'''

        
#%% Run the function every 2 minutes
"""
from apscheduler.schedulers.blocking import BlockingScheduler
            
scheduler = BlockingScheduler()
job = scheduler.add_job(run_analysis, 'interval', minutes=2, id='batch_analysis_meso')
scheduler.start()
"""


#%% Run the function above every data at 8pm 

#Check THIS:
'''    
from datetime import datetime, timedelta
from threading import Timer

x = datetime.today()
y = x.replace(day=x.day, hour=20, minute=0, second=0, microsecond=0) + timedelta(days=1)
delta_t = y-x
secs = delta_t.total_seconds()


###%%
t = Timer(secs, run_convert_level_1_to_level_2_mesoscope_2(dates2analyze='20190602'))
t.start()
'''



