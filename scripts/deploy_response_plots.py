import os
import argparse
import sys
import time
import pandas as pd
import numpy as  np
from visual_behavior import database as db
import visual_behavior.data_access.loading as loading
sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/src/')
from pbstools import pbstools  # NOQA E402

parser = argparse.ArgumentParser(description='run sdk validation')
parser.add_argument('--env', type=str, default='visual_behavior', metavar='name of conda environment to use')

job_dir = r"/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/cluster_jobs/sdk_validation"

job_settings = {'queue': 'braintv',
                'mem': '100g',
                'walltime': '3:00:00',
                'ppn': 1,
                }

def load_flashwise_summary(behavior_session_id=None):
    conn = db.Database('visual_behavior_data')
    collection = conn['behavior_analysis']['annotated_stimulus_presentations']

    if behavior_session_id is None:
        # load all
        df = pd.DataFrame(list(collection.find({})))
    else:
        # load data from one behavior session
        df = pd.DataFrame(list(collection.find({'behavior_session_id': int(behavior_session_id)})))

    conn.close()

    return df.sort_values(by=['behavior_session_id', 'flash_index'])

if __name__ == "__main__":
    args = parser.parse_args()
    python_executable = "{}/.conda/envs/{}/bin/python".format(os.path.expanduser('~'), args.env)
    print('python executable = {}'.format(python_executable))
    python_file = "{}/code/visual_behavior_analysis/scripts/generate_engaged_disengaged_response_plots.py".format(os.path.expanduser('~'))
    
    flash_summary = load_flashwise_summary()

    cache = loading.get_visual_behavior_cache()
    experiments_table = loading.get_filtered_ophys_experiment_table()
    experiments_table['in_flash_summary'] = experiments_table['behavior_session_id'].map(lambda bsid: bsid in list(flash_summary['behavior_session_id'].unique()))

    experiment_ids = experiments_table.reset_index().query('in_flash_summary == True')['ophys_experiment_id'].unique()

    for ii, experiment_id in enumerate(experiment_ids):

        print('starting cluster job for {}'.format(experiment_id))
        job_title = 'response_plots_oeid_{}'.format(experiment_id)
        pbstools.PythonJob(
            python_file,
            python_executable,
            python_args="--oeid {}".format(experiment_id),
            jobname=job_title,
            jobdir=job_dir,
            **job_settings
        ).run(dryrun=False)
        time.sleep(0.001)