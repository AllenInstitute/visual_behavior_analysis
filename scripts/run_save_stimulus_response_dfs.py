import os
import sys
import argparse
from visual_behavior.data_access import loading as loading

from simple_slurm import Slurm

parser = argparse.ArgumentParser(description='run stimulus_response_df generation functions on the cluster')
parser.add_argument('--env', type=str, default='visual_behavior_sdk', metavar='name of conda environment to use')
parser.add_argument('--scriptname', type=str, default='save_stimulus_response_dfs.py', metavar='name of script to run (must be in same folder)')


experiments_table = loading.get_platform_paper_experiment_table()
ophys_experiment_ids = experiments_table.index.values

# import pandas as pd
# import numpy as np
# df = pd.read_csv(os.path.join(loading.get_stimulus_response_df_dir(), 'expts_to_reprocess.csv'))
# ophys_experiment_ids = np.unique(df.ophys_experiment_id.values)


if __name__ == "__main__":
    args = parser.parse_args()
    python_executable = "{}/anaconda2/envs/{}/bin/python".format(os.path.expanduser('~'), args.env)
    python_file = os.path.join(os.getcwd(), args.scriptname)

    # define the job record output folder
    stdout_location = r"/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/cluster_jobs/stimulus_response_dfs"

    # instantiate a Slurm object
    slurm = Slurm(
        mem='60g',  # '24g'
        cpus_per_task=10,
        time='10:00:00',
        partition='braintv',
        job_name='stim_response_dfs',
        output=f'{stdout_location}/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
    )

    for ii, ophys_experiment_id in enumerate(ophys_experiment_ids):
        args_to_pass = '--ophys_experiment_id {}'.format(ophys_experiment_id)
        print('experiment ID = {}, number {} of {}'.format(ophys_experiment_id, ii + 1, len(ophys_experiment_ids)))
        job_title = 'experiment_{}'.format(ophys_experiment_id)

        slurm.sbatch(python_executable + ' ' + python_file + ' ' + args_to_pass)
