import os
import sys
import argparse
import pandas as pd
from visual_behavior.data_access import loading as loading
from visual_behavior.data_access import utilities as utilities

from simple_slurm import Slurm

parser = argparse.ArgumentParser(description='run cell plots generation functions on the cluster')
parser.add_argument('--env', type=str, default='learning_mFISH', metavar='name of conda environment to use')
parser.add_argument('--scriptname', type=str, default='create_single_cell_plots.py', metavar='name of script to run (must be in same folder)')


save_dir = r'/allen/programs/mindscope/workgroups/learning/ophys/learning_project_cache'
experiments_table = pd.read_csv(os.path.join(save_dir, 'mFISH_project_expts.csv'))
# experiments_table = experiments_table[experiments_table.project_code=='omFISHGad2Meso']
print(len(experiments_table))
container_ids = experiments_table.ophys_container_id.unique()

# container_ids = loading.get_ophys_container_ids(platform_paper_only=True, add_extra_columns=True)

if __name__ == "__main__":
    args = parser.parse_args()
    python_executable = "{}/anaconda3/envs/{}/bin/python".format(os.path.expanduser('~'), args.env)
    python_file = os.path.join(os.getcwd(), args.scriptname)

    # define the job record output folder
    stdout_location = r"/allen/programs/mindscope/workgroups/learning/ophys/cluster_jobs/single_cell_plots"

    # instantiate a Slurm object
    slurm = Slurm(
        mem='80g',  # '24g'
        cpus_per_task=1,
        time='20:00:00',
        partition='braintv',
        job_name='single_cell_plots',
        output=f'{stdout_location}/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
    )

    for ii, container_id in enumerate(container_ids):
        args_to_pass = '--ophys_container_id {}'.format(container_id)
        print('container ID = {}, number {} of {}'.format(container_id, ii + 1, len(container_ids)))
        job_title = 'container_{}'.format(container_id)

        slurm.sbatch(python_executable + ' ' + python_file + ' ' + args_to_pass)
