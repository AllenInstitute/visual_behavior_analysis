import os
import sys
import argparse
from visual_behavior.data_access import loading as loading
from visual_behavior.data_access import utilities as utilities

from simple_slurm import Slurm

parser = argparse.ArgumentParser(description='run cell plots generation functions on the cluster')
parser.add_argument('--env', type=str, default='visual_behavior_sdk', metavar='name of conda environment to use')
parser.add_argument('--scriptname', type=str, default='create_single_cell_plots.py', metavar='name of script to run (must be in same folder)')



container_ids = loading.get_ophys_container_ids(platform_paper_only=True, add_extra_columns=True)

if __name__ == "__main__":
    args = parser.parse_args()
    python_executable = "{}/anaconda2/envs/{}/bin/python".format(os.path.expanduser('~'), args.env)
    python_file = os.path.join(os.getcwd(), args.scriptname)

    # define the job record output folder
    stdout_location = r"/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/cluster_jobs/paper_figures"

    # instantiate a Slurm object
    slurm = Slurm(
        mem='60g',  # '24g'
        cpus_per_task=1,
        time='5:00:00',
        partition='braintv',
        job_name='single_cell_plots',
        output=f'{stdout_location}/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
    )

    for ii, container_id in enumerate(container_ids):
        args_to_pass = '--ophys_container_id {}'.format(container_id)
        print('container ID = {}, number {} of {}'.format(container_id, ii + 1, len(container_ids)))
        job_title = 'container_{}'.format(container_id)

        slurm.sbatch(python_executable + ' ' + python_file + ' ' + args_to_pass)
