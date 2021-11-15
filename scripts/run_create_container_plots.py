import os
import sys
import argparse
from visual_behavior.data_access import loading as loading
from visual_behavior.data_access import utilities as utilities

from simple_slurm import Slurm

parser = argparse.ArgumentParser(description='run container plot creation functions on the cluster')
parser.add_argument('--env', type=str, default='visual_behavior_sdk', metavar='name of conda environment to use')
parser.add_argument('--scriptname', type=str, default='create_container_plots.py', metavar='name of script to run (must be in same folder)')


ophys_container_ids = loading.get_ophys_container_ids()


if __name__ == "__main__":
    args = parser.parse_args()
    python_executable = "{}/anaconda2/envs/{}/bin/python".format(os.path.expanduser('~'), args.env)
    python_file = os.path.join(os.getcwd(), args.scriptname)

    # define the job record output folder
    stdout_location = r"/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/cluster_jobs/paper_figures"

    # instantiate a Slurm object
    slurm = Slurm(
        mem='40g',  # '24g'
        cpus_per_task=10,
        time='10:00:00',
        partition='braintv',
        job_name='container_plots',
        output=f'{stdout_location}/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
    )

    for ii, ophys_container_id in enumerate(ophys_container_ids):
        args_to_pass = '--ophys_container_id {}'.format(ophys_container_id)
        print('container ID = {}, number {} of {}'.format(ophys_container_id, ii + 1, len(ophys_container_ids)))
        job_title = 'container_{}'.format(ophys_container_id)

        slurm.sbatch(python_executable + ' ' + python_file + ' ' + args_to_pass)
