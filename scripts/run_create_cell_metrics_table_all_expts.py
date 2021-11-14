import os
import argparse
from simple_slurm import Slurm

parser = argparse.ArgumentParser(description='run cell metrics generation functions on the cluster')
parser.add_argument('--env', type=str, default='visual_behavior_sdk', metavar='name of conda environment to use')
parser.add_argument('--scriptname', type=str, default='create_cell_metrics_table_all_expts.py', metavar='name of script to run (must be in same folder)')


if __name__ == "__main__":
    args = parser.parse_args()
    python_executable = "{}/anaconda2/envs/{}/bin/python".format(os.path.expanduser('~'), args.env)
    python_file = os.path.join(os.getcwd(), args.scriptname)

    # define the job record output folder
    stdout_location = r"/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/cluster_jobs/vba_qc_plots"

    # instantiate a Slurm object
    slurm = Slurm(
        mem='120g',  # '24g'
        cpus_per_task=12,
        time='20:00:00',
        partition='braintv',
        job_name='metrics_table',
        output=f'{stdout_location}/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
    )

    slurm.sbatch(python_executable + ' ' + python_file)
