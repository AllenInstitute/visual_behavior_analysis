import os
import argparse
import sys
import time
import pandas as pd
import numpy as np

from simple_slurm import Slurm
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache


parser = argparse.ArgumentParser(description='deploy computing population sparseness')
parser.add_argument('--env-path', type=str, default='visual_behavior', metavar='path to conda environment to use')


if __name__ == "__main__":
    args = parser.parse_args()
    python_executable = "{}/bin/python".format(args.env_path)
    print('python executable = {}'.format(python_executable))
    python_file = "/allen/programs/braintv/workgroups/nc-ophys/alex.piet/sparse/run_sparse.py"

    stdout_basedir = "/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/sparsity"
    stdout_location = os.path.join(stdout_basedir, 'job_records')
    if not os.path.exists(stdout_location):
        print('making folder {}'.format(stdout_location))
        os.mkdir(stdout_location)
    print('stdout files will be at {}'.format(stdout_location))

    cache_dir = r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/platform_paper_cache'
    cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=cache_dir)
    experiments_table = cache.get_ophys_experiment_table()
    experiments_table = experiments_table[(experiments_table.project_code!="VisualBehaviorMultiscope4areasx2d")&(experiments_table.reporter_line!="Ai94(TITL-GCaMP6s)")].reset_index()
    print('experiments table loaded')

    job_count = 0

    job_string = "--oeid {} "

    experiment_ids = experiments_table['ophys_experiment_id'].values
    n_experiment_ids = len(experiment_ids)

    for experiment_id in experiment_ids:
        job_count += 1
        print('starting cluster job for {}, job count = {}'.format(experiment_id, job_count))
        job_title = 'oeid_{}'.format(experiment_id)
        job_id = Slurm.JOB_ARRAY_ID
        job_array_id = Slurm.JOB_ARRAY_MASTER_ID
        output = stdout_location+"/"+str(job_array_id)+"_"+str(job_id)+"_"+str(experiment_id)+".out"
    
        # instantiate a SLURM object
        slurm = Slurm(
            cpus_per_task=4,
            job_name=job_title,
            time='0:30:00',
            mem='50gb',
            output=output,
            partition="braintv"
        )

        args_string = job_string.format(experiment_id)
        slurm.sbatch('{} {} {}'.format(
                python_executable,
                python_file,
                args_string,
            )
        )
        time.sleep(.05)
