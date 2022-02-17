import os
import sys
import argparse
from visual_behavior.data_access import loading as loading
from visual_behavior.data_access import utilities as utilities

from simple_slurm import Slurm

parser = argparse.ArgumentParser(description='run cell metrics generation functions on the cluster')
parser.add_argument('--env', type=str, default='visual_behavior_sdk', metavar='name of conda environment to use')
parser.add_argument('--scriptname', type=str, default='create_experiment_plots.py', metavar='name of script to run (must be in same folder)')


# from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache as bpc
# cache_dir = loading.get_platform_analysis_cache_dir()
# cache = bpc.from_s3_cache(cache_dir=cache_dir)
# experiments_table = cache.get_ophys_experiment_table()
# print(cache_dir)

# experiments_table = loading.get_platform_paper_experiment_table()
# experiments_table = utilities.limit_to_last_familiar_second_novel_active(experiments_table)
# ophys_experiment_ids = experiments_table.index.values

from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache
cache = VisualBehaviorOphysProjectCache.from_lims()
experiments_table = cache.get_ophys_experiment_table(passed_only=False)
ophys_experiment_ids = experiments_table[experiments_table.project_code=='LearningmFISHTask1A'].index.values


if __name__ == "__main__":
    args = parser.parse_args()
    python_executable = "{}/anaconda2/envs/{}/bin/python".format(os.path.expanduser('~'), args.env)
    python_file = os.path.join(os.getcwd(), args.scriptname)

    # define the job record output folder
    stdout_location = r"/allen/programs/mindscope/workgroups/learning/ophys/cluster_jobs/qc_plots"

    # instantiate a Slurm object
    slurm = Slurm(
        mem='40g',  # '24g'
        cpus_per_task=1,
        time='5:00:00',
        partition='braintv',
        job_name='experiment_plots',
        output=f'{stdout_location}/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
    )

    for ii, ophys_experiment_id in enumerate(ophys_experiment_ids):
        args_to_pass = '--ophys_experiment_id {}'.format(ophys_experiment_id)
        print('experiment ID = {}, number {} of {}'.format(ophys_experiment_id, ii + 1, len(ophys_experiment_ids)))
        job_title = 'experiment_{}'.format(ophys_experiment_id)

        slurm.sbatch(python_executable + ' ' + python_file + ' ' + args_to_pass)
