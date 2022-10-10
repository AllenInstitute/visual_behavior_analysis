import os
import argparse
from simple_slurm import Slurm

import visual_behavior.data_access.loading as loading
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache


parser = argparse.ArgumentParser(description='run container qc plot generation functions on the cluster')
parser.add_argument('--env', type=str, default='learning_mFISH', metavar='name of conda environment to use')
parser.add_argument('--scriptname', type=str, default='save_all_container_plots.py', metavar='name of script to run (must be in same folder)')
parser.add_argument("--plots", type=str, default=None, metavar='plot name to generate')

# job_dir = r"/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/cluster_jobs/vba_qc_plots"
stdout_location = r"/allen/programs/mindscope/workgroups/learning/ophys/cluster_jobs/qc_plots"

# python file to execute on cluster
python_file = r"/home/marinag/visual_behavior_analysis/visual_behavior/visualization/qc/save_all_container_plots.py"

# container_ids = loading.get_ophys_container_ids()

# from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache
#
# cache = VisualBehaviorOphysProjectCache.from_lims()
# experiments_table = cache.get_ophys_experiment_table(passed_only=False)
# experiments = experiments_table[experiments_table.project_code=='LearningmFISHTask1A']
# experiments = experiments[experiments.mouse_id.isin(['603892', '612764', '624942'])]
# experiments = experiments_table[experiments_table.project_code=='LearningmFISHDevelopment']
# experiments = experiments[experiments.mouse_id.isin(['582466'])]

import pandas as pd
save_dir = r'/allen/programs/mindscope/workgroups/learning/ophys/learning_project_cache'
experiments_table = pd.read_csv(os.path.join(save_dir, 'mFISH_project_expts.csv'))
experiments_table = experiments_table.set_index('ophys_experiment_id')

container_ids = experiments.ophys_container_id.unique()

if __name__ == "__main__":
    args = parser.parse_args()
    python_executable = "{}/anaconda2/envs/{}/bin/python".format(os.path.expanduser('~'), args.env)
    python_file = os.path.join(os.getcwd(), args.scriptname)

    for ii, container_id in enumerate(container_ids):
        # instantiate a Slurm object
        slurm = Slurm(
            mem='100g',
            cpus_per_task=1,
            time='120:00:00',
            partition='braintv',
            job_name='container_' + str(container_id),
            output=f'{stdout_location}/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
        )

        slurm.sbatch(python_executable + ' ' + python_file + ' --container-id ' + str(container_id))



