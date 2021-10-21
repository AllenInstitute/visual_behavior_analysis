import os
import argparse
from simple_slurm import Slurm

import visual_behavior.data_access.loading as loading
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache


parser = argparse.ArgumentParser(description='run container qc plot generation functions on the cluster')
parser.add_argument('--env', type=str, default='visual_behavior_sdk', metavar='name of conda environment to use')
parser.add_argument('--scriptname', type=str, default='save_all_container_plots.py', metavar='name of script to run (must be in same folder)')
parser.add_argument("--plots", type=str, default=None, metavar='plot name to generate')

stdout_location = r"/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/cluster_jobs/vba_qc_plots"

# python file to execute on cluster
python_file = r"/home/marinag/visual_behavior_analysis/visual_behavior/visualization/qc/save_all_container_plots.py"

conda_environment = 'visual_behavior_sdk'
#
# build the python path
# this assumes that the environments are saved in the user's home directory in a folder called 'anaconda2'
python_path = os.path.join(
    os.path.expanduser("~"),
    'anaconda2',
    'envs',
    conda_environment,
    'bin',
    'python'
)

# container_ids = loading.get_ophys_container_ids()
cache = VisualBehaviorOphysProjectCache.from_lims()
experiments = cache.get_ophys_experiment_table(passed_only=False)
experiments = experiments[experiments.project_code=='LearningmFISHDevelopment']
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
            time='60:00:00',
            partition='braintv',
            job_name='container_' + str(container_id),
            output=f'{stdout_location}/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
        )

        slurm.sbatch(python_path + ' ' + python_file + ' --container_id ' + str(container_id))



