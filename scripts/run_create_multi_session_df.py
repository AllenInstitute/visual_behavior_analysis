import os
from simple_slurm import Slurm

import visual_behavior.data_access.loading as loading
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache as bpc

# python file to execute on cluster
python_file = r"/home/marinag/visual_behavior_analysis/scripts/create_multi_session_df.py"

# conda environment to use
conda_environment = 'visual_behavior_sdk'

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

# define the job record output folder
stdout_location = r"/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/cluster_jobs/multi_session_dfs"

cache_dir = loading.get_platform_analysis_cache_dir()
cache = bpc.from_s3_cache(cache_dir=cache_dir)
print(cache_dir)

experiments_table = cache.get_ophys_experiment_table()

# call the `sbatch` command to run the jobs.
for project_code in experiments_table.project_code.unique():
    print(project_code)
    for session_number in experiments_table.session_number.unique():

        # instantiate a Slurm object
        slurm = Slurm(
            mem='200g',  # '24g'
            cpus_per_task=1,
            time='120:00:00',
            partition='braintv',
            job_name='multi_session_df_'+project_code+'_'+str(session_number),
            output=f'{stdout_location}/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
        )

        slurm.sbatch(python_path+' '+python_file+' --project_code '+str(project_code)+' --session_number'+' '+str(session_number))


