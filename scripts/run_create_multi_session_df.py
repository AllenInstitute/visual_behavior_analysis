import os
import sys
import platform
if platform.system() == 'Linux':
    # sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/Doug/pbstools')
    sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/src/pbstools')
from pbstools import PythonJob  # flake8: noqa: E999


import visual_behavior.data_access.loading as loading
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache as bpc

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
stdout_location = r'/allen/programs/braintv/workgroups/nc-ophys/Marina/ClusterJobs/JobRecords'

cache_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/platform_paper_cache'
cache = bpc.from_s3_cache(cache_dir=cache_dir)

experiments_table = cache.get_ophys_experiment_table()

for project_code in experiments_table.project_code.unique():
    print(project_code)
    for session_number in experiments_table.session_number.unique():

        PythonJob(
            python_file,
            python_executable='/home/marinag/anaconda2/envs/visual_behavior_sdk/bin/python',
            python_args=[project_code, session_number],
            conda_env=None,
            jobname='multi_session_df_' + project_code + '_' + str(session_number),
            **job_settings
        ).run(dryrun=False)
