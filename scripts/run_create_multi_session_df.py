import os
import argparse
from simple_slurm import Slurm

import visual_behavior.data_access.loading as loading
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache

# python file to execute on cluster
python_file = r"/home/marinag/visual_behavior_analysis/scripts/create_multi_session_df.py"

# conda environment to use
conda_environment = 'learning_mFISH'

# build the python path
# this assumes that the environments are saved in the user's home directory in a folder called 'anaconda2'
# python_path = os.path.join(
#     os.path.expanduser("~"),
#     'anaconda2',
#     'envs',
#     conda_environment,
#     'bin',
#     'python'
# )

python_executable = "{}/anaconda3/envs/{}/bin/python".format(os.path.expanduser('~'), conda_environment)

# define the job record output folder
stdout_location = r"/allen/programs/mindscope/workgroups/learning/ophys/cluster_jobs/multi_session_dfs"


# cache = VisualBehaviorOphysProjectCache.from_lims()
# experiments_table = cache.get_ophys_experiment_table(passed_only=False)
# experiments_table = experiments_table[experiments_table.project_code.isin(['LearningmFISHTask1A', 'LearningmFISHDevelopment',
#                                                                            'omFISHGad2Meso'])]
# experiments_table = experiments_table[experiments_table.session_type.isin(['STAGE_0', 'STAGE_1',
#                                                                            'OPHYS_7_receptive_field_mapping'])==False]

save_dir = r'/allen/programs/mindscope/workgroups/learning/ophys/learning_project_cache'
import pandas as pd
experiments_table = pd.read_csv(os.path.join(save_dir, 'mFISH_project_expts.csv'))
print(len(experiments_table), 'experiments')

# experiments_table = loading.get_filtered_ophys_experiment_table()
# experiments_table = experiments_table[experiments_table.project_code=='LearningmFISHTask1A']

# call the `sbatch` command to run the jobs.
for mouse_id in experiments_table.mouse_id.unique():
    print('mouse_id:', mouse_id)
    ophys_container_ids = experiments_table[experiments_table.mouse_id==mouse_id].ophys_container_id.unique()
    print('there are', len(ophys_container_ids),'ophys_container_ids for this mouse')
    for ophys_container_id in ophys_container_ids:
        print('ophys_container_id:', ophys_container_id)
        # instantiate a Slurm object
        slurm = Slurm(mem='120g',  # '24g'
                      cpus_per_task=1,
                      time='20:00:00',
                      partition='braintv',
                      job_name='multi_session_df_'+str(mouse_id)+'_'+str(ophys_container_id),
                      output=f'{stdout_location}/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
                      )


        slurm.sbatch(python_executable+' '+python_file+' --mouse_id '+str(mouse_id)+' --ophys_container_id'+' '+str(ophys_container_id))


