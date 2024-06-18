import os
import argparse
from simple_slurm import Slurm

import visual_behavior.data_access.loading as loading
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache

# python file to execute on cluster
python_file = r"/home/marinag/visual_behavior_analysis/scripts/convert_level_1_to_level_2.py"

# conda environment to use
conda_environment = 'learning_mfish'

python_executable = "{}/anaconda3/envs/{}/bin/python".format(os.path.expanduser('~'), conda_environment)

# define the job record output folder
stdout_location = r"/allen/programs/mindscope/workgroups/learning/ophys/cluster_jobs/convert"

save_dir = r'/allen/programs/mindscope/workgroups/learning/ophys/learning_project_cache'
# import pandas as pd
# experiments_table = pd.read_csv(os.path.join(save_dir, 'mFISH_project_expts.csv'), index_col=0)
# # limit to non-failed experiments
# experiments_table = experiments_table[(experiments_table.experiment_workflow_state.isin(['passed', 'qc'])) &
#                                       (experiments_table.container_workflow_state != 'failed')]
# print(len(experiments_table), 'experiments')
# print(len(experiments_table.mouse_id.unique()), 'mice')

# experiments_table = loading.get_filtered_ophys_experiment_table()
# experiments_table = experiments_table[experiments_table.project_code=='LearningmFISHTask1A']

# experiment_ids = experiments_table.index.unique()

experiment_ids = [1372472098, 1372472096, 1372159256, 1372159254, 1371522715,
       1371522713, 1371073935, 1371073933, 1370814965, 1370814962,
       1370605604, 1370605601, 1367808840, 1367808838, 1367257322,
       1367257319, 1367058338, 1367058336, 1366815095, 1366815092]

# call the `sbatch` command to run the jobs.
for experiment_id in experiment_ids:
    print('ophys_experiment_id:', experiment_id)

    slurm = Slurm(mem='50g',  # '24g'
                  cpus_per_task=1,
                  time='20:00:00',
                  partition='braintv',
                  job_name='convert'+str(experiment_id), #+'_'+str(ophys_container_id),
                  output=f'{stdout_location}/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
                  )

    # slurm.sbatch( python_executable + ' ' + python_file + ' --mouse_id ' + str(mouse_id) + ' --ophys_container_id' + ' ' + str(ophys_container_id))
    slurm.sbatch(python_executable+' '+python_file+' --ophys_experiment_id '+str(experiment_id))
