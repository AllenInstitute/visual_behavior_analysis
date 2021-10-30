import os
# import allensdk.brain_observatory.behavior.behavior_project_cache as bpc
from simple_slurm import Slurm
import argparse
import pathlib
import time


def deploy_get_behavior_summary_for_all_sessions():

    current_location = pathlib.Path(__file__).parent.resolve()
    python_script_to_run = os.path.join(current_location, 'cache_behavior_performance_for_one_session.py')

    # define the job record output folder
    stdout_location = r"/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/cluster_jobs/behavior_metrics"


    slurm = Slurm(
        job_name='cache_performance',
        partition='braintv', 
        cpus_per_task=1, 
        mem='20g',
        time='01:00:00',
        output=f'{stdout_location}/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
    )

    import visual_behavior.data_access.utilities as utilities
    behavior_session_ids = utilities.get_behavior_session_ids_to_analyze()

    methods = ['stimulus_based', 'trial_based', 'sdk']

    for method in methods:
        for behavior_session_id in behavior_session_ids:
            print('deploying job for bsid {}'.format(behavior_session_id))
            args_to_pass = '--behavior-session-id {} --method {}'.format(behavior_session_id, method)
            job_title = 'behavior_session_id_{}'.format(behavior_session_id)

            slurm.sbatch(python_executable + ' ' + python_script_to_run + ' ' + args_to_pass)


if __name__ == "__main__":
    deploy_get_behavior_summary_for_all_sessions()