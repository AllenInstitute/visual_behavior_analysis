import os
import allensdk.brain_observatory.behavior.behavior_project_cache as bpc
from slurm_deploy import Slurm
import pathlib
import time


def deploy_get_behavior_summary_for_all_sessions():
    # data_folder = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/platform_paper_cache'
    # cache = bpc.VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=data_folder)
    # behavior_session_table = cache.get_behavior_session_table()
    # behavior_session_ids = behavior_session_table.index.values

    import visual_behavior.data_access.utilities as utilities
    behavior_session_ids = utilities.get_behavior_session_ids_to_analyze()


    slurm = Slurm(
        python_script=None, 
        job_name=None, 
        partition='braintv', 
        cpus_per_task=1, 
        memory='20gb',
        time='01:00:00',
        username=None,
    )

    current_location = pathlib.Path(__file__).parent.resolve()
    python_script_to_run = os.path.join(current_location, 'cache_behavior_performance_for_one_session.py')

    methods = ['stimulus_based', 'trial_based', 'sdk']

    for method in methods:
        for behavior_session_id in behavior_session_ids:
            print('deploying job for bsid {}'.format(behavior_session_id))
            slurm.python_script = '{} --behavior-session-id {} --method {}'.format(python_script_to_run, behavior_session_id, method)
            slurm.job_name = 'bsid_{}'.format(behavior_session_id)

            slurm.deploy_job()

            time.sleep(0.05)


if __name__ == "__main__":
    deploy_get_behavior_summary_for_all_sessions()