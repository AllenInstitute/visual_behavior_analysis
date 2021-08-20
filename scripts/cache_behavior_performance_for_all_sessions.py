import os
import allensdk.brain_observatory.behavior.behavior_project_cache as bpc
from slurm_deploy import Slurm
import pathlib


def deploy_get_behavior_summary_for_all_sessions():
    data_folder = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/platform_paper_cache'
    cache = bpc.VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=data_folder)

    behavior_session_table = cache.get_behavior_session_table()

    slurm = Slurm(
        python_script=None, 
        job_name=None, 
        partition='braintv', 
        cpus_per_task=1, 
        memory='16gb', 
        time='00:10:00',
        username=None,
    )

    current_location = pathlib.Path(__file__).parent.resolve()
    python_script_to_run = os.path.join(current_location, 'cache_behavior_performance_for_one_session.py')

    for behavior_session_id, row in behavior_session_table.iterrows():
        print('deploying job for bsid {}'.format(behavior_session_id))
        slurm.python_script = '{} --behavior-session-id {}'.format(python_script_to_run, behavior_session_id)
        slurm.job_name = 'bsid_{}'.format(behavior_session_id)

        slurm.deploy_job()

if __name__ == "__main__":
    deploy_get_behavior_summary_for_all_sessions()