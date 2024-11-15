import os
import time
import pathlib
import argparse
from simple_slurm import Slurm


def deploy_get_behavior_summary_for_all_sessions():

    current_location = pathlib.Path(__file__).parent.resolve()
    python_script_to_run = os.path.join(current_location, 'cache_behavior_performance_for_one_session.py')

    python_executable = "{}/anaconda2/envs/{}/bin/python".format(os.path.expanduser('~'), 'visual_behavior_sdk_new')

    # define the job record output folder
    stdout_location = r"/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/cluster_jobs/behavior_metrics"


    slurm = Slurm(
        job_name='cache_performance',
        partition='braintv', 
        cpus_per_task=1, 
        mem='80g',
        time='05:00:00',
        output=f'{stdout_location}/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
    )

    import visual_behavior.data_access.loading as loading
    from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache as bpc

    cache_dir = loading.get_platform_analysis_cache_dir()
    cache = bpc.from_s3_cache(cache_dir=cache_dir)
    behavior_sessions = cache.get_behavior_session_table()
    behavior_session_ids = behavior_sessions.index.values

    # behavior_sessions = loading.get_platform_paper_behavior_session_table(include_4x2_data=True)
    # behavior_session_ids = behavior_sessions.index.values

    # save_dir = loading.get_platform_analysis_cache_dir()
    # problem_sessions = pd.read_csv(os.path.join(save_dir, 'problem_behavior_sessions.csv'))
    # behavior_session_ids = problem_sessions.behavior_session_id.values

    methods = ['sdk', 'response_probability', 'stimulus_based', 'trial_based']
    for method in methods:
        for behavior_session_id in behavior_session_ids:
            print('deploying job for bsid {}'.format(behavior_session_id))
            args_to_pass = '--behavior-session-id {} --method {}'.format(behavior_session_id, method)
            job_title = 'behavior_session_id_{}'.format(behavior_session_id)

            slurm.sbatch(python_executable + ' ' + python_script_to_run + ' ' + args_to_pass)


if __name__ == "__main__":
    deploy_get_behavior_summary_for_all_sessions()