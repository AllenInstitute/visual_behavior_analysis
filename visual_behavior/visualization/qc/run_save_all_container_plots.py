import os
import sys
import argparse
from visual_behavior.data_access import loading as loading

from simple_slurm import Slurm

parser = argparse.ArgumentParser(description='run container qc plot generation functions on the cluster')
parser.add_argument('--env', type=str, default='', metavar='name of conda environment to use')
parser.add_argument('--scriptname', type=str, default='save_all_container_plots.py', metavar='name of script to run (must be in same folder)')
parser.add_argument("--plots", type=str, default=None, metavar='plot name to generate')


# container_ids = loading.get_ophys_container_ids(include_failed_data=False, release_data_only=False, add_extra_columns=False,
#                                                 exclude_ai94=False, from_cached_file=False)

# second release candidates
save_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/august_release'
save_path = os.path.join(save_dir, 'second_release_candidates_073021.csv')
import pandas as pd
release_candidates = pd.read_csv(save_path)
container_ids = release_candidates.ophys_container_id.unique()

# #
# # try for MultiscopeSignalNoise project
# from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache as bpc
# cache = bpc.from_lims()
# all_experiments = cache.get_ophys_experiment_table()
# experiments = all_experiments[all_experiments.project_code=='MultiscopeSignalNoise']
# passed_experiments = experiments[(experiments.experiment_workflow_state=='passed')&
#                             (experiments.container_workflow_state=='completed')]
# container_ids = passed_experiments.ophys_container_id.unique()


if __name__ == "__main__":
    args = parser.parse_args()
    # python_executable = "{}/.conda/envs/{}/bin/python".format(os.path.expanduser('~'), args.env)
    python_executable = "{}/anaconda2/envs/{}/bin/python".format(os.path.expanduser('~'), args.env)
    python_file = os.path.join(os.getcwd(), args.scriptname)

    # define the job record output folder
    stdout_location = r"/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/cluster_jobs/vba_qc_plots"

    # instantiate a Slurm object
    slurm = Slurm(
        mem='60g',  # '24g'
        cpus_per_task=10,
        time='10:00:00',
        partition='braintv',
        job_name='container_plots',
        output=f'{stdout_location}/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
    )

    for ii, container_id in enumerate(container_ids):
        if args.plots is None:
            args_to_pass = '--container-id {}'.format(container_id)
        else:
            args_to_pass = '--container-id {} --plots {}'.format(container_id, args.plots)
        print('container ID = {}, number {} of {}'.format(container_id, ii + 1, len(container_ids)))
        job_title = 'container_{}'.format(container_id)

        slurm.sbatch(python_executable+' '+python_file+' '+args_to_pass)
