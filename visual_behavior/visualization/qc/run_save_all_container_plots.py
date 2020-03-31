from pbstools import pbstools
import sys
from visual_behavior.visualization.qc import data_loading
import argparse
import os

sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/src/')

parser = argparse.ArgumentParser(description='run container qc plot generation functions on the cluster')
parser.add_argument('--env', type=str, default='', metavar='name of conda environment to use')
parser.add_argument('--scriptname', type=str, default='save_all_container_plots.py', metavar='name of script to run (must be in same folder)')

job_dir = r"/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/cluster_jobs/vba_qc_plots"

job_settings = {'queue': 'braintv',
                'mem': '30g',
                'walltime': '2:00:00',
                'ppn': 1,
                }


container_ids = data_loading.get_filtered_ophys_container_ids()

if __name__ == "__main__":
    args = parser.parse_args()
    python_executable = "{}/.conda/envs/{}/bin/python".format(os.path.expanduser('~'), args.env)
    python_file = os.path.join(os.getcwd(), args.scriptname)

    for container_id in container_ids:

        job_title = 'container_{}'.format(container_id)
        pbstools.PythonJob(
            python_file,
            python_executable,
            python_args='--container-id {}'.format(container_id),
            jobname=job_title,
            jobdir=job_dir,
            **job_settings
        ).run(dryrun=False)
