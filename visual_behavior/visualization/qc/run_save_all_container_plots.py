import os
import sys
import argparse
from visual_behavior.data_access import loading as loading

sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/src/')
from pbstools import pbstools  # NOQA E402


parser = argparse.ArgumentParser(description='run container qc plot generation functions on the cluster')
parser.add_argument('--env', type=str, default='', metavar='name of conda environment to use')
parser.add_argument('--scriptname', type=str, default='save_all_container_plots.py', metavar='name of script to run (must be in same folder)')
parser.add_argument("--plots", type=str, default=None, metavar='plot name to generate')

job_dir = r"/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/cluster_jobs/vba_qc_plots"

job_settings = {'queue': 'braintv',
                'mem': '60g',
                'walltime': '3:00:00',
                'ppn': 1,
                }


container_ids = loading.get_ophys_container_ids()


if __name__ == "__main__":
    args = parser.parse_args()
    # python_executable = "{}/.conda/envs/{}/bin/python".format(os.path.expanduser('~'), args.env)
    python_executable = "{}/anaconda2/envs/{}/bin/python".format(os.path.expanduser('~'), args.env)
    python_file = os.path.join(os.getcwd(), args.scriptname)

    for ii, container_id in enumerate(container_ids):
        if args.plots is None:
            args_to_pass = '--container-id {}'.format(container_id)
        else:
            args_to_pass = '--container-id {} --plots {}'.format(container_id, args.plots)
        print('container ID = {}, number {} of {}'.format(container_id, ii + 1, len(container_ids)))
        job_title = 'container_{}'.format(container_id)
        pbstools.PythonJob(
            python_file,
            python_executable,
            python_args=args_to_pass,
            jobname=job_title,
            jobdir=job_dir,
            **job_settings
        ).run(dryrun=False)
