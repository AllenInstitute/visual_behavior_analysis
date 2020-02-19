import sys
sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/src/')
from pbstools import pbstools
python_executable = r"/home/marinag/anaconda2/envs/visual_behavior_sdk/bin/python"
python_file = r"/home/marinag/visual_behavior_analysis/visual_behavior/visualization/qc/save_all_container_plots.py"
job_dir = r"/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/cluster_jobs/vba_qc_plots"

job_settings = {'queue': 'braintv',
                'mem': '30g',
                'walltime': '2:00:00',
                'ppn':1,
                }

from visual_behavior.visualization.qc import data_loading
container_ids = data_loading.get_filtered_ophys_container_ids()

if __name__=="__main__":
    for container_id in container_ids:

        job_title = 'container_{}'.format(container_id)
        pbstools.PythonJob(
            python_file,
            python_executable,
            python_args = '--container-id {}'.format(container_id),
            jobname = job_title,
            jobdir = job_dir,
            **job_settings
        ).run(dryrun=False)
