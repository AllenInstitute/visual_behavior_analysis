import sys
import platform
if platform.system() == 'Linux':
    sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/Doug/pbstools')
from pbstools import PythonJob # flake8: noqa: E999


#Recent VB production 3/20/19
experiment_ids = [787461073, 791453299, 796306417, 798392580, 818073631, 819432482,
       820307042, 821011078, 822028587, 822641265, 822647116, 822656725,
       823392290, 823396897, 823401226, 824333777, 825120601, 825130141,
       825623170, 826583436, 826585773, 826587940, 830093338, 830700781,
       830700800, 831330404, 832117336, 833629942, 833631914, 834275038,
       834279496, 836258957, 836260147, 836911939, 837296345, 837729902]


python_file = r"/home/marinag/visual_behavior_analysis/scripts/create_analysis_files.py"

jobdir = '/allen/programs/braintv/workgroups/nc-ophys/Marina/ClusterJobs/JobRecords2'

job_settings = {'queue': 'braintv',
                'mem': '80g',
                'walltime': '8:00:00',
                'ppn': 1,
                'jobdir': jobdir,
                }

for experiment_id in experiment_ids:
    PythonJob(
        python_file,
        python_executable='/home/marinag/anaconda2/envs/visual_behavior_ophys/bin/python',
        python_args=experiment_id,
        conda_env=None,
        jobname='process_{}'.format(experiment_id),
        **job_settings
    ).run(dryrun=False)
