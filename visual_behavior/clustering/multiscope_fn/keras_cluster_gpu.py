module load cuda/10.0

import os
import sys
from pbstools import PythonJob
from shutil import copyfile
import datetime

python_file = r"/home/farzaneh.najafi/analysis_codes/multiscope_fn/test_keras_pbs.py" # function to call 


jobdir = os.path.join('/allen/programs/braintv/workgroups/nc-ophys/Farzaneh/ClusterJobs', 'DNN')
job_settings = {'queue': 'braintv',
                'mem': '4g',
                'walltime': '1:00:00',
                'ppn': 4,
                'gpus': 1
                } 

job_settings.update({
                'outfile':os.path.join(jobdir, '$PBS_JOBID.out'),
                'errfile':os.path.join(jobdir, '$PBS_JOBID.err'),
                'email': 'farzaneh.najafi@alleninstitute.org',
                'email_options': 'a'
                })


PythonJob(
    python_file,
#    python_executable = '/home/jeromel/.conda/envs/deep_work_gpu/bin/python',
    python_executable='/home/farzaneh.najafi/anaconda3/envs/tf-gpu/bin/python',
    conda_env = 'None', # 'None' #'deep_work_gpu',
#    jobname = 'movie_2p',
#    python_args = arg_to_pass+' > '+output_terminal,
    **job_settings	
).run(dryrun=False)
