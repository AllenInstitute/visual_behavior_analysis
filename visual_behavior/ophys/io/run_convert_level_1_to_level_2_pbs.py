import sys
import platform
if platform.system() == 'Linux':
    sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/Doug/pbstools')
from pbstools import PythonJob

lims_ids = [644942849, 645035903, 645086795, 645362806, 646922970, 647108734,
       647551128, 647887770, 648647430, 649318212, 652844352, 652989428,
       653053906, 653123781, 639253368, 639438856, 639769395, 639932228,
       661423848, 663771245, 663773621, 665286182, 670396087, 671152642,
       672185644, 672584839, 695471168, 696136550, 698724265, 700914412,
       701325132, 702134928, 702723649, 685744008, 686442570, 686726085,
       696537738, 692342909, 692841424, 693272975, 693862238, 712178916,
       712860764, 713525580, 714126693, 715161256, 715887497, 716327871,
       716600289, 715228642, 715887471, 716337289, 716602547, 720001924,
       720793118, 719321260, 719996589, 720379522, 720786478]

python_file = r"/home/marinag/visual_behavior_analysis/visual_behavior/ophys/io/convert_level_1_to_level_2.py"

jobdir = '/allen/programs/braintv/workgroups/nc-ophys/Marina/ClusterJobs/JobRecords'

job_settings = {'queue': 'braintv',
                'mem': '60g',
                'walltime': '32:00:00',
                'ppn': 1,
                'jobdir': jobdir,
                }

for lims_id in lims_ids[-1:]:
    print lims_id
    PythonJob(
        python_file,
        python_executable='/home/marinag/anaconda2/envs/visual_behavior_ophys/bin/python',
        python_args=lims_id,
        conda_env=None,
        jobname='process_{}'.format(lims_id),
        **job_settings
        ).run(dryrun=False)
