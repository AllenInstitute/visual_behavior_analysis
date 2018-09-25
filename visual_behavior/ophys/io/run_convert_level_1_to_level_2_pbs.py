import sys
import platform
if platform.system() == 'Linux':
    sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/Doug/pbstools')
from pbstools import PythonJob # flake8: noqa: E999

lims_ids = [644942849, 645035903, 645086795, 645362806, 646922970, 647108734,
            647551128, 647887770, 648647430, 649118720, 649318212, 639253368,
            639438856, 639769395, 639932228, 661423848, 663771245, 663773621,
            665286182, 673139359, 673460976, 670396087, 671152642, 672185644,
            672584839, 695471168, 696136550, 698244621, 698724265, 700914412,
            701325132, 702134928, 702723649, 692342909, 692841424, 693272975,
            693862238, 712178916, 712860764, 713525580, 714126693, 715161256,
            715887497, 716327871, 716600289, 715228642, 715887471, 716337289,
            716602547, 720001924, 720793118, 723064523, 723750115, 719321260,
            719996589, 723748162, 723037901, 729951441, 730863840, 736490031,
            737471012, 731936595, 732911072, 733691636, 736927574, 745353761,
            745637183, 747248249, 750469573, 751935154, 752966796, 753931104,
            754552635]
#
# lims_ids = [729951441, 730863840, 736490031, 737471012,
#             731936595, 732911072, 733691636, 736927574,
#             745353761, 745637183, 747248249, 750469573,
#             ]

python_file = r"/home/marinag/visual_behavior_analysis/visual_behavior/ophys/io/convert_level_1_to_level_2.py"

jobdir = '/allen/programs/braintv/workgroups/nc-ophys/Marina/ClusterJobs/JobRecords2'

job_settings = {'queue': 'braintv',
                'mem': '60g',
                'walltime': '32:00:00',
                'ppn': 1,
                'jobdir': jobdir,
                }

for lims_id in lims_ids:
    print(lims_id)
    PythonJob(
        python_file,
        python_executable='/home/marinag/anaconda2/envs/visual_behavior_ophys/bin/python',
        python_args=lims_id,
        conda_env=None,
        jobname='process_{}'.format(lims_id),
        **job_settings
    ).run(dryrun=False)
