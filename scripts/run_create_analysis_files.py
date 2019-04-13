import sys
import platform
if platform.system() == 'Linux':
    sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/Doug/pbstools')
from pbstools import PythonJob # flake8: noqa: E999

# #VisualBehaviorDevelopment - complete dataset as of 11/15/18
experiment_ids = [639253368, 639438856, 639769395, 639932228, 644942849, 645035903,
       645086795, 645362806, 646922970, 647108734, 647551128, 647887770,
       648647430, 649118720, 649318212, 661423848, 663771245, 663773621,
       664886336, 665285900, 665286182, 670396087, 671152642, 672185644,
       672584839, 673139359, 673460976, 685744008, 686726085, 692342909,
       692841424, 693272975, 693862238, 695471168, 696136550, 698244621,
       698724265, 700914412, 701325132, 702134928, 702723649, 712178916,
       712860764, 713525580, 714126693, 715161256, 715228642, 715887471,
       715887497, 716327871, 716337289, 716600289, 716602547, 719321260,
       719996589, 720001924, 720793118, 723037901, 723064523, 723748162,
       723750115, 729951441, 730863840, 731936595, 732911072, 733691636,
       736490031, 736927574, 737471012, 745353761, 745637183, 747248249,
       750469573, 751935154, 752966796, 753931104, 754552635, 754566180,
       754943841, 756715598, 758274779, 760003838, 760400119, 760696146,
       760986090, 761861597, 762214438, 762214650, 766779984, 767424894,
       768223868, 768224465, 768225217, 768865460, 768871217, 769514560,
       770094844, 771381093, 771427955, 772131949, 772696884, 772735942,
       773816712, 773843260, 774370025, 774379465, 775011398, 775429615,
       776042634]


python_file = r"/home/marinag/visual_behavior_analysis/scripts/create_analysis_files.py"

jobdir = '/allen/programs/braintv/workgroups/nc-ophys/Marina/ClusterJobs/JobRecords2'

job_settings = {'queue': 'braintv',
                'mem': '80g',
                'walltime': '10:00:00',
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
