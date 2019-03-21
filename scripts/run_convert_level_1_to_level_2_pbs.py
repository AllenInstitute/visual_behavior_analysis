import sys
import platform
if platform.system() == 'Linux':
    sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/Doug/pbstools')
from pbstools import PythonJob # flake8: noqa: E999


# VisualBehavior production as of 3/20/19
experiment_ids = [775614751, 778644591, 782675436, 783927872, 783928214, 784482326,
                  787461073, 787498309, 787501821, 788488596, 788489531, 788490510,
                  789359614, 790149413, 790709081, 791119849, 791453282, 791453299,
                  791980891, 792812544, 792813858, 792815735, 792816531, 794378505,
                  794381992, 795073741, 795075034, 795076128, 795948257, 795952471,
                  795952488, 795953296, 796105304, 796105823, 796106321, 796106850,
                  796108483, 796306417, 796308505, 797255551, 798392580, 798403387,
                  798404219, 799366517, 799368262, 799368904, 803736273, 805100431,
                  805784313, 805784331, 806455766, 806456687, 806989729, 807752719,
                  807753318, 807753334, 807753920, 808619543, 808621015, 808621034,
                  808621958, 809497730, 809501118, 811456530, 811458048, 813083478,
                  814610580, 815097949, 815652334, 817267785, 817267860, 818073631,
                  819432482, 819434449, 820307042, 820307518, 821011078, 822024770,
                  822028017, 822028587, 822641265, 822647116, 822647135, 822656725,
                  823392290, 823396897, 823401226, 824333777, 825120601, 825130141,
                  825623170, 826583436, 826585773, 826587940, 830093338, 830700781,
                  830700800, 831330404, 832117336, 833629926, 833629942, 833631914,
                  834275020, 834275038, 834279496, 836258936, 836258957, 836260147,
                  836910438, 836911939, 837296345, 837729902, 838849930]


python_file = r"/home/marinag/visual_behavior_analysis/scripts/convert_level_1_to_level_2.py"

jobdir = '/allen/programs/braintv/workgroups/nc-ophys/Marina/ClusterJobs/JobRecords2'

job_settings = {'queue': 'braintv',
                'mem': '30g',
                'walltime': '5:00:00',
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
