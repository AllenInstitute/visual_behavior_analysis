import sys
import platform
if platform.system() == 'Linux':
    sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/Doug/pbstools')
from pbstools import PythonJob # flake8: noqa: E999


#VisualBehavior production as of 3/12/19
experiment_ids = [834275020, 833629942, 833629926, 833631914, 832117336, 831330404,
       830700781, 830700800, 830093338, 826583436, 826587940, 826585773,
       825623170, 825130141, 825120601, 824333777, 823396897, 823401226,
       823392290, 822647135, 822647116, 822641265, 822656725, 822028017,
       822024770, 822028587, 821011078, 820307518, 820307042, 819434449,
       819432482, 818073631, 817267785, 817267860, 815652334, 815097949,
       814610580, 813083478, 811458048, 811456530, 809501118, 809497730,
       808621015, 808621034, 808619543, 808621958, 807753920, 807753334,
       807752719, 807753318, 806989729, 806455766, 806456687, 805784313,
       805784331, 805100431, 803736273, 799368262, 799368904, 799366517,
       798392580, 798404219, 798403387, 797255551, 796306417, 796308505,
       796105823, 796108483, 796106321, 796105304, 796106850, 795948257,
       795953296, 795952488, 795952471, 795073741, 795075034, 795076128,
       794378505, 794381992, 792815735, 792816531, 792812544, 792813858,
       791980891, 791453299, 791453282, 791119849, 790709081, 790149413,
       789359614, 788489531, 788488596, 788490510, 787498309, 787501821,
       784482326, 783927872, 783928214, 782675436, 787461073, 778644591,
       775614751]


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
