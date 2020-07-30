import sys
import platform
if platform.system() == 'Linux':
    # sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/Doug/pbstools')
    sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/src/pbstools')
from pbstools import PythonJob # flake8: noqa: E999


# pilot study manuscript final expts
experiment_ids = [644942849, 645035903, 645086795, 645362806, 646922970, 647108734,
                   647551128, 647887770, 639253368, 639438856, 639769395, 639932228,
                   661423848, 663771245, 663773621, 665286182, 670396087, 671152642,
                   672185644, 672584839, 695471168, 696136550, 698244621, 698724265,
                   700914412, 701325132, 702134928, 702723649, 692342909, 692841424,
                   693272975, 693862238, 712178916, 712860764, 713525580, 714126693,
                   715161256, 715887497, 716327871, 716600289, 729951441, 730863840,
                   736490031, 737471012, 715228642, 715887471, 716337289, 716602547,
                   720001924, 720793118, 723064523, 723750115, 719321260, 719996589,
                   723748162, 723037901, 731936595, 732911072, 733691636, 736927574,
                   745353761, 745637183, 747248249, 750469573, 754566180, 754943841,
                   756715598, 758274779, 751935154, 752966796, 753931104, 754552635,
                   766779984, 767424894, 768223868, 768865460, 771381093, 772696884,
                   773816712, 774370025, 771427955, 772131949, 772735942, 773843260,
                   768224465, 768871217, 769514560, 770094844, 760696146, 760986090,
                   762214438, 768225217, 774379465, 775011398, 775429615, 776042634,
                   648647430, 649118720, 649318212, 673139359, 673460976]


python_file = r"/home/nick.ponvert/src/visual_behavior_analysis/scripts/generate_event_smoothing_comparison_figures.py"

jobdir = '/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/cluster_jobs/event_smoothing_figures'

job_settings = {'queue': 'braintv',
                'mem': '15g',
                'walltime': '00:30:00',
                'ppn': 1,
                'jobdir': jobdir,
                }

for experiment_id in experiment_ids:
    PythonJob(
        python_file,
        python_executable='/home/nick.ponvert/anaconda3/envs/allen/bin/python',
        python_args=experiment_id,
        conda_env=None,
        jobname='process_{}'.format(experiment_id),
        **job_settings
    ).run(dryrun=False)
