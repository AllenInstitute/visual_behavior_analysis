import matplotlib
matplotlib.use('Agg')
import numpy as np
from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset
from visual_behavior.visualization.ophys import summary_figures as sf
# from visual_coding_2p_analysis.l0_analysis import L0_analysis
from visual_behavior.ophys.io.convert_level_1_to_level_2 import convert_level_1_to_level_2, get_ophys_experiment_dir, get_lims_data
from visual_behavior.ophys.io.create_analysis_files import create_analysis_files

cache_dir = r"\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\visual_behavior_production_analysis"

#VisualBehavior Mesoscope production as of 3/15/19
experiment_ids = [
       787282617, 787282625, 787282643, 787282662, 787282676, 787282685, 787282699, 787282708,
       788325934, 788325938, 788325940, 788325944, 788325946, 788325948, 788325950, 788325953,
       790002022, 790002024, 790002026, 790002030, 790002034, 790002038, 790002040, 790002044,
       789989571, 789989573, 789989575, 789989578, 789989582, 789989586, 789989590, 789989594,
       790261676, 790261687, 790261695, 790261701, 790261711, 790261714, 790261719, 790261723,
       791262690, 791262693, 791262695, 791262698, 791262701, 791262705, 791262708, 791262710,
       791748112, 791748114, 791748116, 791748118, 791748122, 791748124, 791748126, 791748128,
       792694983, 792694987, 792694996, 792695013, 792695018, 792695021, 792695028, 792695031
       ]

def event_detection(lims_id, cache_dir, events_dir, plot=True):
    dataset = VisualBehaviorOphysDataset(lims_id, cache_dir=cache_dir)
    if 'Slc17a7' in dataset.metadata.cre_line.values[0]:
        print('Slc17a7, using Ai93 halflife')
        genotype = 'Ai93'
        dataset.metadata['genotype'] = genotype
        halflife = 314
    elif 'Vip' in dataset.metadata.cre_line.values[0]:
        print('Vip, using Ai93 halflife')
        genotype = 'Ai94'
        dataset.metadata['genotype'] = genotype
        halflife = 649

    l0 = L0_analysis(dataset, cache_directory=events_dir, genotype=genotype, halflife_ms=halflife)

    print('getting events')
    events = l0.get_events()
    print('done getting events')

    dff_traces_file = l0.dff_file
    f = np.load(dff_traces_file)
    dff_traces = f['dff']

    if plot:
        sf.plot_event_detection(dff_traces, events, dataset.analysis_dir)

for experiment_id in experiment_ids:
    ophys_data = convert_level_1_to_level_2(experiment_id, cache_dir)

    # TODO: Need to consolidate event extraction code with its own class
    #lims_data = get_lims_data(lims_id)
    #exp_cach_folder = get_ophys_experiment_dir(lims_data)
    #events_dir = os.path.join(exp_cach_folder, 'events')
    #event_detection(lims_id,cache_dir=cache_dir,events_dir=events_dir)

    create_analysis_files(experiment_id, cache_dir, overwrite_analysis_files=False)

