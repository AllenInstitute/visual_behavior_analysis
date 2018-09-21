import numpy as np

from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset
from visual_behavior.ophys.plotting import summary_figures as sf
from visual_coding_2p_analysis.l0_analysis import L0_analysis

def event_detection(lims_id):
    cache_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_pilot_analysis'
    dataset= VisualBehaviorOphysDataset(lims_id, cache_dir=cache_dir)
    dataset.metadata['genotype'] = 'Ai93'

    events_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_pilot_analysis/events'
    l0 = L0_analysis(dataset, cache_directory=events_dir, genotype='Ai93', halflife_ms=315)

    events = l0.get_events()

    dff_traces_file = l0.dff_file
    f = np.load(dff_traces_file)
    dff_traces = f['dff']

    sf.plot_event_detection(dff_traces, events, dataset.analysis_dir)


if __name__ == '__main__':
        import sys
        lims_id = sys.argv[1]
        event_detection(lims_id)




