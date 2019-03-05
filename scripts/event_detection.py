#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')

import numpy as np

from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset
from visual_behavior.visualization.ophys import summary_figures as sf
from visual_coding_2p_analysis.l0_analysis import L0_analysis


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


if __name__ == '__main__':
    import sys

    lims_id = sys.argv[1]

    cache_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_pilot_analysis'
    events_dir = os.path.join(cache_dir, 'events')
    event_detection(lims_id,cache_dir=cache_dir,events_dir=events_dir)
