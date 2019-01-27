#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')

from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis
from visual_behavior.visualization.ophys.summary_figures import plot_image_response_for_trial_types


if __name__ == '__main__':

    experiment_id = 719996589

    dataset = VisualBehaviorOphysDataset(experiment_id)
    analysis = ResponseAnalysis(dataset)

    print('plotting cell responses')
    for cell in dataset.get_cell_indices():
        plot_image_response_for_trial_types(analysis, cell)
    print('done')
