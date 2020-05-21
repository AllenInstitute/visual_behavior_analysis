import os
import numpy as np
import pandas as pd
import visual_behavior.visualization.ophys.summary_figures as sf
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis

import argparse

def load_dataset(experiment_id):
    cache_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\visual_behavior_production_analysis'.replace('\\', '/')
    return loading.get_ophys_dataset(experiment_id, cache_dir)


def generate_save_plots(experiment_id, cell_specimen_id):
    dataset = load_dataset(experiment_id)
    analysis = ResponseAnalysis(dataset, overwrite_analysis_files=False, dataframe_format='tidy')
    make_cell_response_summary_plot(analysis, cell_specimen_id, save=False, show=True, errorbar_bootstrap_iterations=1000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate engaged/disengaged response plots')
    parser.add_argument(
        '--experiment_id',
        type=int,
        default=0,
        metavar='ophys_experiment_id'
    )
    parser.add_argument(
        '--cell_id',
        type=int,
        default=0,
        metavar='cell_specimen_id'
    )
    args = parser.parse_args()
    generate_save_plots(args.experiment_id, args.cell_id)
