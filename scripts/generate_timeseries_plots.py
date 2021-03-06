#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import visual_behavior.data_access.loading as loading
from visual_behavior.visualization.ophys import timeseries_figures as tf


if __name__ == '__main__':
    import sys
    experiment_id = sys.argv[1]

    dataset = loading.get_ophys_dataset(experiment_id)
    experiments_table = loading.get_filtered_ophys_experiment_table()
    ophys_session_id = experiments_table.loc[int(experiment_id)].ophys_session_id

    xlim_seconds = [400, 450]
    tf.plot_behavior_and_pop_avg(dataset, xlim_seconds, save_figure=True)
    xlim_seconds = [500, 600]
    tf.plot_behavior_and_cell_traces_pop_avg(dataset, xlim_seconds, save_figure=True)
    tf.plot_behavior_and_pop_avg_mesoscope(ophys_session_id, xlim_seconds=xlim_seconds, save_figure=True)

    xlim_seconds = [3400, 3450]
    tf.plot_behavior_and_pop_avg(dataset, xlim_seconds, save_figure=True)
    xlim_seconds = [3500, 3600]
    tf.plot_behavior_and_cell_traces_pop_avg(dataset, xlim_seconds, save_figure=True)
    tf.plot_behavior_and_pop_avg_mesoscope(ophys_session_id, xlim_seconds=xlim_seconds, save_figure=True)

    xlim_seconds = None
    tf.plot_behavior_and_pop_avg(dataset, xlim_seconds, save_figure=True)
    tf.plot_behavior_and_cell_traces_pop_avg(dataset, xlim_seconds, save_figure=True)
    tf.plot_behavior_and_pop_avg_mesoscope(ophys_session_id, xlim_seconds=None, save_figure=True)
