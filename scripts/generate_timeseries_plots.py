#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import visual_behavior.data_access.loading as loading



if __name__ == '__main__':
    import sys
    experiment_id = sys.argv[1]

    dataset = loading.get_ophys_dataset(experiment_id)

    xlim_seconds = [400, 500]
    plot_behavior_and_pop_avg(dataset, xlim_seconds, save_figure=True)
    plot_behavior_and_cell_traces_pop_avg(dataset, xlim_seconds, save_figure=True)

    xlim_seconds = [3500, 3600]
    plot_behavior_and_pop_avg(dataset, xlim_seconds, save_figure=True)
    plot_behavior_and_cell_traces_pop_avg(dataset, xlim_seconds, save_figure=True)

    xlim_seconds = None
    plot_behavior_and_pop_avg(dataset, xlim_seconds, save_figure=True)
    plot_behavior_and_cell_traces_pop_avg(dataset, xlim_seconds, save_figure=True)

