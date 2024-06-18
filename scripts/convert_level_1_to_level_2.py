#!/usr/bin/env python

import os
import argparse
import numpy as np
import pandas as pd
import visual_behavior.data_access.loading as loading
import visual_behavior.ophys.io.create_multi_session_df as io

from visual_behavior.ophys.io.convert_level_1_to_level_2 import convert_level_1_to_level_2


if __name__ == '__main__':
    # define args
    parser = argparse.ArgumentParser()
    # parser.add_argument('--ophys_container_id', type=str, help='ophys_container_id to use')
    parser.add_argument('--ophys_experiment_id', type=str, help='ophys_experiment_id to use')
    args = parser.parse_args()
    # ophys_container_id = int(args.ophys_container_id)
    ophys_experiment_id = int(args.ophys_experiment_id)

    cache_dir = r'/allen/programs/mindscope/workgroups/learning/ophys/learning_project_cache'
    ophys_data = convert_level_1_to_level_2(ophys_experiment_id, cache_dir)

