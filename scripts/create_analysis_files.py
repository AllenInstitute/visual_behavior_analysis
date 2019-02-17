#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')

from visual_behavior.ophys.io.create_analysis_files import create_analysis_files



if __name__ == '__main__':
    import sys
    experiment_id = sys.argv[1]
    cache_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_pilot_analysis'

    import os
    folder = [folder for folder in os.listdir(cache_dir) if str(experiment_id) in folder]
    if len(folder) > 0:
        file = [file for file in os.listdir(os.path.join(cache_dir, folder[0])) if 'flash_response_df' in file]
        if len(file) == 0:
            print('no flash events for', experiment_id, '- generating')
            create_analysis_files(experiment_id, cache_dir, overwrite_analysis_files=True)
    # create_analysis_files(experiment_id, cache_dir, overwrite_analysis_files=True)
