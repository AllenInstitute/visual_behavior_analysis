#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import visual_behavior.data_access.loading as loading
import visual_behavior.ophys.io.create_multi_session_df as io



if __name__ == '__main__':
    import sys
    project_code = sys.argv[1][1:-1]
    session_number = int(sys.argv[2][:-1])
    print(project_code, session_number)

    df_name = 'stimulus_response_df'
    conditions = ['cell_specimen_id', 'change', 'image_name']

    df = io.get_multi_session_df(project_code, session_number, df_name, conditions, use_events=False, use_extended_stimulus_presentations=False)
    print('done')