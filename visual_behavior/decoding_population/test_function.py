#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This function is called in svm_images_init_pbs.py. We run svm_images_init_pbs.py in command line (cluster: slurm), so at the end of the function below we argeparse the inputs to this function.

This function sets svm vars and calls svm_images_main_pbs.py


Created on Wed Jul  7 14:24:17 2021
@author: farzaneh
"""

import os
import numpy as np
import pandas as pd
import pickle
import sys
import visual_behavior.data_access.loading as loading
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis
from visual_behavior.data_access import utilities
from svm_images_main_pbs import *


def test_function():

    print('___')
    print(sys.path)
    print('___')
    
    # get the list of all the 8 experiments for each session; we need this to set the depth and area for all experiments, since we need the metadata for all the 8 experiments for our analysis even if we dont analyze some experiments. this is because we want to know which experiment belongs to which plane. we later make plots for each plot individually.
    experiments_table = loading.get_filtered_ophys_experiment_table(include_failed_data=True) # , release_data_only=False



######################################################
######################################################
#%% For SLURM
######################################################
######################################################


if __name__ == "__main__":
    test_function()