import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import visual_behavior.data_access.loading as loading
import visual_behavior.database as db
from visual_behavior.translator.foraging2 import data_to_change_detection_core

from visual_behavior.encoder_processing import utilities
from visual_behavior.encoder_processing.spline_regression import spline_regression

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run spline regression on a session')
    parser.add_argument(
        '--bsid',
        type=int,
        default=0,
        metavar='behavior_session_id'
    )
    parser.add_argument(
        '--f',
        type=int,
        default=12,
        metavar='f-value'
    )
    args = parser.parse_args()

    main(args.bsid)