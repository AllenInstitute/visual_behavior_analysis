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

def load_running_df(bsid=None,pkl_path=None,camstim_type='foraging2'):
    '''
    loads running data from pkl file using VBA
    input is either the behavior session ID (bsid) or the pkl path (not both!)
    '''
    if bsid:
        pkl_path = db.get_pkl_path(int(bsid))
    
    data = pd.read_pickle(pkl_path)
    if camstim_type == 'foraging2':
        core_data = data_to_change_detection_core(data)
    else:
        core_data = data_to_change_detection_core_legacy(data)
    return core_data['running']

def main(bsid):
    running_data = load_running_df(bsid)
    df_sample = running_data.query('time > 300 and time < 600')

    print('generating spline fits for a range of n_knot_factors')
    n_knot_factors = [3,10,20,50,100]
    for n_knot_factor in n_knot_factors:
        print('n_knot_factor = {}'.format(n_knot_factor))
        df_sample = utilities.apply_spline_regression(df_sample, n_knot_factor)

    fig,ax = utilities.make_visualization(df_sample.query('time >= 310 and time <=330'), n_knot_factors)
    plt.subplots_adjust(top=0.9)
    fig.suptitle('bsid = {}'.format(bsid))
    fig.savefig(
        '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/running_smoothing/sample_data_figures/bsid={}.png'.format(bsid),
        dpi=150
    )

    print('calculating cost function')
    optimization_results = utilities.calculate_cost_function(df_sample)
    optimal_f = utilities.calculate_optimal_knot_factor(optimization_results)

    document = {}
    document['behavior_session_id'] = bsid
    document['optimal_f'] = optimal_f
    # document['data_sample'] = df_sample.to_dict('records')
    document['optimization_results'] = optimization_results.to_dict('records')

    print('logging to db')
    conn = db.Database('visual_behavior_data')
    collection = conn['behavior_analysis']['running_smoothing']
    db.update_or_create(
        collection, 
        document, 
        keys_to_check = ['behavior_session_id'], 
        force_write=False
    )
    print('done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run spline regression on a data subset')
    parser.add_argument(
        '--bsid',
        type=int,
        default=0,
        metavar='behavior_session_id'
    )
    args = parser.parse_args()

    main(args.bsid)