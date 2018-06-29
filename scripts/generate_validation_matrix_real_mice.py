import os
import glob
import pandas as pd
import numpy as np

from visual_behavior.translator.foraging2 import data_to_change_detection_core
from visual_behavior.translator.core import create_extended_dataframe
from visual_behavior.validation.qc import generate_qc_report

import seaborn as sns
import matplotlib.pyplot as plt

def get_files():
    path=r'//allen/programs/braintv/workgroups/ophysdev/oPhysQC/mouse_seeks/reports'
    mouse_seeks_output = pd.read_csv(os.path.join(path,'BEHAVIOR_report.csv'))
    mouse_seeks_output['timestamp']=mouse_seeks_output['lims_behavior_session_created_at'].map(lambda x:pd.to_datetime(x))
    outcomes = pd.read_csv('//allen/programs/braintv/workgroups/ophysdev/oPhysQC/mouse_seeks/reports/visual_behavior_mouse_validation_matrix.csv')

    f2_files = mouse_seeks_output[
        (mouse_seeks_output['timestamp']>pd.to_datetime('2018-06-01'))
    ].sort_values(by='timestamp',ascending=True)

    return outcomes,f2_files

def run_qc(f2_files,outcomes=None):
    all_results = []
    for idx,row in f2_files.iterrows():
        if outcomes is not None and row['lims_behavior_session_foraging_id'] not in outcomes.session_id.unique():
            try:
                data = pd.read_pickle(glob.glob(row['lims_behavior_session_storage_directory']+'*.pkl')[0])
                core_data = data_to_change_detection_core(data)

                intervals = np.diff(core_data['time'])
                percent_dropped_frames = 100.0 * len(intervals[intervals >= (0.03)]) / len(intervals)

                print('mouse = {}'.format(core_data['metadata']['mouseid']))
                print('stage = {}'.format(core_data['metadata']['stage']))
                print('date = {}'.format(pd.to_datetime(core_data['metadata']['startdatetime']).strftime('%m-%d-%Y')))
                print('rig ID = {}'.format(core_data['metadata']['rig_id']))
                print('')

                results = generate_qc_report(core_data)
                results.update({
                    'mouse': core_data['metadata']['mouseid'], 
                    'stage': core_data['metadata']['stage'],
                    'date':pd.to_datetime(core_data['metadata']['startdatetime']).strftime('%m-%d-%Y'),
                    'rig_id':core_data['metadata']['rig_id'],
                    'session_id':row['lims_behavior_session_foraging_id'],
    #                 'percent_dropped_frames':percent_dropped_frames
                })

                all_results.append(results)
            except Exception as e:
                print('failed on {}'.format(glob.glob(row['lims_behavior_session_storage_directory']+'*.pkl')[0]))
                print('mouse ID = {}'.format(row['lims_behavior_session_external_donor_name']))
                print(e)
                print(' ')
    
    return all_results

def add_new_results_to_outcomes(new_results,outcomes):
    all_results = run_qc(f2_files,outcomes)
    outcomes = pd.concat((outcomes,pd.DataFrame(all_results)))
    outcomes.to_csv('//allen/programs/braintv/workgroups/ophysdev/oPhysQC/mouse_seeks/reports/visual_behavior_mouse_validation_matrix.csv',index=False)

    return outcomes 

def make_plot(outcomes):
    outcomes_sorted = (
        outcomes
        .drop(labels=['session_id'],axis=1)
        .sort_values(['date','stage', 'mouse','rig_id'])
        .set_index(['date','stage', 'mouse','rig_id'])
    )
    fig, ax = plt.subplots(figsize=(12, 0.15*len(outcomes)))
    ax = sns.heatmap(
        data=outcomes_sorted,
        mask=pd.isnull(outcomes_sorted),
        xticklabels=True,
        yticklabels=True,
        cmap='copper',
        cbar=False,
        alpha=0.75,
        vmin=0,
        vmax=1,
    )

    xticklabels = ax.get_xticklabels()
    ax.set_xticklabels(xticklabels, rotation=40, ha='right', fontsize=8)
    yticklabels = ax.get_yticklabels()
    ax.set_yticklabels(yticklabels, rotation=0, ha='right', fontsize=8)
    ax.grid(False)
    ax.set_title('Fraction of passing validations = {:.2f}'.format(np.nanmean(np.array(outcomes_sorted.fillna(0).values.astype(float)), axis=(0, 1))))
    fig.tight_layout()

    fig.savefig('//allen/programs/braintv/workgroups/ophysdev/oPhysQC/mouse_seeks/reports/validation_matrix.png',dpi=300)

    return fig

if __name__ == '__main__':
    outcomes,f2_files = get_files()
    new_results = run_qc(f2_files,outcomes)
    outcomes = add_new_results_to_outcomes(new_results,outcomes)
    fig=make_plot(outcomes)