import os
import glob
import pandas as pd
import numpy as np

from visual_behavior.translator.foraging2 import data_to_change_detection_core
from visual_behavior.translator.core import create_extended_dataframe
from visual_behavior.validation.qc import generate_qc_report

import visual_behavior_research.plotting as vbp

import seaborn as sns
import matplotlib.pyplot as plt


def get_PKL_path(row):
    '''
    get the PKL path in the mouse-seeks log.
    It's in a different column depending on whether it's a behavior only or behavior+ophys session
    '''
    if 'lims_behavior_session_storage_directory' in row.keys():
        return row['lims_behavior_session_storage_directory']             
    else:
        return row['lims_ophys_session_storage_directory']  

def get_mouse_seeks_output():
    '''
    Use the cached mouse-seeks log files to find all PKL files in behavior and ophys
    '''
    mouse_seeks_output = pd.DataFrame()

    report_names=['BEHAVIOR_report.csv','VisualBehaviorIntegrationTest_report.csv',]
    date_columns=['lims_behavior_session_created_at','lims_ophys_session_date_of_acquisition']

    for report_name,date_column in zip(report_names,date_columns):
        path=r'\\allen\programs\braintv\workgroups\ophysdev\oPhysQC\mouse_seeks\reports'
        new_data = pd.read_csv(os.path.join(path,report_name))
        new_data['timestamp']=new_data[date_column].map(lambda x:pd.to_datetime(x))
        new_data['PKL_path']=new_data.apply(get_PKL_path,axis=1)
        mouse_seeks_output = pd.concat((mouse_seeks_output,new_data)).reset_index()

        f2_files = mouse_seeks_output[
                (mouse_seeks_output['timestamp']>pd.to_datetime('2018-10-01'))
                &(mouse_seeks_output['change_detection_task'].str.contains('DoC'))
            ].sort_values(by='timestamp',ascending=True)

    return f2_files

def load_data(f2_files,dropped_frame_log):
    '''
    load any files that aren't already in the dropped frame log into a dictionary called 'all_data'
    '''
    #figure out which foraging2 files aren't already in the dropped frame log
    all_timestamps = list(dropped_frame_log['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S'))
    already_loaded = f2_files.timestamp.dt.strftime('%Y-%m-%d %H:%M:%S').isin(all_timestamps)
    recent_f2_files = f2_files[~already_loaded].copy().reset_index(drop=True)

    all_data = {}
    for idx,row in recent_f2_files.iterrows():
        try:
            pkl_path = glob.glob(os.path.join(row['PKL_path'],'*.pkl'))[0]

            all_data[row['timestamp']] = {}
            all_data[row['timestamp']]['data'] = pd.read_pickle(pkl_path)
            all_data[row['timestamp']]['core_data'] = data_to_change_detection_core(all_data[row['timestamp']]['data'])
        
            print('loaded file {} of {}'.format(idx+1,len(recent_f2_files),),end='\r')
        except:
            pass
    print('\n')

    return all_data

def build_dropped_frame_log(all_data,dropped_frame_log):
    '''
    combines data in all_data dictionary with existing dropped frame log
    '''
    dropped_frame_log_list = []
    for key in all_data.keys():
        cd = all_data[key]['core_data']
        frame_intervals = np.diff(cd['time'])
        dropped_frame_log_list.append(
            pd.DataFrame({
                'mouse_id':cd['metadata']['mouseid'],
                'rig_id':cd['metadata']['rig_id'],
                'stage':cd['metadata']['stage'],
                'timestamp':key,
                'frame_intervals':[frame_intervals],
                'N_dropped_frames':len(frame_intervals[frame_intervals>0.025]),
                'mean_dropped_frame_length':np.mean(frame_intervals[frame_intervals>0.025]),
                'median_dropped_frame_length':np.median(frame_intervals[frame_intervals>0.025]),
                'max_dropped_frame_length':np.max(frame_intervals[frame_intervals>0.025]) if len(frame_intervals[frame_intervals>0.025])>0 else np.nan,
            },index=[0])
        )

    if len(dropped_frame_log_list) > 0:
        dropped_frame_log_new = pd.concat(dropped_frame_log_list,sort=True).reset_index()
        dropped_frame_log = pd.concat((dropped_frame_log,dropped_frame_log_new),sort=True).drop(columns=['index'])
    dropped_frame_log['week']=dropped_frame_log['timestamp'].dt.week
    dropped_frame_log['dayofweek']=dropped_frame_log['timestamp'].dt.dayofweek
    dropped_frame_log['weekday_name']=dropped_frame_log['timestamp'].dt.weekday_name

    return dropped_frame_log

def make_boxplot(row,ax,x=0,swarmplot_max=250):
    '''
    makes a single boxplot and swarmplot
    '''
    frame_intervals = row['frame_intervals']*1000
    longest_frame=np.max(frame_intervals)
    dropped_frames = frame_intervals[frame_intervals>25]

    sns.boxplot(x=x+np.zeros_like(frame_intervals),y=frame_intervals,ax=ax,orient='vertical',fliersize=0,whis=np.inf)

    if len(dropped_frames)>0:
        if len(dropped_frames)>swarmplot_max:
            
            dropped_frames = np.hstack((
                np.random.choice(dropped_frames,size=swarmplot_max,replace=False),
                np.max(dropped_frames),
                np.min(dropped_frames)
            ))
            
        sns.swarmplot(x=x+np.zeros_like(dropped_frames),y=dropped_frames,ax=ax,color='darkred')

    ax.set_ylim(0,250)
    ax.set_title('{}\n{} dropped frames\nlongest_frame = {:.1f} ms\nmouse {}\n{}'.format(
        row['rig_id'],
        row['N_dropped_frames'],
        longest_frame,
        row['mouse_id'],
        row['timestamp'].strftime('%Y-%m-%d_%H:%M:%S')
    ),fontsize=9)

def make_summary_plot(summary,title=None):
    '''
    builds a summary plot for all sessions in a passed dataframe
    '''
    summary = summary.copy().sort_values(by='rig_id').reset_index()

    n_cols = 6
    n_rows = int(np.ceil(len(summary)/n_cols))
    fig,ax=plt.subplots(n_rows,n_cols,sharex=True,sharey=True,figsize=(12,n_rows*2.5))

    for idx,row in summary.iterrows():
        print('making boxplot for {}, #{} of {}          '.format(title,idx+1,len(summary)),end='\r')
        make_boxplot(row,ax.flatten()[idx],x=idx)

    for ii in range(idx+1,len(ax.flatten())):
        ax.flatten()[ii].axis('off')
        
    for v in range(0,len(summary),n_cols):
        ax.flatten()[v].set_ylabel('frame intervals (ms)')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if title:
        fig.suptitle(title)

    return fig,ax


def generate_dropped_frame_log():

    save_dir = r"\\ALLEN\programs\braintv\workgroups\nc-ophys\visual_behavior\dropped_frame_logs"

    print('\nloading existing dropped frame log...')
    existing_dropped_frame_log = pd.read_pickle(os.path.join(save_dir,"dropped_frame_df.pkl"))
    previous_length = len(existing_dropped_frame_log)
    print('length of existing log = {}'.format(previous_length))

    f2_files = get_mouse_seeks_output()
    new_data = load_data(f2_files,existing_dropped_frame_log)

    dropped_frame_log = build_dropped_frame_log(new_data,existing_dropped_frame_log)
    print('length of new log = {}'.format(len(dropped_frame_log)))
    if len(dropped_frame_log) > previous_length:
        # only save the log if we've added to it (i.e., new sessions logs were found)
        print('saving...')
        dropped_frame_log.to_pickle(os.path.join(save_dir,'dropped_frame_df.pkl'))

    # save a CSV of the dropped frame log. 
    # By removing the 'frame_intervals' column, it becomes possible to save in tabular form and the file size is reasonable
    df = dropped_frame_log.sort_values(by='rig_id').drop(columns='frame_intervals')
    df.to_csv(os.path.join(save_dir,'dropped_frame_log.csv'),index=False)

    weeks = np.sort(dropped_frame_log.week.unique())
    clusters = {
        'B':['B1', 'B2', 'B3', 'B4', 'B5', 'B6'],
        'D':['D4', 'D2', 'D1', 'D3', 'D5', 'D6'],
        'F':['F3', 'F2', 'F6', 'F5', 'F1', 'F4'],
        'Physiology':['2P4', 'NP3', 'MS1', '2P3', '2P5']
    }

    print('generating plots...')
    for week in weeks:
        for cluster in clusters:
            df = dropped_frame_log[
                (dropped_frame_log['rig_id'].isin(clusters[cluster]))
                &(dropped_frame_log['week']==week)
            ].sort_values(by=['rig_id','timestamp'])

            first_day = df['timestamp'].min().strftime('%Y-%m-%d')
            title='Week starting {}, {} Cluster Dropped Frame log'.format(first_day,cluster)

            plot_exists = title+'.png' in os.listdir(os.path.join(save_dir,'plots'))
            # only make the plot if the week matches the current week (to ensure that new data is plotted), or if the plot doesn't already exist
            if week == weeks[-1] or not plot_exists:
                fig,ax=make_summary_plot(
                    df,
                    title = title
                )
                # Save the figure
                vbp.save_figure(fig,os.path.join(save_dir,'plots',title),formats=['.png'])
                # Save a CSV of this week/cluster .The CSV can't be saved if someone has it open:
                try:
                    df.drop(columns=['frame_intervals']).to_csv(os.path.join(save_dir,'logs',title+'.csv'),index=False, date_format='%Y-%m-%d %H:%M:%S')
                except PermissionError:
                    print('Permission Error when writing save {}, does someone have it open?'.format(os.path.join(save_dir,'logs',title+'.csv')))
    print('\ndone\n')

if __name__ == '__main__':
    generate_dropped_frame_log()