import os
import sys
import glob
import pandas as pd
import numpy as np
import six
import json

from dateutil import tz
from datetime import datetime

from visual_behavior.translator.foraging2 import data_to_change_detection_core
from visual_behavior.translator.core import create_extended_dataframe

import visual_behavior_research.plotting as vbp

import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


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
        if six.PY2:
            mouse_seeks_output = pd.concat((mouse_seeks_output,new_data)).reset_index()
        elif six.PY3:
            mouse_seeks_output = pd.concat((mouse_seeks_output,new_data),sort=True).reset_index()

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
    print('len(dropped_frame_log):{}'.format(len(dropped_frame_log)))
    if len(dropped_frame_log)>0:

        # dropped_frame_log['timestamp_gmt'] = pd.to_datetime(dropped_frame_log['timestamp_gmt'])
        all_timestamps = list(dropped_frame_log['timestamp_local'].dt.strftime('%Y-%m-%d %H:%M:%S'))
        already_loaded = f2_files.timestamp.dt.strftime('%Y-%m-%d %H:%M:%S').isin(all_timestamps)
    else:
        already_loaded =  pd.Series([False]*len(f2_files),index=f2_files.index)

    recent_f2_files = f2_files[~already_loaded].copy().reset_index(drop=True)
    print('len(recent_f2_files):{}'.format(len(recent_f2_files)))
    all_data = {}
    for idx,row in recent_f2_files.iterrows():
        try:
            pkl_path = glob.glob(os.path.join(row['PKL_path'],'*.pkl'))[0]

            data = pd.read_pickle(pkl_path)
            # timestamp = data['start_time'].strftime('%Y-%m-%d %H:%M:%S')
            timestamp = data['start_time']
            all_data[timestamp] = {}
            all_data[timestamp]['data'] = data
            all_data[timestamp]['core_data'] = data_to_change_detection_core(all_data[timestamp]['data'])
        
            print('loaded file {} of {}'.format(idx+1,len(recent_f2_files),),end='\r')
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    print('\n')

    return all_data

def count_long_dropped_frames(frame_intervals,threshold=0.050):
    return len(frame_intervals[frame_intervals>threshold])

def get_dropped_frame_times(row):
    long_dropped_frames = eval(row['long_dropped_frames'])
    times = np.cumsum(row['frame_intervals'])
    return str(times[long_dropped_frames])

def get_dropped_frame_durations(row):
    long_dropped_frames = np.array(eval(row['long_dropped_frames']))
    intervals = row['frame_intervals']
    if len(long_dropped_frames) > 0:
        return str(intervals[long_dropped_frames-1].tolist())
    else:
        return '[]'

def build_list_of_long_dropped_frames(frame_intervals,threshold=0.050):
    '''
    returns a string of comma seperated values
    '''
    long_frames_indices = np.array(range(1,len(frame_intervals)+1))[frame_intervals>threshold]
    return str(long_frames_indices.tolist())

def convert_timestamp_to_local(ts):
    from_zone = tz.tzutc()
    to_zone = tz.tzlocal()
    return ts.replace(tzinfo=from_zone).astimezone(to_zone)

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
                'timestamp_local':key,
                'timestamp_gmt':None,
                'frame_intervals':[frame_intervals],
                'mean_dropped_frame_length':np.mean(frame_intervals[frame_intervals>0.025]),
                'median_dropped_frame_length':np.median(frame_intervals[frame_intervals>0.025]),
                'max_dropped_frame_length':np.max(frame_intervals[frame_intervals>0.025]) if len(frame_intervals[frame_intervals>0.025])>0 else np.nan,
            },index=[0])
        )

    if len(dropped_frame_log_list) > 0:
        dropped_frame_log_new = pd.concat(dropped_frame_log_list,sort=True).reset_index()

        dropped_frame_log_new['week']=dropped_frame_log_new['timestamp_local'].dt.week
        dropped_frame_log_new['dayofweek']=dropped_frame_log_new['timestamp_local'].dt.dayofweek
        dropped_frame_log_new['weekday_name']=dropped_frame_log_new['timestamp_local'].dt.weekday_name
        dropped_frame_log_new['n_dropped_frames'] = dropped_frame_log_new['frame_intervals'].map(lambda x: len(x[x>0.025]))
        dropped_frame_log_new['long_dropped_frame_count'] = dropped_frame_log_new['frame_intervals'].map(lambda x: count_long_dropped_frames(x))
        dropped_frame_log_new['long_dropped_frames'] = dropped_frame_log_new['frame_intervals'].map(lambda x: build_list_of_long_dropped_frames(x))
        dropped_frame_log_new['long_dropped_frame_times'] = dropped_frame_log_new.apply(get_dropped_frame_times,axis=1)
        dropped_frame_log_new['long_frame_durations'] = dropped_frame_log_new.apply(get_dropped_frame_durations,axis=1)

        dropped_frame_log = pd.concat((dropped_frame_log,dropped_frame_log_new),sort=True).drop(columns=['index'])

    if 'level_0' in dropped_frame_log.columns:
        dropped_frame_log.drop(columns='level_0',inplace=True)

    return dropped_frame_log

def make_boxplot(frame_intervals,ax,swarmplot_max=250,orient='vertical'):
    '''
    makes a single boxplot and swarmplot
    '''

    longest_frame=np.max(frame_intervals)
    dropped_frames = frame_intervals[frame_intervals>25]
    
    if orient == 'vertical':
        ax.set_ylim(0,1.1*max(dropped_frames))
        x=np.zeros_like(frame_intervals)
        y=frame_intervals
    elif orient == 'horizontal':
        ax.set_xlim(0,1.1*max(dropped_frames))
        y=np.zeros_like(frame_intervals)
        x=frame_intervals

    sns.boxplot(x,y,ax=ax,orient=orient,fliersize=0,whis=np.inf)

    if len(dropped_frames)>0:
        if len(dropped_frames)>swarmplot_max:
            
            dropped_frames = np.hstack((
                np.random.choice(dropped_frames[dropped_frames<100],size=swarmplot_max,replace=False),
                np.min(dropped_frames),
                dropped_frames[dropped_frames>=100]
            ))
        
        if orient == 'vertical':
            sns.swarmplot(x=np.zeros_like(dropped_frames),y=dropped_frames,ax=ax,color='darkred',orient=orient)
        elif orient == 'horizontal':
            sns.swarmplot(x=dropped_frames,y=np.zeros_like(dropped_frames),ax=ax,color='darkred',orient=orient)

    ax.set_ylim(0,250)
    ax.set_title('{}\n{} dropped frames\nlongest_frame = {:.1f} ms\nmouse {}\n{}'.format(
        row['rig_id'],
        row['n_dropped_frames'],
        longest_frame,
        row['mouse_id'],
        row['timestamp_local'].strftime('%Y-%m-%d_%I:%M:%S %p')
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

def generate_text_summary(log_to_summarize,report_threshold=5):
    message = 'This is an autogenerated email. To opt out, contact Doug Ollerenshaw (dougo@alleninstitute.org)\n\n'
    message += 'There were {} behavior sessions run between {} and {}\n'.format(
        len(log_to_summarize),
        log_to_summarize['timestamp_local'].min().strftime('%Y-%m-%d'),
        log_to_summarize['timestamp_local'].max().strftime('%Y-%m-%d'),
        )
    message += '{} of those sessions had at least {} frames longer than 50 ms\n'.format(
        len(log_to_summarize[log_to_summarize['long_dropped_frame_count']>=report_threshold]),
        report_threshold
    )
    message += '-------------------------------------------------------------\n'
    for idx,row in log_to_summarize.sort_values(by=['long_dropped_frame_count','rig_id'],ascending=[False,True]).iterrows():
        if row['long_dropped_frame_count'] > report_threshold:
            message += '{} had {} {} > 50 ms on {}. The longest frame was {:0.0f} ms long \n'.format(
                row['rig_id'],
                row['long_dropped_frame_count'],
                'frame' if row['long_dropped_frame_count'] <= 1 else 'frames',
                row['timestamp_local'].strftime('%Y-%m-%d at %I:%M %p'),
                1000*row['max_dropped_frame_length']
            )
    path_to_plots = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\dropped_frame_logs\plots'
    path_to_logs = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\dropped_frame_logs\logs'
    message += '\nView dropped frame summary plots at: {}'.format(path_to_plots)
    message += '\nView dropped frame summary logs at:  {}'.format(path_to_logs)
    return message

def send_email(to_list,subject="Empty",message = "Empty"):
    import smtplib
    import pandas as pd
    gmail_credential_path=r"\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\dropped_frame_logs\.gmail_credentials.json"
    with open(gmail_credential_path) as f:
        credentials = json.load(f)

    FROM = 'Behavior Dropped Frame Monitor'

    for recipient in to_list:
    # Prepare actual message
        msg = """\From: %s\nTo: %s\nSubject: %s\n\n%s
        """ % (FROM, ", ".join([recipient]), subject, message)
        try:
            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.ehlo()
            server.starttls()
            server.login(credentials['username'], credentials['pwd'])
            server.sendmail(credentials['username'], [recipient], msg)
            server.quit()
            server.close()
        except Exception as e:
            print("failed to send mail")
            print(e)


def generate_dropped_frame_log(send_email=False):

    save_dir = r"\\ALLEN\programs\braintv\workgroups\nc-ophys\visual_behavior\dropped_frame_logs"
    # save_dir = r"F:\dropped_frame_logs"

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
            ].sort_values(by=['rig_id','timestamp_local'])

            if len(df) > 0:
                first_day = df['timestamp_local'].min().strftime('%Y-%m-%d')
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

        if week == weeks[-1] and send_email == True:
            this_weeks_log = dropped_frame_log[dropped_frame_log['week']==week]
            text_summary = generate_text_summary(this_weeks_log)
            recipient_list = pd.read_csv(r"\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\dropped_frame_logs\recipient_list.csv")
            
            # note: windows firewall blocks outgoing emails from python
            # it's possible to temporarily enable this, but settings revert within minutes
            # see: https://stackoverflow.com/questions/2778840/socket-error-errno-10013-an-attempt-was-made-to-access-a-socket-in-a-way-forb
            send_email(
                to_list = recipient_list.address.values.tolist(),
                subject = 'Visual Behavior Dropped Frame Summary for {} - {}'.format(
                    this_weeks_log['timestamp_local'].min().strftime('%Y-%m-%d'),
                    this_weeks_log['timestamp_local'].max().strftime('%Y-%m-%d'),
                    ),
                message = text_summary
            )

    print('\ndone\n')

if __name__ == '__main__':
    generate_dropped_frame_log()
