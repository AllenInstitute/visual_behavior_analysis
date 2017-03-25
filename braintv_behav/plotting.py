import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import imp
import os
import time
import platform
import sys
import getpass
try:
    import seaborn as sns
except:
    pass

import imaging_behavior.core.utilities as ut
import imaging_behavior.plotting.plotting_functions as pf
import imaging_behavior.plotting.utilities as pu

if platform.system()=='Linux':
    dro_path = os.path.dirname(os.path.realpath(__file__))
    imp.load_source('dro',os.path.join(dro_path,'utilities.py'))
    import dro
elif getpass.getuser() == 'dougo':
    import dro
else:
    dro_path = '//aibsdata2/nc-ophys/BehaviorCode/dro'
    imp.load_source('dro',os.path.join(dro_path,'utilities.py'))
    import dro




def make_daily_figure(df_in,mouse_id=None,reward_window=None,sliding_window=100,mouse_image_before=None,mouse_image_after=None):
    '''
    Generates a daily summary plot for the detection of change task
    '''
    date = df_in.startdatetime.iloc[0].strftime('%Y-%m-%d')
    if mouse_id is None:
        mouse_id = df_in.mouse_id.unique()[0]
    df_nonaborted = df_in[(df_in.trial_type != 'aborted')&(df_in.trial_type != 'other')]

    if reward_window == None:
        try:
            reward_window = dro.get_reward_window(df_in)
        except:
            reward_window = [0.15,1]
    if sliding_window == None:
        sliding_window = len(df_nonaborted)

    fig = plt.figure(figsize=(12,8))

    #place axes
    ax = pu.placeAxesOnGrid(fig,dim=(1,4),xspan=(0,1),yspan=(0.425,1),sharey=True)
    ax_timeline = pu.placeAxesOnGrid(fig,xspan=(0.5,1),yspan=(0.225,0.3))
    ax_table = pu.placeAxesOnGrid(fig,xspan=(0.1,0.6),yspan=(0,0.25),frameon=False)

    if mouse_image_before is not None:
        try:
            titles = ['before session','after session']
            ax_image = pu.placeAxesOnGrid(fig,dim=(1,2),xspan=(0.5,1),yspan=(0,0.18),frameon=False)
            for index,im in enumerate([mouse_image_before,mouse_image_after]):
                if im is not None:
                    ax_image[index].imshow(im,cmap='gray')
                ax_image[index].grid(False)
                print index
                ax_image[index].axis('off')
                ax_image[index].set_title(titles[index],fontsize=14)
        except:
            pass

    #make table
    make_info_table(df_in,ax_table)

    #make timeline plot
    make_session_timeline_plot(df_in,ax_timeline)

    #make trial-based plots
    make_lick_raster_plot(df_nonaborted,ax[0],reward_window=reward_window)
    make_cumulative_volume_plot(df_nonaborted,ax[1])
    hit_rate,fa_rate,d_prime = dro.get_response_rates(df_nonaborted,sliding_window=sliding_window,reward_window=reward_window)
    make_rolling_response_probability_plot(hit_rate,fa_rate,ax[2])
    mean_rate = np.mean(dro.check_responses(df_nonaborted,reward_window=reward_window)==1.0)
    ax[2].axvline(mean_rate,color='0.5',linestyle=':')
    make_rolling_dprime_plot(d_prime,ax[3])


    plt.subplots_adjust(top=0.9)
    fig.suptitle('mouse = '+mouse_id+', '+date,
                 fontsize=20)

    return fig

def make_daily_plot(*args,**kwargs):
    '''original function name. Putting this here to avoid breaking old code'''
    make_daily_figure(*args,**kwargs)

def make_summary_figure(df,mouse_id):
    dfm = df[(df.mouse_id == mouse_id)]
    session_dates = np.sort(dfm.startdatetime.unique())
    #fig,ax=plt.subplots(1,5,sharey=True,figsize=(11.5,8))
    fig = plt.figure(figsize=(11.5,8))
    ax = []
    ax.append(pu.placeAxesOnGrid(fig,xspan=(0,0.2),yspan=(0,1)))
    ax.append(pu.placeAxesOnGrid(fig,xspan=(0.22,0.4),yspan=(0,1)))
    ax.append(pu.placeAxesOnGrid(fig,xspan=(0.42,0.6),yspan=(0,1)))
    ax.append(pu.placeAxesOnGrid(fig,xspan=(0.62,0.8),yspan=(0,1)))
    ax.append(pu.placeAxesOnGrid(fig,xspan=(0.82,1),yspan=(0,1)))

    make_ILI_plot(dfm,session_dates,ax[0])

    make_trial_type_plot(dfm,ax[1])

    make_performance_plot(dfm,ax[2])

    make_dprime_plot(dfm,ax[3])

    make_total_volume_plot(dfm,ax[4])

    for i in range(1,len(ax)):
        ax[i].set_yticklabels([])

    #fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig.suptitle('MOUSE = '+mouse_id,fontsize=18)

    return fig


def make_cumulative_volume_plot(df_in,ax):
    ax.barh(np.arange(len(df_in)),df_in.cumulative_volume,height=1.0,linewidth=0)
    ax.set_xlabel('Cumulative \nVolume (mL)',fontsize=14)
    ax.set_title('Cumulative Volume',fontsize=16)
    if np.max(df_in.cumulative_volume)<=0.5:
        ax.set_xticks(np.arange(0,0.5,0.1))
    else:
        ax.set_xticks(np.arange(0,np.max(df_in.cumulative_volume),0.25))



def make_rolling_response_probability_plot(hit_rate,fa_rate,ax):

    ax.plot(hit_rate,np.arange(len(hit_rate)),color='darkgreen',linewidth=2)
    ax.plot(fa_rate,np.arange(len(fa_rate)),color='orange',linewidth=2)

    ax.set_title('Resp. Prob.',fontsize=16)
    ax.set_xticks([0,0.25,0.5,0.75,1])
    ax.set_xlabel('Response Prob.',fontsize=14)

def make_rolling_dprime_plot(d_prime,ax,format='vertical'):
    if format=='vertical':
        ax.plot(d_prime,np.arange(len(d_prime)),color='black',linewidth=2)
        ax.set_xlabel("d'",fontsize=14)
    elif format=='horizontal':
        ax.plot(np.arange(len(d_prime)),d_prime,color='black',linewidth=2)
        ax.set_ylabel("d'",fontsize=14)
    ax.set_title("Rolling d'",fontsize=16)
    

def make_lick_raster_plot(df_in,ax,reward_window=None):

    if reward_window == None:
        try:
            reward_window = dro.get_reward_window(df_in)
        except:
            reward_window = [0.15,1]

    ax.axvspan(reward_window[0],reward_window[1],facecolor='k',alpha=0.5)
    lick_x = []
    lick_y = []

    reward_x = []
    reward_y = []
    for ii,idx in enumerate(df_in.index):
        if len(df_in.loc[idx]['lick_times'])>0:
            lt = np.array(df_in.loc[idx]['lick_times']) - df_in.loc[idx]['change_time']
            lick_x.append(lt)
            lick_y.append(np.ones_like(lt)*ii)

        if len(df_in.loc[idx]['reward_times'])>0:
            rt = np.array(df_in.loc[idx]['reward_times']) - df_in.loc[idx]['change_time']
            reward_x.append(rt)
            reward_y.append(np.ones_like(rt)*ii)

        ax.axhspan(ii-0.5,ii+0.5, facecolor=df_in.loc[idx]['color'], alpha=0.5)

    ax.plot(ut.flatten_list(lick_x),ut.flatten_list(lick_y),'.k')
    ax.plot(ut.flatten_list(reward_x),ut.flatten_list(reward_y),'o',color='blue')

    ax.set_xlim(-1,5)
    ax.set_ylim(-0.5,ii+0.5)
    ax.invert_yaxis()

    ax.set_title('Lick Raster',fontsize=16)
    ax.set_ylabel('Trial Number',fontsize=14)
    ax.set_xlabel('Time from \nstimulus onset (s)',fontsize=14)


def make_info_table(df,ax):
    '''
    generates a table with info extracted from the dataframe
    DRO - 10/13/16
    '''

    #define the data
    try:
        user_id = df.iloc[0]['user_id']
    except:
        user_id = 'unspecified'

    #note: training day is only calculated if the dataframe was loaded from the data folder, as opposed to individually
    try:
        training_day = df.iloc[0].training_day
    except:
        training_day = np.nan

    #I'm using a list of lists instead of a dictionary so that it maintains order
    #the second entries are in quotes so they can be evaluated below in a try/except
    data = [['Date','df.iloc[0].startdatetime.strftime("%m-%d-%Y")'],
            ['Training Day','training_day'],
            ['Time','df.iloc[0].startdatetime.strftime("%H:%M")'],
            ['Duration (minutes)','round(df.iloc[0]["session_duration"]/60.,2)'],
            ['Total water received (ml)','df["cumulative_volume"].max()'],
            ['Mouse ID','df.iloc[0]["mouse_id"]'],
            ['Task ID','df.iloc[0]["task"]'],
            ['Trained by','user_id'],
            ['Rig ID','df.iloc[0]["rig_id"]'],
            ['Stimulus Description','df.iloc[0]["stimulus"]'],
            ['Time between flashes','df.iloc[0].blank_duration_range[0]'],
            ['Black screen on timeout','df.iloc[0].blank_screen_timeout'],
            ['Minimum pre-change time','df.iloc[0]["prechange_minimum"]'],
            ['Trial duration','df.iloc[0].trial_duration']]

    cell_text = []
    for x in data:
        try:
            cell_text.append([eval(x[1])])
        except:
            cell_text.append([np.nan])

    #define row colors
    row_colors = [['lightgray'],['white']]*(len(data))


    #make the table
    table = ax.table(cellText=cell_text,
                          rowLabels=[x[0] for x in data],
                          rowColours=dro.flatten_list(row_colors)[:len(data)],
                          colLabels=None,
                          loc='center',
                          cellLoc='left',
                          rowLoc='right',
                          cellColours=row_colors[:len(data)])
    ax.grid(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    #do some cell resizing
    cell_dict=table.get_celld()
    for cell in cell_dict:
        if cell[1] == -1:
            cell_dict[cell].set_width(0.1)
        if cell[1] == 0:
            cell_dict[cell].set_width(0.5)


def make_session_timeline_plot(df_in,ax):
    licks = list(df_in['lick_times'])
    rewards = list(df_in['reward_times'])
    stimuli = list(df_in['change_time'])

    #This plots a vertical span of a defined color for every trial type
    #to save time, I'm only plotting a span when the trial type changes
    spanstart = 0
    trial = 0
    for trial in range(1,len(df_in)):
        if df_in.iloc[trial]['color'] != df_in.iloc[trial-1]['color']:
            ax.axvspan(spanstart,
                       df_in.iloc[trial]['starttime'],
                       color=df_in.iloc[trial-1]['color'],
                       alpha=0.75)
            spanstart = df_in.iloc[trial]['starttime']
    #plot a span for the final trial(s)
    ax.axvspan(spanstart,
               df_in.iloc[trial]['starttime']+df_in.iloc[trial]['trial_length'],
               color=df_in.iloc[trial-1]['color'],
               alpha=0.75)

    rewards = np.array(ut.flatten_list(rewards))
    licks = np.array(ut.flatten_list(licks))
    stimuli = np.array(ut.flatten_list(stimuli))

    ax.plot(licks,np.ones_like(licks),'.',color='black')
    ax.plot(rewards,1.1*np.ones_like(rewards),'o',
                     color='blue',markersize=6,alpha=0.75)
    ax.plot(stimuli,1.2*np.ones_like(stimuli),'d',color='indigo')

    ax.set_ylim(0.95,1.25)
    ax.set_xlim(0,df_in.iloc[trial]['starttime']+df_in.iloc[trial]['trial_length'])
    ax.set_yticklabels([])
    ax.set_xlabel('session time (s)',fontsize=14)
    ax.set_title('Full Session Timeline',fontsize=14)


def save_figure(fig, fname, formats = ['.pdf'],transparent=False,dpi=300,facecolor=None,**kwargs):
    import matplotlib as mpl
    mpl.rcParams['pdf.fonttype'] = 42

    if 'size' in kwargs.keys():
        fig.set_size_inches(kwargs['size'])

    elif 'figsize' in kwargs.keys():
        fig.set_size_inches(kwargs['figsize'])
    else:
        fig.set_size_inches(11,8.5)
    for f in formats:
        fig.savefig(fname + f, transparent = transparent, orientation = 'landscape',dpi=dpi)


def make_ILI_plot(dfm,session_dates,ax):
    ILIs = []
    positions = []
    dates = []
    for ii,date in enumerate(session_dates[:]):

        df1 = dfm[(dfm.startdatetime == date)]
        dates.append(df1.startdatetime.iloc[0].strftime('%Y-%m-%d'))

        ILI = np.diff(np.array(ut.flatten_list(list(df1.lick_times))))
        ILI = ILI[ILI>0.500]
        #if no licks are recorded, this will keep the violin plot from failing below
        if len(ILI) == 0:
            ILI = [0]
        ILIs.append(ILI)
        ax.scatter(ILI,ii+0.075*np.random.randn(len(ILI)),color='dodgerblue')

        med = np.median(ILI[np.logical_and(ILI>2,ILI<15)])
        ax.plot([med,med],[ii-0.25,ii+0.25],color='white',linewidth=3)

        positions.append(ii)

    ax.set_xlim(0,15)
    ax.set_xlabel('time between licks (s)')
    ax.set_yticks(np.arange(0,len(ILIs)))
    ax.set_ylim(-1,len(session_dates[:]))
    ax.set_yticklabels(dates)
    ax.set_title('Inter-lick \nintervals')
    ax.invert_yaxis()
    vplot = ax.violinplot(ILIs,positions, widths=0.8,showmeans=False, showextrema=False, vert=False)
    for patch in vplot['bodies']: patch.set_color('black'),patch.set_alpha(0.5)



def make_trial_type_plot(dfm,ax):
    sums = dfm.groupby(['startdatetime','color',]).sum()
    colors = ['blue','red','darkgreen','lightgreen','darkorange','yellow']
    dates = dfm.startdatetime.unique()
    all_vals = []
    for date in dates:
        total_dur = sums.loc[date]['trial_length'].sum()
        vals = []
        for color in colors:

            if color in sums.loc[date].index:
                fraction = sums.loc[date].loc[color]['trial_length']/total_dur
            else:
                fraction = 0
            vals.append(fraction)
        all_vals.append(vals)

    all_vals = np.array(all_vals)
    cumsum = np.hstack((np.zeros((np.shape(all_vals)[0],1)),np.cumsum(all_vals,axis=1)))
    width = 0.8
    for jj in range(np.shape(all_vals)[1]):
        ax.barh(np.arange(np.shape(all_vals)[0])-0.25, all_vals[:,jj], height=0.6,color=colors[jj],left=cumsum[:,jj])
    ax.set_xlim(0,1)
    ax.set_ylim(-1,len(dates))
    ax.set_title('fraction of time in \neach trial type')
    ax.set_xlabel('Time fraction of session')
    ax.invert_yaxis()


def make_performance_plot(df_in,ax,reward_window=None,sliding_window=None):

    if sliding_window == None:
        calculate_sliding_window = True
    else:
        caclulate_sliding_window = False



    dates = df_in.startdatetime.unique()
    max_hit_rates = []
    mean_hit_rates = []
    max_false_alarm_rates = []
    mean_false_alarm_rates = []
    max_dprime = []
    mean_dprime = []
    for ii,date in enumerate(dates):

        df1 = df_in[(df_in.startdatetime == date)&(df_in.trial_type != 'aborted')]

        if calculate_sliding_window == True:
            sliding_window = len(df1)

        hit_rate,fa_rate,d_prime = dro.get_response_rates(df1,sliding_window=sliding_window,reward_window=reward_window)

        max_hit_rates.append(np.nanmax(hit_rate[int(len(hit_rate)/3):]))
        max_false_alarm_rates.append(np.nanmax(fa_rate[int(len(hit_rate)/3):]))
        max_dprime.append(np.nanmax(d_prime[int(len(hit_rate)/3):]))

        mean_hit_rates.append(np.nanmean(hit_rate[int(len(hit_rate)/3):]))
        mean_false_alarm_rates.append(np.nanmean(fa_rate[int(len(hit_rate)/3):]))
        mean_dprime.append(np.nanmean(d_prime[int(len(hit_rate)/3):]))

    height = 0.35
    ax.barh(np.arange(len(max_hit_rates))-height,max_hit_rates,height=height,color='darkgreen',alpha=1)
    ax.barh(np.arange(len(max_hit_rates)),max_false_alarm_rates,height=height,color='orange',alpha=1)

    ax.set_title('PEAK Hit \nand FA Rates')
    ax.set_xlabel('Max Response Probability')
    ax.set_xlim(0,1)
    ax.set_ylim(-1,len(dates))
    ax.invert_yaxis()

def make_dprime_plot(df_in,ax,reward_window=None,return_vals=False,sliding_window=None):

    if sliding_window == None:
        calculate_sliding_window = True
    else:
        calculate_sliding_window = False

    dates = df_in.startdatetime.unique()
    max_hit_rates = []
    mean_hit_rates = []
    max_false_alarm_rates = []
    mean_false_alarm_rates = []
    max_dprime = []
    mean_dprime = []
    for ii,date in enumerate(dates):

        df1 = df_in[(df_in.startdatetime == date)&(df_in.trial_type != 'aborted')]

        if calculate_sliding_window == True:
            sliding_window = len(df1)

        hit_rate,fa_rate,d_prime = dro.get_response_rates(df1,sliding_window=sliding_window,reward_window=reward_window)

        max_hit_rates.append(np.nanmax(hit_rate[int(len(hit_rate)/3):]))
        max_false_alarm_rates.append(np.nanmax(fa_rate[int(len(hit_rate)/3):]))
        max_dprime.append(np.nanmax(d_prime[int(len(hit_rate)/3):]))

        mean_hit_rates.append(np.nanmean(hit_rate[int(len(hit_rate)/3):]))
        mean_false_alarm_rates.append(np.nanmean(fa_rate[int(len(hit_rate)/3):]))
        mean_dprime.append(np.nanmean(d_prime[int(len(hit_rate)/3):]))

    height = 0.7
    ax.barh(np.arange(len(max_hit_rates))-height/2,max_dprime,height=height,color='black',alpha=1)
    # ax.barh(np.arange(len(max_hit_rates)),mean_dprime,height=height,color='gray',alpha=1)

    ax.set_title('PEAK \ndprime')
    ax.set_xlabel('Max dprime')
    ax.set_xlim(0,4.75)
    ax.set_ylim(-1,len(dates))
    ax.invert_yaxis()
    if return_vals==True:
        return max_dprime



def make_total_volume_plot(df_in,ax):
    dates = df_in.startdatetime.unique()
    total_volume = []
    number_correct = []
    for ii,date in enumerate(dates):



        df1 = df_in[(df_in.startdatetime == date)&(df_in.trial_type != 'aborted')]

        total_volume.append(df1.number_of_rewards.sum()*df1.reward_volume.max())
        number_correct.append(df1.number_of_rewards.sum())

    ax.plot(total_volume,np.arange(len(total_volume)),'o-',color='blue',alpha=1)

    ax.set_title('Total \nVolume Earned')
    ax.set_xlabel('Volume (mL)')
    ax.set_xlim(0,1.5)
    ax.set_ylim(-1,len(dates))
    ax.invert_yaxis()

def DoC_PsychometricCurve(input,ax=None,parameter='delta_ori',title="",linecolor='black',
    xval_jitter=0,initial_guess=(0.1,1,0.2,0.2),fontsize=14,logscale=True,minval=0.4,
    xticks=[0.0,1.0,2.5,5.0,10.0,20.0,45.0,90.0],xlim=(0,90)):


    if isinstance(input,str):
        #if a string is input, assume it's a filname. Load it
        df = dro.create_doc_dataframe(input)
        response_df = dro.make_response_df(df[((df.trial_type=='go')|(df.trial_type=='catch'))])
    elif isinstance(input,pd.DataFrame) and 'response_probability' not in input.columns:
        #this would mean that a response_probability dataframe has not been passed. Create it
        df = input
        response_df = dro.make_response_df(df[((df.trial_type=='go')|(df.trial_type=='catch'))])
    elif isinstance(input,pd.DataFrame) and 'response_probability' in input.columns:
        response_df = input
    else:
        print "can't deal with input"

    if ax == None:
        fig,ax=plt.subplots()

    if parameter == 'delta_ori':
        xlabel = '$\Delta$Orientation'
    else:
        xlabel=parameter

    pf.plotPsychometric(response_df[parameter],
                        response_df['response_probability'],
                        CI=response_df['CI'],
                        xlim=xlim,
                        xlabel=xlabel,
                        xticks = xticks,
                        title = title,
                        minval = minval,
                        logscale = logscale,
                        ax = ax,
                        linecolor = linecolor,
                        alpha=0.75,
                        xval_jitter=xval_jitter,
                        initial_guess=initial_guess,
                        fontsize=fontsize)

    return ax
