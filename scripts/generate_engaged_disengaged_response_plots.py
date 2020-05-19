import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from visual_behavior import database as db
from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession
import seaborn as sns
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis
import visual_behavior.ophys.response_analysis.response_processing as rp
import visual_behavior.data_access.loading as loading
import time
from dro.modules import plotting as plotting
import visual_behavior.visualization.ophys.summary_figures as sf

import argparse

sns.set_context('notebook', font_scale=1.5)

# convert 'flash_metrics_label' to 'engagement_state'


def assign_engagement_label(input_label):
    if 'high-lick' in input_label:
        return 'engaged'
    elif 'low-lick' in input_label:
        return 'disengaged'


def load_df(dataset, df_name):

    query = '''
        select bs.id
        from behavior_sessions bs
        join ophys_experiments oe on oe.ophys_session_id = bs.ophys_session_id
        where oe.id = {}
    '''
    behavior_session_id = db.lims_query(query.format(dataset.ophys_experiment_id))

    analysis = ResponseAnalysis(dataset, overwrite_analysis_files=False)
    sdf = analysis.get_response_df(df_name='stimulus_response_df', format='wide')
    flash_metrics = loading.get_stim_metrics_summary(behavior_session_id, load_location='from_database')
    flash_metrics = flash_metrics.drop(columns=['start_time']).merge(
        sdf[['stimulus_presentations_id', 'start_time']].drop_duplicates(),
        left_on='flash_index',
        right_on='stimulus_presentations_id',
        how='left'
    )
    merge_columns = {
        'stimulus_response_df': {
            'left': 'stimulus_presentations_id',
            'right': 'flash_index',
        },
        'omission_response_df': {
            'left': 'stimulus_presentations_id',
            'right': 'flash_index',
        },
        'trials_response_df': {
            'left': 'change_time',
            'right': 'start_time',
        },
    }
    df = analysis.get_response_df(df_name=df_name, format='long')
    df = df.merge(
        flash_metrics,
        left_on=merge_columns[df_name]['left'],
        right_on=merge_columns[df_name]['right'],
        how='left',
        suffixes=('', '_duplicate')
    )
    df['engagement_state'] = df['flash_metrics_labels'].map(lambda x: assign_engagement_label(x))
    return df


def colormap():
    colormap = {
        'engaged': 'olivedrab',
        'disengaged': 'firebrick'
    }
    return colormap

def make_engagement_time_summary_plot(sdf,ax):
    sdf_engagement_state = sdf.drop_duplicates('start_time')[['start_time','engagement_state']].copy().reset_index()
    
    sdf_engagement_state['next_start_time'] = sdf_engagement_state['start_time'].shift(-1)
    sdf_engagement_state['state_change'] = sdf_engagement_state['engagement_state'] != sdf_engagement_state['engagement_state'].shift()
    
    state_changes = sdf_engagement_state.query('state_change == True').copy()
    state_changes['next_state_change'] = state_changes['start_time'].shift(-1)
    
    state_colors = colormap()
    
    for idx,row in state_changes.iterrows():
        ax.axvspan(row['start_time']/60.,row['next_state_change']/60.,color=state_colors[row['engagement_state']])
    ax.set_yticks([])
    ax.set_xlabel('session time (minutes)')
    ax.set_title('engagement state vs. session time')
    ax.set_xlim(0,sdf['start_time'].max()/60.)

def seaborn_plot(df, ax, cell_id, legend='brief'):
    state_colors = colormap()

    sns.lineplot(
        x='eventlocked_timestamps',
        y='eventlocked_traces',
        data=df.query('cell_specimen_id == @cell_id'),
        hue='engagement_state',
        hue_order = ['engaged','disengaged'],
        palette = [state_colors[state] for state in ['engaged','disengaged']],
        ax=ax,
        legend=legend,
    )


def make_plots(dataset):
    figure_savedir = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/summary_plots/single_cell_plots/response_plots'
    hspace = 0.05

    titles = [
        'stimulus response',
        'omission response',
        'change response'
    ]
    dfs = {}
    for ii, df_name in enumerate(['stimulus_response_df', 'omission_response_df', 'trials_response_df']):
        dfs[df_name] = load_df(dataset, df_name)

    oeid = dataset.ophys_experiment_id

    for cell_id in dfs['omission_response_df']['cell_specimen_id'].unique():
        fig = plt.figure(figsize=(20, 8))
        ax = {
            'state_summary': sf.placeAxesOnGrid(fig, xspan=[0, 1], yspan=[0, 0.2]),
            'response_dfs': sf.placeAxesOnGrid(fig, dim = [1,3], xspan=[0, 1], yspan=[0.3, 1], sharey=True),
        }

        make_engagement_time_summary_plot(dfs['stimulus_response_df'], ax['state_summary'])

        for ii, df_name in enumerate(['stimulus_response_df', 'omission_response_df', 'trials_response_df']):
            ax['response_dfs'][ii].cla()
            df = dfs[df_name]
            if df_name == 'stimulus_response_df':
                legend = 'brief'
            else:
                legend = False

            seaborn_plot(df, ax['response_dfs'][ii], cell_id, legend=legend)

            if df_name == 'stimulus_response_df':
                plotting.designate_flashes(ax['response_dfs'][ii])
                ax['response_dfs'][ii].set_xlim(-0.5, 0.75)
            elif df_name == 'omission_response_df':
                plotting.designate_flashes(ax['response_dfs'][ii], omit=0)
                ax['response_dfs'][ii].set_xlim(-3, 3)
            elif df_name == 'trials_response_df':
                plotting.designate_flashes(ax['response_dfs'][ii], pre_color='black', post_color='green')
                ax['response_dfs'][ii].set_xlim(-3, 3)
            ax['response_dfs'][ii].set_xlabel('time (s)')
            ax['response_dfs'][0].set_ylabel('$\Delta$F/F')

        fig.tight_layout()
        plt.subplots_adjust(top=0.9)
        title = get_title(oeid, cell_id)
        fig.suptitle(title)
        fig.savefig(os.path.join(figure_savedir, title + '.png'), dpi=200)


def get_title(oeid, cell_specimen_id):
    cache = loading.get_visual_behavior_cache()
    experiments_table = loading.get_filtered_ophys_experiment_table()

    row = experiments_table.query('ophys_experiment_id == @oeid').iloc[0].to_dict()
    title = '{}__specimen_id={}__exp_id={}__{}__{}__depth={}__cell_id={}'.format(
        row['cre_line'],
        row['specimen_id'],
        row['ophys_experiment_id'],
        row['session_type'],
        row['targeted_structure'],
        row['imaging_depth'],
        cell_specimen_id,
    )
    return title


def load_flashwise_summary(behavior_session_id=None):
    conn = db.Database('visual_behavior_data')
    collection = conn['behavior_analysis']['annotated_stimulus_presentations']

    if behavior_session_id is None:
        # load all
        df = pd.DataFrame(list(collection.find({})))
    else:
        # load data from one behavior session
        df = pd.DataFrame(list(collection.find({'behavior_session_id': int(behavior_session_id)})))

    conn.close()

    return df.sort_values(by=['behavior_session_id', 'flash_index'])


def load_dataset(experiment_id):
    cache_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\visual_behavior_production_analysis'.replace('\\', '/')
    return loading.get_ophys_dataset(experiment_id, cache_dir)


def generate_save_plots(experiment_id):
    dataset = load_dataset(experiment_id)
    make_plots(dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate engaged/disengaged response plots')
    parser.add_argument(
        '--oeid',
        type=int,
        default=0,
        metavar='ophys_experiment_id'
    )
    args = parser.parse_args()
    generate_save_plots(args.oeid)
