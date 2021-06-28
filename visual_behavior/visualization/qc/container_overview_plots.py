from visual_behavior.data import loading as data_loading
import pandas as pd
import numpy as np

from visual_behavior.translator.foraging2 import data_to_change_detection_core
from visual_behavior.translator.core import create_extended_dataframe
from visual_behavior.change_detection.trials import summarize
import visual_behavior.database as db

import plotly.graph_objects as go

import xarray as xr


def oeid_to_uuid(oeid):
    return db.convert_id({'ophys_experiment_id': oeid}, 'behavior_session_uuid')


def get_oeid(container_df, ophys_container_id, session_number):
    entry_string = container_df.query('ophys_container_id == @ophys_container_id')['session_{}'.format(session_number)].iloc[0]
    if pd.notnull(entry_string):
        return int(entry_string.split(' ')[1])


def get_uuid(container_df, ophys_container_id, session_number):
    oeid = get_oeid(container_df, ophys_container_id, session_number)
    if pd.notnull(oeid):
        return oeid_to_uuid(oeid)
    else:
        return None


def get_session_stats(container_df, ophys_container_id, session_number):
    vb = db.Database('visual_behavior_data')
    behavior_session_uuid = get_uuid(container_df, ophys_container_id, session_number)
    stats = vb.behavior_data['summary'].find_one({'behavior_session_uuid': behavior_session_uuid})
    vb.close()
    return stats


def get_value(container_df, ophys_container_id, session_number, value):
    '''
    get summary value from visual behavior database
    '''
    behavior_session_uuid = get_uuid(container_df, ophys_container_id, session_number)
    session_stats = get_session_stats(container_df, ophys_container_id, session_number)
    if value == 'session_prefix':
        oeid = get_oeid(container_df, ophys_container_id, session_number)
        return data_loading.get_session_type_for_ophys_experiment_id(oeid)[:7]
    elif value == 'ophys_experiment_id' or value == 'ophys_session_id':
        return db.convert_id({'behavior_session_uuid': behavior_session_uuid}, value)
    else:
        return session_stats[value]


def rewrite_record(uuid):
    pkl_path = db.get_pkl_path(uuid)
    data = pd.read_pickle(pkl_path)

    core_data = data_to_change_detection_core(data)
    trials = create_extended_dataframe(**core_data).drop(columns=['date', 'LDT_mode'])
    summary = summarize.session_level_summary(trials).iloc[0].to_dict()
    summary.update({'error_on_load': 0})

    vb = db.Database('visual_behavior_data')
    db.update_or_create(
        vb['behavior_data']['summary'],
        db.simplify_entry(summary),
        ['behavior_session_uuid'],
        force_write=False
    )
    vb.close()


def load_data():
    container_df = data_loading.build_container_df()
    filtered_container_list = data_loading.get_filtered_ophys_container_ids()  # NOQA F841
    return container_df.query('ophys_container_id in @filtered_container_list')


def populate_xarray(values=['d_prime_peak', 'number_of_licks', 'num_contingent_trials']):
    print('populating xarray...')
    container_df = load_data()
    container_df['line'] = container_df['driver_line'].map(lambda s: ';'.join(s))

    container_df = container_df.sort_values(by=['line', 'targeted_structure', 'first_acquistion_date'])
    ophys_container_ids = container_df.ophys_container_id.values
    sessions = ['session_{}'.format(i) for i in range(6)]

    session_prefixes = []
    for ophys_container_id in ophys_container_ids:
        session_prefixes += [s.split(' ')[0][:7] for s in container_df[container_df['ophys_container_id'] == ophys_container_id][sessions].values[0] if pd.notnull(s)]
    session_prefixes = np.sort(np.unique(np.array(session_prefixes)))

    val_array = xr.DataArray(
        np.zeros((len(ophys_container_ids), len(sessions), len(values))),
        dims=('ophys_container_id', 'session_prefix', 'value'),
        coords={'ophys_container_id': ophys_container_ids, 'session_prefix': session_prefixes, 'value': values}
    ) * np.nan

    for ophys_container_id in ophys_container_ids:
        for session_number in range(6):
            bs_uuid = get_uuid(container_df, ophys_container_id, session_number)
            if pd.notnull(bs_uuid):
                session_stats = get_session_stats(container_df, ophys_container_id, session_number)
                if 'error_on_load' in session_stats.keys() and session_stats['error_on_load'] == 1:
                    rewrite_record(bs_uuid)
                session_prefix = get_value(container_df, ophys_container_id, session_number, 'session_prefix')

                for value in values:
                    v = get_value(container_df, ophys_container_id, session_number, value)
                    val_array.loc[{'ophys_container_id': ophys_container_id, 'session_prefix': session_prefix, 'value': value}] = v

    return val_array


def make_container_overview_plots(values=['d_prime_peak', 'number_of_licks', 'num_contingent_trials']):
    val_array = populate_xarray(values)
    ophys_container_ids = None
    for value in values:
        print('making plot for {}'.format(value))
        fig = go.Figure(
            data=go.Heatmap(
                z=val_array.loc[{'value': value}].values,
                # x=session_prefixes,
                #         y = [str(i) for i in range(len(ophys_container_ids))],
                # y=ophys_container_ids,
                hoverongaps=True,
                colorbar={'title': value},
                colorscale='viridis',
            )
        )

        fig.update_layout(
            autosize=False,
            width=700,
            height=20 * len(ophys_container_ids),
            margin=dict(
                l=0, # NOQA E741
                r=0,
                b=0,
                t=50,
                pad=0
            ),
            xaxis_title='ophys session',
            yaxis_title='container ID',
            title='{} by container ID and ophys session'.format(value)
        )
        fig.update_yaxes(autorange="reversed", type='category', dtick=1)
        fig.update_xaxes(dtick=1)
        fig.write_html("/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/overview_plots/{}_container_overview.html".format(value))


if __name__ == '__main__':
    make_container_overview_plots()
