#!/usr/bin/env python

import base64
import os
import yaml
import json
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import datetime
import uuid
import sys
from visual_behavior import database as db
from importlib import reload

from visual_behavior.data_access import loading


def load_container_data():
    sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/container_plots')
    import container_data
    reload(container_data)
    container_df = container_data.load_data()
    container_df = container_data.update_data(container_df)
    return container_df


def load_session_data():
    try:
        sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/session_plots')
        import session_data
        reload(session_data)
        session_df = session_data.load_data()
        session_df = session_data.update_data(session_df)
        return session_df
    except Exception as e:
        print("ERROR LOADING SESSION DATA TABLE")
        print(e)
        return pd.DataFrame({'error': [e]})


def load_yaml(yaml_path):
    with open(yaml_path, 'r') as stream:
        yaml_contents = yaml.safe_load(stream)

    options = []
    for k, v in yaml_contents.items():
        options.append({'label': k, 'value': v})
    return options


def get_plot_list(qc_definitions):
    plot_list = []
    for plot_title, attributes in qc_definitions.items():
        if attributes['show_plots']:
            plot_list.append({'label': plot_title, 'value': attributes['plot_folder_name']})
    return plot_list


def load_container_qc_definitions():
    container_qc_definition_path = "/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/container_plots/qc_definitions.json"
    return json.load(open(container_qc_definition_path))


def load_session_qc_definitions():
    container_qc_definition_path = "/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/session_plots/qc_definitions.json"
    return json.load(open(container_qc_definition_path))


def load_container_plot_options():
    container_options = get_plot_list(load_container_qc_definitions())
    print('container_options:')
    print(container_options)
    return container_options


def load_session_plot_options():
    session_options = get_plot_list(load_session_qc_definitions())
    return session_options


def load_container_overview_plot_options():
    plot_definition_path = "/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/overview_plots/plot_definitions.yml"
    container_overview_options = load_yaml(plot_definition_path)
    return container_overview_options


def get_plot_path(_id, plot_type, display_level):
    qc_plot_folder = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots'
    plot_folder = os.path.join(qc_plot_folder, '{}_plots'.format(display_level))

    plot_image_path = os.path.join(
        plot_folder,
        plot_type, '{}_{}.png'.format(display_level, _id)
    )
    return plot_image_path


def get_plot(_id, plot_type, display_level):
    plot_image_path = get_plot_path(_id, plot_type, display_level)
    try:
        encoded_image = base64.b64encode(open(plot_image_path, 'rb').read())
    except FileNotFoundError:
        print('not found, ophys_container_id = {}, plot_type = {}'.format(_id, plot_type))
        qc_plot_folder = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots'
        plot_folder = os.path.join(qc_plot_folder, '{}_plots'.format(display_level))

        plot_not_found_path = os.path.join(
            plot_folder,
            'no_cached_plot_small.png'
        )
        encoded_image = base64.b64encode(
            open(plot_not_found_path, 'rb').read())

    return encoded_image


def generate_plot_inventory():
    global CONTAINER_TABLE

    CONTAINER_TABLE = load_container_data().sort_values('first_acquistion_date')
    container_table = CONTAINER_TABLE
    plots = load_container_plot_options()
    list_of_dicts = []
    for ophys_container_id in container_table['ophys_container_id'].values:
        d = {'ophys_container_id': ophys_container_id}
        for entry in plots:
            plot_type = entry['value']
            d.update({plot_type: os.path.exists(get_plot_path(ophys_container_id, plot_type, 'container'))})
        list_of_dicts.append(d)
    return pd.DataFrame(list_of_dicts).set_index('ophys_container_id').sort_index()


def make_plot_inventory_heatmap(plot_inventory):
    fig = go.Figure(
        data=go.Heatmap(
            z=plot_inventory.values.astype(int),
            x=plot_inventory.columns,
            y=plot_inventory.index,
            hoverongaps=True,
            showscale=False,
            colorscale='inferno',
            xgap=3,
            ygap=3,
        )
    )

    timestamp = datetime.datetime.now()
    timestamp_string = 'last updated on {} @ {}'.format(timestamp.strftime('%D'), timestamp.strftime('%H:%M:%S'))

    fig.update_layout(
        autosize=False,
        width=1000,
        height=3000,
        margin=dict(
            l=0,  # NOQA E741
            r=0,
            b=0,
            t=50,
            pad=0
        ),
        xaxis_title='plot type',
        yaxis_title='container ID',
        title='Plot Inventory (black = missing) {}'.format(timestamp_string)
    )
    fig.update_yaxes(autorange="reversed", type='category', dtick=1)
    fig.update_xaxes(dtick=1)

    return fig


def get_motion_corrected_movie_path(oeid):
    query = '''
    SELECT wkft.name, wkf.storage_directory || wkf.filename as path
    FROM well_known_files AS wkf
    JOIN well_known_file_types AS wkft on wkft.id=wkf.well_known_file_type_id
    JOIN ophys_experiments AS oe on wkf.attachable_id=oe.id
    WHERe oe.id={}
    '''
    try:
        return db.lims_query(query.format(oeid)).set_index('name').loc['OphysMotionPreview']['path']
    except Exception as e:
        return e


def get_motion_corrected_movie_paths(ophys_container_id):
    et = loading.get_filtered_ophys_experiment_table().reset_index()
    paths = []
    for oeid in et.query('ophys_container_id == @ophys_container_id').sort_values(by='ophys_experiment_id')['ophys_experiment_id']:
        paths.append(
            '/' + get_motion_corrected_movie_path(oeid)
        )
    return paths


def print_motion_corrected_movie_paths(ophys_container_id):
    et = loading.get_filtered_ophys_experiment_table().reset_index()
    lines = []
    for oeid in et.query('ophys_container_id == @ophys_container_id')['ophys_experiment_id']:
        movie_path = loading.get_motion_corrected_movie_h5_location(oeid).replace('motion_corrected_video.h5', 'motion_preview.10x.mp4')
        lines.append('ophys experiment ID = {}\n'.format(oeid))
        lines.append("LINUX PATH:")
        lines.append('\t<a href="url">{}</a>'.format(movie_path))
        lines.append('WINDOWS PATH')
        lines.append('\t{}'.format(movie_path.replace('/', '\\')))
        lines.append('')
    return '\n'.join(lines)


def to_json(data_to_log, display_level):
    '''log data to filesystem'''
    saveloc = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_records/{}_level'.format(display_level)
    filename = os.path.join(saveloc, '{}.json'.format(data_to_log['_id']))
    json.dump(data_to_log, open(filename, 'w' ))


def to_mongo(data_to_log, display_level):
    '''log data to mongo'''
    conn = db.Database('visual_behavior_data')
    collection = conn['ophys_qc']['{}_qc_records'.format(display_level)]
    collection.insert_one(db.clean_and_timestamp(data_to_log))
    conn.close()


def log_feedback(feedback, display_level):
    '''logs feedback from app to mongo and filesystem'''
    if pd.notnull(feedback['timestamp']):
        random_id = uuid.uuid4().hex
        feedback.update({'_id': random_id})
        to_json(feedback, display_level)
        to_mongo(feedback, display_level)


# def get_roi_overlap_plots_links(session_id, plots_dir='/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/single_cell_plots/mesoscope_decrosstalk'):
#     """
#     function to build links ot the roi-level plots given session_id
#     session_id : int, session ID form lims
#     plots_dir: str, path to outer directory
#     returns dict, where {'pair_0_overlaps' : "path_to_roi_level_dir"}
#     """
#     session_path = os.path.join(plots_dir, f"session_{session_id}")
#     roi_links = {}
#     pairs = get_paired_planes(session_id)
#     for i, pair in enumerate(pairs):
#         pair_dir_path = os.path.join(session_path, f"pair_{i}_overlaps")
#         if os.path.isdir(pair_dir_path):
#             roi_links[f'pair_{i}'] = pair_dir_path
#         else:
#             roi_links[f'pair_{i}'] = "roi level plots don't exist"
#     return roi_links


def get_experiment_ids_for_session_id(ophys_session_id):
    query = 'select * from ophys_experiments where ophys_session_id = {}'
    experiments_table = db.lims_query(query.format(ophys_session_id))
    return np.sort(experiments_table['id'].tolist())
