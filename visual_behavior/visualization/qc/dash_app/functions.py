#!/usr/bin/env python

import base64
import os
import yaml
import json
import pandas as pd
import plotly.graph_objs as go
import datetime
import uuid
from visual_behavior import database as db

import visual_behavior.visualization.qc.data_loading as dl
from visual_behavior.data_access import loading


def load_data():
    container_df = dl.build_container_df()
    return container_df

def load_yaml(yaml_path):
    with open(yaml_path, 'r') as stream:
        yaml_contents = yaml.safe_load(stream)

    options = []
    for k, v in yaml_contents.items():
        options.append({'label': k, 'value': v})
    return options


def get_plot_list(container_qc_definitions):
    plot_list = []
    for plot_title, attributes in container_qc_definitions.items():
        if attributes['show_plots']:
            plot_list.append({'label': plot_title, 'value': attributes['plot_folder_name']})
    return plot_list


def load_container_qc_definitions():
    container_qc_definition_path = "/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/container_plots/qc_definitions.json"
    return json.load(open(container_qc_definition_path))


def load_container_plot_options():
    container_options = get_plot_list(load_container_qc_definitions())
    print('container_options:')
    print(container_options)
    return container_options


def load_container_overview_plot_options():
    plot_definition_path = "/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/overview_plots/plot_definitions.yml"
    container_overview_options = load_yaml(plot_definition_path)
    return container_overview_options


def get_container_plot_path(container_id, plot_type):
    qc_plot_folder = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots'
    container_plot_folder = os.path.join(qc_plot_folder, 'container_plots')

    plot_image_path = os.path.join(
        container_plot_folder,
        plot_type, 'container_{}.png'.format(container_id)
    )
    return plot_image_path


def get_container_plot(container_id, plot_type):
    plot_image_path = get_container_plot_path(container_id, plot_type)
    try:
        encoded_image = base64.b64encode(open(plot_image_path, 'rb').read())
    except FileNotFoundError:
        print('not found, container_id = {}, plot_type = {}'.format(container_id, plot_type))
        qc_plot_folder = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots'
        container_plot_folder = os.path.join(qc_plot_folder, 'container_plots')

        plot_not_found_path = os.path.join(
            container_plot_folder,
            'no_cached_plot_small.png'
        )
        encoded_image = base64.b64encode(
            open(plot_not_found_path, 'rb').read())

    return encoded_image


CONTAINER_TABLE = load_data().sort_values('first_acquistion_date')


def generate_plot_inventory():
    global CONTAINER_TABLE
    container_table = CONTAINER_TABLE
    plots = load_container_plot_options()
    list_of_dicts = []
    for container_id in container_table['container_id'].values:
        d = {'container_id': container_id}
        for entry in plots:
            plot_type = entry['value']
            d.update({plot_type: os.path.exists(get_container_plot_path(container_id, plot_type))})
        list_of_dicts.append(d)
    return pd.DataFrame(list_of_dicts).set_index('container_id').sort_index()


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
            l=0, # NOQA E741
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

def get_motion_corrected_movie_paths(container_id):
    et = loading.get_filtered_ophys_experiment_table().reset_index()
    paths = []
    for oeid in et.query('container_id == @container_id')['ophys_experiment_id']:
        paths.append(
            loading.get_motion_corrected_movie_h5_location(oeid).replace(
                'motion_corrected_video.h5',
                'motion_preview.10x.mp4'
            )
        )
    return paths

def print_motion_corrected_movie_paths(container_id):
    et = loading.get_filtered_ophys_experiment_table().reset_index()
    lines = []
    for oeid in et.query('container_id == @container_id')['ophys_experiment_id']:
        movie_path = loading.get_motion_corrected_movie_h5_location(oeid).replace('motion_corrected_video.h5','motion_preview.10x.mp4')
        lines.append('ophys experiment ID = {}\n'.format(oeid))
        lines.append("LINUX PATH:")
        lines.append('\t<a href="url">{}</a>'.format(movie_path))
        lines.append('WINDOWS PATH')
        lines.append('\t{}'.format(movie_path.replace('/','\\')))
        lines.append('')
    return '\n'.join(lines)

def to_json(data_to_log):
    '''log data to filesystem'''
    saveloc = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_records'
    filename = os.path.join(saveloc, '{}.json'.format(data_to_log['_id']))
    json.dump(data_to_log, open(filename, 'w' ))

def to_mongo(data_to_log):
    '''log data to mongo'''
    conn = db.Database('visual_behavior_data')
    collection = conn['ophys_qc']['container_qc_records']
    collection.insert_one(db.clean_and_timestamp(data_to_log))
    conn.close()

def log_feedback(feedback):
    '''logs feedback from app to mongo and filesystem'''
    if pd.notnull(feedback['timestamp']):
        random_id = uuid.uuid4().hex
        feedback.update({'_id':random_id})
        to_json(feedback)
        to_mongo(feedback)

def update_qc_status(feedback):
    '''
    updates:
        motion_correction_has_qc to True if feedback['qc_attribute'] == 'Motion Correction'
        cell_matching_has_qc to True if feedback['qc_attribute'] == 'Nway Production Warp Summary'
    '''
    if feedback['qc_attribute'] == 'Motion Correction':
        conn = db.Database('visual_behavior_data')
        collection = conn['ophys_qc']['container_qc_records']
        collection.update_or_create(db.clean_and_timestamp({}))
        conn.close()

def get_qcd_oeids(container_id, qc_attribute):
    '''
    get all experiment IDS that have been QC'd for a given container_id and attribute
    '''
    conn = db.Database('visual_behavior_data')
    collection = conn['ophys_qc']['container_qc_records']
    res = pd.DataFrame(list(collection.find({'container_id':container_id, 'qc_attribute':qc_attribute})))
    conn.close

    if len(res) > 0:
        oeids = [item for sublist in res['experiment_ids'] for item in sublist]
        return oeids
    else:
        return []


def qc_for_all_experiments(container_id, qc_attribute):
    '''check to see that all experiments for a given container have been QCd'''
    oeids_with_qc = set(get_qcd_oeids(container_id, qc_attribute = qc_attribute))
    oeids = set(loading.get_filtered_ophys_experiment_table().query('container_id == @container_id').reset_index()['ophys_experiment_id'].values)
    
    # symmetric_difference is an empty set if all oeids are in oeids_with_qc
    return len(oeids.symmetric_difference(oeids_with_qc)) == 0


def set_qc_complete_flags(feedback):
    container_id = feedback['container_id']
    qc_attribute = feedback['qc_attribute']
    
    entry = None
    try:
        if qc_for_all_experiments(container_id, qc_attribute):
            if qc_attribute == 'Motion Correction':
                entry = {
                        'container_id':container_id,
                        'motion_correction_has_qc':True
                }
            elif qc_attribute == 'Nway Production Warp Summary':
                entry = {
                        'container_id':container_id,
                        'cell_matching_has_qc':True
                }
    except ValueError:
        pass

    if entry:
        conn = db.Database('visual_behavior_data')
        collection = conn['ophys_qc']['container_qc']
        db.update_or_create(collection, db.clean_and_timestamp(entry), keys_to_check=['container_id'])
        conn.close()

        print('Updating qc state for {}'.format(qc_attribute))