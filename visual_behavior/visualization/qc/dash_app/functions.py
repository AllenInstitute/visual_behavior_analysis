#!/usr/bin/env python

import base64
import os
import yaml
import json
import pandas as pd
import plotly.graph_objs as go
import datetime

import visual_behavior.visualization.qc.data_loading as dl
from visual_behavior.data_access import loading


def load_data():
    container_df = dl.build_container_df()
    filtered_container_list = dl.get_filtered_ophys_container_ids()  # NOQA F841
    container_subsets = pd.read_csv('/home/dougo/spreadsheet_map.csv')
    res = container_df.query('container_id in @filtered_container_list')
    return res.merge(container_subsets, left_on='container_id', right_on='container_id', how='left')


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