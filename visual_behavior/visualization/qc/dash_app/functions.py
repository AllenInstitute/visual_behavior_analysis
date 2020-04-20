#!/usr/bin/env python

import base64
import os
import yaml
import pandas as pd
import plotly.graph_objs as go
import datetime

import visual_behavior.visualization.qc.data_loading as dl


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


def load_container_plot_options():
    plot_definition_path = "/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/container_plots/plot_definitions.yml"
    container_options = load_yaml(plot_definition_path)
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
