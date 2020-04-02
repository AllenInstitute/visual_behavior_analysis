#!/usr/bin/env python

import base64
import os
import yaml
import pandas as pd

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
    print('loading new container plot definitions')
    plot_definition_path = "/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/container_plots/plot_definitions.yml"
    container_options = load_yaml(plot_definition_path)
    return container_options


def load_container_overview_plot_options():
    print('loading new container overview plot definitions')
    plot_definition_path = "/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/overview_plots/plot_definitions.yml"
    container_overview_options = load_yaml(plot_definition_path)
    return container_overview_options


def get_container_plot(container_id, plot_type):
    qc_plot_folder = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots'
    container_plot_folder = os.path.join(qc_plot_folder, 'container_plots')

    plot_image_path = os.path.join(
        container_plot_folder,
        plot_type, 'container_{}.png'.format(container_id)
    )
    try:
        encoded_image = base64.b64encode(open(plot_image_path, 'rb').read())
    except FileNotFoundError:
        print('not found')
        plot_not_found_path = os.path.join(
            container_plot_folder,
            'no_cached_plot_small.png'
        )
        encoded_image = base64.b64encode(
            open(plot_not_found_path, 'rb').read())

    return encoded_image
