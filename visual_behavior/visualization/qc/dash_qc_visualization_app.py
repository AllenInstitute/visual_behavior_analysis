#!/usr/bin/env python

import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
import numpy as np
import datetime
import pytz
import platform
from flask import send_file
import datetime
import base64
import os

from allensdk.brain_observatory.behavior.behavior_project_cache import BehaviorProjectCache as bpc
import visual_behavior.visualization.qc.data_loading as dl

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app = dash.Dash(__name__,)
app.title = 'Visual Behavior Data QC'

# app = dash.Dash()
app.config['suppress_callback_exceptions'] = True

def load_data():
    container_df = dl.build_container_df()
    filtered_container_list = dl.get_filtered_ophys_container_ids()
    return container_df.query('container_id in @filtered_container_list')

container_table = load_data()

## components



dropdown = dcc.Dropdown(
    id='dropdown',
    options=[
        {'label': 'Dummy Plot 1', 'value': 'dummy_plot_1'},
        {'label': 'Dummy Plot 2', 'value': 'dummy_plot_2'},
        {'label': 'Max Intensity Projection', 'value': 'max_intensity_projection'},
        {'label': 'Segmentation Mask Overlays', 'value': 'segmentation_mask_overlays'},
        {'label': 'Segmentation Masks', 'value': 'segmentation_masks'},
    ],
    value=['dummy_plot_1','dummy_plot_2'],
    multi=True
)

app.layout = html.Div([
    html.H4('Visual Behavior Data'),
    html.H4('  '),
    dash_table.DataTable(
        id='data_table',
        columns=[{"name": i.replace('_',' '), "id": i} for i in container_table.columns],
        data=container_table.to_dict('records'),
        row_selectable="single",
        selected_rows=[0],
        page_size=15,
        filter_action='native',
        style_header={
            'backgroundColor': 'white',
            'fontWeight': 'bold',
            'textAlign': 'center',
        },
        style_table={'overflowX': 'scroll'},
    ),
    html.H4('  '),
    dropdown,
    html.H4('  '),
    html.Img(
        id='image_frame_1',
        style={'width': '1500px'}
    ),
    html.Img(
        id='image_frame_2',
        style={'width': '1500px'}
    ),
    html.Img(
        id='image_frame_3',
        style={'width': '1500px'}
    ),
], className='container')

def get_container_plot(container_id, plot_type):
    qc_plot_folder = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots'
    container_plot_folder = os.path.join(qc_plot_folder,'container_plots')

    plot_image_path = os.path.join(
        container_plot_folder,
        plot_type,'container_{}.png'.format(container_id)
    )
    print('frame path = {}'.format(plot_image_path))
    try:
        encoded_image = base64.b64encode(open(plot_image_path, 'rb').read())
    except FileNotFoundError:
        print('not found')
        encoded_image = base64.b64encode(
            open('//olsenlab1/data/support_files/no_processed_data.png', 'rb').read())

    return encoded_image


## first image frame
@app.callback(Output('image_frame_1', 'src'),
              [Input('data_table', 'selected_rows'),
              Input('dropdown','value'),
               ])
def update_frame_1(row_index,plot_types):
    if len(plot_types) >= 1:
        plot_type = plot_types[0]
        data_table_row = container_table.iloc[row_index[0]]
        container_id = data_table_row['container_id']
        encoded_image = get_container_plot(container_id, plot_type=plot_type)
        return 'data:image/png;base64,{}'.format(encoded_image.decode())

## second image frame
@app.callback(Output('image_frame_2', 'src'),
              [Input('data_table', 'selected_rows'),
              Input('dropdown','value'),
               ])
def update_frame_2(row_index,plot_types):
    if len(plot_types) >= 2:
        plot_type = plot_types[1]
        data_table_row = container_table.iloc[row_index[0]]
        container_id = data_table_row['container_id']
        encoded_image = get_container_plot(container_id, plot_type=plot_type)
        return 'data:image/png;base64,{}'.format(encoded_image.decode())


if __name__ == '__main__':
    app.run_server(debug=True, port=5678, host='0.0.0.0')