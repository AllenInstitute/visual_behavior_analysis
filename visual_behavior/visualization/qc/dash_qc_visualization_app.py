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


# GLOBAL CONSTANTS:
ENTRIES_PER_PAGE = 5

# APP SETUP
app = dash.Dash(__name__,)
app.title = 'Visual Behavior Data QC'

app.config['suppress_callback_exceptions'] = True

# FUNCTIONS


def load_data():
    container_df = dl.build_container_df()
    filtered_container_list = dl.get_filtered_ophys_container_ids()
    return container_df.query('container_id in @filtered_container_list')


container_table = load_data().sort_values('first_acquistion_date')

# COMPONENTS
dropdown = dcc.Dropdown(
    id='dropdown',
    options=[
        {'label': 'Average Images', 'value': 'average_images'},
        {'label': 'Average Intensity Timeseries', 'value': 'average_intensity_timeseries'},
        {'label': 'Dff Traces Heatmaps', 'value': 'dff_traces_heatmaps'},
        {'label': 'Eyetracking Sample Frames', 'value': 'eyetracking_sample_frames'},
        {'label': 'Fraction Matched Cells', 'value': 'fraction_matched_cells'},
        {'label': 'Lick Rasters', 'value': 'lick_rasters'},
        {'label': 'Max Intensity Projection', 'value': 'max_intensity_projection'},
        {'label': 'Motion Correction XY Shift', 'value': 'motion_correction_xy_shift'},
        {'label': 'Number Matched Cells', 'value': 'number_matched_cells'},
        {'label': 'Ophys Session Sequence', 'value': 'ophys_session_sequence'},
        {'label': 'PMT Gain', 'value': 'PMT_gain'},
        {'label': 'Running Speed', 'value': 'running_speed'},
        {'label': 'Segmentation Mask Overlays', 'value': 'segmentation_mask_overlays'},
        {'label': 'Segmentation Masks', 'value': 'segmentation_masks'},
    ],
    value=['ophys_session_sequence'],
    multi=True
)

app.layout = html.Div([
    html.H4('Visual Behavior Data'),
    html.H4('  '),
    dash_table.DataTable(
        id='data_table',
        columns=[{"name": i.replace('_', ' '), "id": i} for i in container_table.columns],
        data=container_table.to_dict('records'),
        row_selectable="single",
        selected_rows=[0],
        page_size=ENTRIES_PER_PAGE,
        filter_action='native',
        style_header={
            'backgroundColor': 'white',
            'fontWeight': 'bold',
            'textAlign': 'center',
            'whiteSpace': 'normal',
        },
        style_data={
            'whiteSpace': 'normal',
            'textAlign': 'center',
            'height': 'auto'
        },
        style_table={'overflowX': 'scroll'},
    ),
    html.H4('Select plots to generate from the dropdown (max 10)'),
    dropdown,
    html.H4(id='plot_title_0', children=''),
    html.Img(
        id='image_frame_0',
        style={'width': '1500px'}
    ),
    html.H4(id='plot_title_1', children=''),
    html.Img(
        id='image_frame_1',
        style={'width': '1500px'}
    ),
    html.H4(id='plot_title_2', children=''),
    html.Img(
        id='image_frame_2',
        style={'width': '1500px'}
    ),
    html.H4(id='plot_title_3', children=''),
    html.Img(
        id='image_frame_3',
        style={'width': '1500px'}
    ),
    html.H4(id='plot_title_4', children=''),
    html.Img(
        id='image_frame_4',
        style={'width': '1500px'}
    ),
    html.H4(id='plot_title_5', children=''),
    html.Img(
        id='image_frame_5',
        style={'width': '1500px'}
    ),
    html.H4(id='plot_title_6', children=''),
    html.Img(
        id='image_frame_6',
        style={'width': '1500px'}
    ),
    html.H4(id='plot_title_7', children=''),
    html.Img(
        id='image_frame_7',
        style={'width': '1500px'}
    ),
    html.H4(id='plot_title_8', children=''),
    html.Img(
        id='image_frame_8',
        style={'width': '1500px'}
    ),
    html.H4(id='plot_title_9', children=''),
    html.Img(
        id='image_frame_9',
        style={'width': '1500px'}
    ),
], className='container')


def get_container_plot(container_id, plot_type):
    qc_plot_folder = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots'
    container_plot_folder = os.path.join(qc_plot_folder, 'container_plots')

    plot_image_path = os.path.join(
        container_plot_folder,
        plot_type, 'container_{}.png'.format(container_id)
    )
    print('frame path = {}'.format(plot_image_path))
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

# highlight row in data table


@app.callback(Output('data_table', 'style_data_conditional'),
              [Input('data_table', 'selected_rows'),
               Input('data_table', 'page_current'),
               Input('data_table', 'derived_viewport_indices')
               ])
def highlight_row(row_index, page_current, derived_viewport_indices):
    # row index is None on the very first call. This avoids an error:
    if row_index is None or derived_viewport_indices is None:
        index_to_highlight = 0
    elif row_index[0] in derived_viewport_indices:
        index_to_highlight = derived_viewport_indices.index(row_index[0])
    else:
        index_to_highlight = 1e6

    print('row_index = {}, page_current = {}, index_to_highlight = {}'.format(row_index, page_current, index_to_highlight))
    print('derived_viewport_indices: {}'.format(derived_viewport_indices))
    style_data_conditional = [{
        "if": {"row_index": index_to_highlight},
        "backgroundColor": "#3D9970",
        'color': 'white'
    }]
    return style_data_conditional

# set plot titles


@app.callback(Output('plot_title_0', 'children'),
              [Input('dropdown', 'value'),
               ])
def update_frame(plot_types):
    if len(plot_types) >= 1:
        return plot_types[0]


@app.callback(Output('plot_title_1', 'children'),
              [Input('dropdown', 'value'),
               ])
def update_frame(plot_types):
    if len(plot_types) >= 2:
        return plot_types[1]


@app.callback(Output('plot_title_2', 'children'),
              [Input('dropdown', 'value'),
               ])
def update_frame(plot_types):
    if len(plot_types) >= 3:
        return plot_types[2]


@app.callback(Output('plot_title_3', 'children'),
              [Input('dropdown', 'value'),
               ])
def update_frame(plot_types):
    if len(plot_types) >= 4:
        return plot_types[3]


@app.callback(Output('plot_title_4', 'children'),
              [Input('dropdown', 'value'),
               ])
def update_frame(plot_types):
    if len(plot_types) >= 5:
        return plot_types[4]


@app.callback(Output('plot_title_5', 'children'),
              [Input('dropdown', 'value'),
               ])
def update_frame(plot_types):
    if len(plot_types) >= 6:
        return plot_types[5]


@app.callback(Output('plot_title_6', 'children'),
              [Input('dropdown', 'value'),
               ])
def update_frame(plot_types):
    if len(plot_types) >= 7:
        return plot_types[6]


@app.callback(Output('plot_title_7', 'children'),
              [Input('dropdown', 'value'),
               ])
def update_frame(plot_types):
    if len(plot_types) >= 8:
        return plot_types[7]


@app.callback(Output('plot_title_8', 'children'),
              [Input('dropdown', 'value'),
               ])
def update_frame(plot_types):
    if len(plot_types) >= 9:
        return plot_types[8]


@app.callback(Output('plot_title_9', 'children'),
              [Input('dropdown', 'value'),
               ])
def update_frame(plot_types):
    if len(plot_types) >= 10:
        return plot_types[9]

# image frames callbacks
# (I can't figure out how to make these in a loop!)


@app.callback(Output('image_frame_0', 'src'),
              [Input('data_table', 'selected_rows'),
               Input('dropdown', 'value'),
               ])
def update_frame(row_index, plot_types):
    if len(plot_types) >= 1:
        plot_type = plot_types[0]
        container_id = container_table.iloc[row_index[0]]['container_id']
        encoded_image = get_container_plot(container_id, plot_type=plot_type)
        return 'data:image/png;base64,{}'.format(encoded_image.decode())


@app.callback(Output('image_frame_1', 'src'),
              [Input('data_table', 'selected_rows'),
               Input('dropdown', 'value'),
               ])
def update_frame(row_index, plot_types):
    if len(plot_types) >= 2:
        plot_type = plot_types[1]
        container_id = container_table.iloc[row_index[0]]['container_id']
        encoded_image = get_container_plot(container_id, plot_type=plot_type)
        return 'data:image/png;base64,{}'.format(encoded_image.decode())


@app.callback(Output('image_frame_2', 'src'),
              [Input('data_table', 'selected_rows'),
               Input('dropdown', 'value'),
               ])
def update_frame(row_index, plot_types):
    if len(plot_types) >= 3:
        plot_type = plot_types[2]
        container_id = container_table.iloc[row_index[0]]['container_id']
        encoded_image = get_container_plot(container_id, plot_type=plot_type)
        return 'data:image/png;base64,{}'.format(encoded_image.decode())


@app.callback(Output('image_frame_3', 'src'),
              [Input('data_table', 'selected_rows'),
               Input('dropdown', 'value'),
               ])
def update_frame(row_index, plot_types):
    if len(plot_types) >= 4:
        plot_type = plot_types[3]
        container_id = container_table.iloc[row_index[0]]['container_id']
        encoded_image = get_container_plot(container_id, plot_type=plot_type)
        return 'data:image/png;base64,{}'.format(encoded_image.decode())


@app.callback(Output('image_frame_4', 'src'),
              [Input('data_table', 'selected_rows'),
               Input('dropdown', 'value'),
               ])
def update_frame(row_index, plot_types):
    if len(plot_types) >= 5:
        plot_type = plot_types[4]
        container_id = container_table.iloc[row_index[0]]['container_id']
        encoded_image = get_container_plot(container_id, plot_type=plot_type)
        return 'data:image/png;base64,{}'.format(encoded_image.decode())


@app.callback(Output('image_frame_5', 'src'),
              [Input('data_table', 'selected_rows'),
               Input('dropdown', 'value'),
               ])
def update_frame(row_index, plot_types):
    if len(plot_types) >= 6:
        plot_type = plot_types[5]
        container_id = container_table.iloc[row_index[0]]['container_id']
        encoded_image = get_container_plot(container_id, plot_type=plot_type)
        return 'data:image/png;base64,{}'.format(encoded_image.decode())


@app.callback(Output('image_frame_6', 'src'),
              [Input('data_table', 'selected_rows'),
               Input('dropdown', 'value'),
               ])
def update_frame(row_index, plot_types):
    if len(plot_types) >= 7:
        plot_type = plot_types[6]
        container_id = container_table.iloc[row_index[0]]['container_id']
        encoded_image = get_container_plot(container_id, plot_type=plot_type)
        return 'data:image/png;base64,{}'.format(encoded_image.decode())


@app.callback(Output('image_frame_7', 'src'),
              [Input('data_table', 'selected_rows'),
               Input('dropdown', 'value'),
               ])
def update_frame(row_index, plot_types):
    if len(plot_types) >= 8:
        plot_type = plot_types[7]
        container_id = container_table.iloc[row_index[0]]['container_id']
        encoded_image = get_container_plot(container_id, plot_type=plot_type)
        return 'data:image/png;base64,{}'.format(encoded_image.decode())


@app.callback(Output('image_frame_8', 'src'),
              [Input('data_table', 'selected_rows'),
               Input('dropdown', 'value'),
               ])
def update_frame(row_index, plot_types):
    if len(plot_types) >= 9:
        plot_type = plot_types[8]
        container_id = container_table.iloc[row_index[0]]['container_id']
        encoded_image = get_container_plot(container_id, plot_type=plot_type)
        return 'data:image/png;base64,{}'.format(encoded_image.decode())


@app.callback(Output('image_frame_9', 'src'),
              [Input('data_table', 'selected_rows'),
               Input('dropdown', 'value'),
               ])
def update_frame(row_index, plot_types):
    if len(plot_types) >= 10:
        plot_type = plot_types[9]
        container_id = container_table.iloc[row_index[0]]['container_id']
        encoded_image = get_container_plot(container_id, plot_type=plot_type)
        return 'data:image/png;base64,{}'.format(encoded_image.decode())


if __name__ == '__main__':
    app.run_server(debug=True, port=5678, host='0.0.0.0')
