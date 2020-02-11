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

from allensdk.brain_observatory.behavior.behavior_project_cache import BehaviorProjectCache as bpc
import visual_behavior.visualization.qc.data_loading as dl

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'Visual Behavior Data QC'

# app = dash.Dash()
app.config['suppress_callback_exceptions'] = True

def load_data():
    container_df = dl.build_container_df()

    return container_df


table = load_data()

app.layout = html.Div([
    html.H4('Visual Behavior Data'),
    html.H4('  '),
    dash_table.DataTable(
        id='table',
        columns=[{"name": i.replace('_',' '), "id": i} for i in table.columns],
        data=table.to_dict('records'),
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
], className='container')



if __name__ == '__main__':
    app.run_server(debug=True, port=5678, host='0.0.0.0')