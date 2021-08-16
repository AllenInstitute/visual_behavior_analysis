#!/usr/bin/env python

import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import argparse
import numpy as np
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from visual_behavior.data_access import loading

import time
import datetime
import functions
import components

# APP SETUP
# app = dash.Dash(__name__,)
app = dash.Dash(external_stylesheets=[dbc.themes.SPACELAB])
app.title = 'Visual Behavior Data QC'
# app.config['suppress_callback_exceptions'] = True

# FUNCTION CALLS
print('setting up table')
t0 = time.time()
container_table = functions.load_container_data().sort_values('first_acquistion_date')
container_plot_options = functions.load_container_plot_options()
session_plot_options = functions.load_session_plot_options()
container_overview_plot_options = functions.load_container_overview_plot_options()
plot_inventory = functions.generate_plot_inventory()
plot_inventory_fig = functions.make_plot_inventory_heatmap(plot_inventory)
experiment_table = loading.get_filtered_ophys_experiment_table().reset_index()
session_table = functions.load_session_data()
print('done setting up table, it took {} seconds'.format(time.time() - t0))

# COMPONENT SETUP
print('setting up components')
t0 = time.time()
components.plot_selection_dropdown.options = container_plot_options
components.container_overview_dropdown.options = container_overview_plot_options
components.container_overview_iframe.src = app.get_asset_url('qc_plots/overview_plots/d_prime_container_overview.html')
components.plot_inventory_iframe.src = 'https://dougollerenshaw.github.io/figures_to_share/container_plot_inventory.html'  # app.get_asset_url('qc_plots/container_plot_inventory.html')
components.data_table.columns = [{"name": i.replace('_', ' '), "id": i} for i in container_table.columns]
components.data_table.data = container_table.to_dict('records')
print('done setting up components, it took {} seconds'.format(time.time() - t0))

app.layout = html.Video(src='/static/my-video.webm')

server = app.server


# APP LAYOUT
app.layout = html.Div(
    [
        html.H3('Visual Behavior Data QC Viewer'),
        # checklist for components to show
        html.Div([components.show_overview_checklist], style={'display': 'none'}),
        components.plot_inventory_graph_div,
        # container level dropdown
        components.container_overview_dropdown,
        # frame with container level plots
        components.container_overview_iframe,
        components.plot_inventory_iframe,
        html.H4('Find container from experiment ID:'),
        html.Label('Enter experiment ID:'),
        dcc.Input(id='experiment_id_entry', placeholder=''),
        html.Label('|  Corresponding Container ID:  '),
        html.Output(id='container_id_output', children=''),
        html.H4('Choose how to organize data:'),
        components.display_level_selection,
        html.H4('Choose preferred path style:'),
        components.path_style,
        html.H4('Data Table:'),
        html.I('Adjust number of rows to display in the data table:'),
        components.table_row_selection,
        components.reload_data_button,
        components.update_data_button,
        # data table
        components.data_table,
        # dropdown for plot selection
        components.previous_button,
        components.next_button,
        html.Div(id='stored_feedback', style={'display': 'none'}),
        html.Div(
            [
                html.H4(children='Links to motion corrected movies for this container'),
                dcc.Link(id='link_0', children='', href='', style={'display': True}, target="_blank"),
                html.H4(''),
                dcc.Link(id='link_1', children='', href='', style={'display': True}, target="_blank"),
                html.H4(''),
                dcc.Link(id='link_2', children='', href='', style={'display': True}, target="_blank"),
                html.H4(''),
                dcc.Link(id='link_3', children='', href='', style={'display': True}, target="_blank"),
                html.H4(''),
                dcc.Link(id='link_4', children='', href='', style={'display': True}, target="_blank"),
                html.H4(''),
                dcc.Link(id='link_5', children='', href='', style={'display': True}, target="_blank"),
                html.H4(''),
                dcc.Link(id='link_6', children='', href='', style={'display': True}, target="_blank"),
                html.H4(''),
                dcc.Link(id='link_7', children='', href='', style={'display': True}, target="_blank"),
                html.H4(''),
                dcc.Link(id='link_8', children='', href='', style={'display': True}, target="_blank"),
                html.H4(''),
                dcc.Link(id='link_9', children='', href='', style={'display': True}, target="_blank"),
            ],
            id='motion_correction_links',
            style={'display': True}
        ),
        html.Div(
            [
                html.H4(id='session_header', children='Associated Experiment IDs'),
                html.P(id='exp_link_0', children=''),
                html.P(id='exp_link_1', children=''),
                html.P(id='exp_link_2', children=''),
                html.P(id='exp_link_3', children=''),
                html.P(id='exp_link_4', children=''),
                html.P(id='exp_link_5', children=''),
                html.P(id='exp_link_6', children=''),
                html.P(id='exp_link_7', children=''),
            ],
            id='experiment_links',
            style={'display': 'none'}
        ),
        components.feedback_button,
        html.H4('Select plots to generate from the dropdown (max 10)'),
        components.plot_selection_dropdown,
        components.plot_titles[0],
        components.plot_frames[0],
        components.plot_titles[1],
        components.plot_frames[1],
        components.plot_titles[2],
        components.plot_frames[2],
        components.plot_titles[3],
        components.plot_frames[3],
        components.plot_titles[4],
        components.plot_frames[4],
        components.plot_titles[5],
        components.plot_frames[5],
        components.plot_titles[6],
        components.plot_frames[6],
        components.plot_titles[7],
        components.plot_frames[7],
        components.plot_titles[8],
        components.plot_frames[8],
        components.plot_titles[9],
        components.plot_frames[9],
        components.plot_titles[10],
        components.plot_frames[10],
        components.plot_titles[11],
        components.plot_frames[11],
        components.plot_titles[12],
        components.plot_frames[12],
        components.plot_titles[13],
        components.plot_frames[13],
        components.plot_titles[14],
        components.plot_frames[14],
        components.plot_titles[15],
        components.plot_frames[15],
        components.plot_titles[16],
        components.plot_frames[16],
        components.plot_titles[17],
        components.plot_frames[17],
        components.plot_titles[18],
        components.plot_frames[18],
        components.plot_titles[19],
        components.plot_frames[19],
    ],
    className='container',
    style={
        # 'padding': '10px',
        'margin-left': '10px',
        'margin-right': '10px',
        'margin-top': '10px',
        'margin-bottom': '10px',
    },
)

# update data table


@app.callback(Output('data_table', 'data'),
              [
                  Input('display_level_selection', 'value'),
                  Input('data_table', 'selected_rows'),
                  Input('feedback_popup_ok', 'n_clicks'),
                  Input("stored_feedback", "children"),
                  Input('update_data_button', 'n_clicks'),
                  Input('reload_data_button', 'n_clicks'),
]
)
def update_data(data_display_level, selected_rows, n_clicks, stored_feedback, update_data_nclicks, reload_data_nclicks):
    print('updating data table at {}'.format(time.time()))
    print('data_display_level = {}'.format(data_display_level))
    if data_display_level == 'container':
        container_table = functions.load_container_data().sort_values('first_acquistion_date')
        data = container_table.to_dict('records')
    elif data_display_level == 'session':
        # session_table = functions.load_session_data()
        data_to_display = functions.load_session_data()
        print('casting exps to string!!!!!!!')
        data_to_display = data_to_display.drop(columns=['ophys_experiment_ids, paired'])
        data = data_to_display.to_dict('records')
    return data

# update data table columns


@app.callback(Output('data_table', 'columns'),
              [Input('display_level_selection', 'value'), ]
              )
def update_data_columns(data_display_level):
    if data_display_level == 'container':
        columns = [{"name": i.replace('_', ' '), "id": i} for i in container_table.columns]
    elif data_display_level == 'session':
        data_to_display = functions.load_session_data()
        data_to_display = data_to_display.drop(columns=['ophys_experiment_ids, paired'])
        columns = [{"name": i.replace('_', ' '), "id": i} for i in data_to_display.columns]
    return columns

# toggle motion correction link visibility


@app.callback(Output('motion_correction_links', 'style'),
              [Input('display_level_selection', 'value'), ]
              )
def display_motion_correction_links(data_display_level):
    if data_display_level == 'container':
        return {'display': True}
    elif data_display_level == 'session':
        return {'display': 'none'}

# toggle experiment ID link visibility


@app.callback(Output('experiment_links', 'style'),
              [Input('display_level_selection', 'value'), ]
              )
def display_experiment_id_links(data_display_level):
    if data_display_level == 'container':
        return {'display': 'none'}
    elif data_display_level == 'session':
        return {'display': True}


# ensure that the table page is set to show the current selection
@app.callback(
    Output("data_table", "page_current"),
    [Input('data_table', 'selected_rows')],
    [
        State('data_table', 'derived_virtual_indices'),
        State('data_table', 'page_current'),
        State('data_table', 'page_size'),
    ]
)
def get_on_correct_page(selected_rows, derived_virtual_indices, page_current, page_size):
    current_selection = selected_rows[0]
    current_index = derived_virtual_indices.index(current_selection)
    current_page = int(current_index / page_size)
    return current_page


@app.callback(
    Output("container_id_output", "children"),
    [Input("experiment_id_entry", "value")],
)
def look_up_container(oeid):
    try:
        res = experiment_table.query('ophys_experiment_id == @oeid')
        if len(res) == 0:
            return 'Not Found'
        else:
            return res.iloc[0]['ophys_container_id']
    except ValueError:
        return ''


# go to previous selection in table
@app.callback(
    Output("data_table", "selected_rows"),
    [
        Input("next_button", "n_clicks"),
        Input("previous_button", "n_clicks"),
        Input('display_level_selection', 'value'),
    ],
    [
        State("data_table", "selected_rows"),
        State('data_table', 'derived_virtual_indices'),
    ]
)
def select_next(next_button_n_clicks, prev_button_n_clicks, display_level_selection, selected_rows, derived_virtual_indices):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'previous_button' in changed_id:
        print('previous_button was clicked')
        advance_index = -1
    elif 'next_button' in changed_id:
        print('next_button was clicked')
        advance_index = 1
    elif 'display_level_selection' in changed_id:
        advance_index = 0
    else:
        advance_index = 0

    if advance_index == 0:
        return [0]
    elif derived_virtual_indices is not None:
        current_selection = selected_rows[0]
        current_index = derived_virtual_indices.index(current_selection)
        next_index = current_index + advance_index
        if next_index >= 0:
            next_selection = derived_virtual_indices[next_index]
        else:
            next_selection = derived_virtual_indices[current_index]
        return [int(next_selection)]
    else:
        return [0]


# feedback qc data log
@app.callback(
    Output("stored_feedback", "children"),
    [
        Input("feedback_popup_ok", "n_clicks"),
    ],
    [
        State("feedback_popup_datetime", "value"),
        State("feedback_popup_username", "value"),
        State("feedback_popup_id", "value"),
        State("feedback_popup_experiments", "value"),
        State("feedback_popup_qc_dropdown", "value"),
        State("feedback_popup_motion_present", "value"),
        State("feedback_popup_qc_labels", "value"),
        State("feedback_popup_text", "value"),
        State('display_level_selection', 'value'),
    ]
)
def log_feedback(n1, timestamp, username, _id, experiment_ids, qc_attribute, motion_present, qc_labels, input_text, display_level):
    print('LOGGING FEEDBACK')
    feedback = {
        'timestamp': timestamp,
        'username': username,
        '{}_id'.format(display_level): _id,
        'experiment_id': experiment_ids,
        'qc_attribute': qc_attribute,
        'motion_present': motion_present,
        'qc_labels': qc_labels,
        'input_text': input_text,
    }
    print(feedback)
    functions.log_feedback(feedback, display_level)
    print('logging feedback at {}'.format(time.time()))
    return 'TEMP'

# toggle popup open/close state


@app.callback(
    Output("plot_qc_popup", "is_open"),
    [
        Input("open_feedback_popup", "n_clicks"),
        Input("feedback_popup_cancel", "n_clicks"),
        Input("feedback_popup_ok", "n_clicks"),
    ],
    [State("plot_qc_popup", "is_open")],
)
def toggle_modal(n1, n2, n3, is_open):
    print('modal is open? {}'.format(is_open))
    if n1 or n2:
        return not is_open
    return is_open

# fill popup with currently selected container ID


@app.callback(
    Output("feedback_popup_id", "value"),
    [
        Input('data_table', 'selected_rows'),
        Input('display_level_selection', 'value')
    ],
)
def fill_container_id(selected_rows, display_level):
    idx = selected_rows[0]
    if display_level == 'container':
        return container_table.iloc[idx]['ophys_container_id']
    elif display_level == 'session':
        return session_table.iloc[idx]['ophys_session_id']

# populate qc attributes in popup when selecting new display level


@app.callback(
    Output('feedback_popup_qc_dropdown', 'options'),
    [
        Input('display_level_selection', 'value'),
    ],
)
def update_qc_attributes(display_level):
    if display_level == 'container':
        qc_attributes = functions.load_container_qc_definitions()
    elif display_level == 'session':
        qc_attributes = functions.load_session_qc_definitions()
    qc_options = [{'label': key, 'value': key} for key in list(qc_attributes.keys())]
    return qc_options

# label radio buttons in popup with currently selected experiment_ids


@app.callback(
    Output('feedback_popup_experiments', 'options'),
    [
        Input('data_table', 'selected_rows'),
        Input('display_level_selection', 'value')
    ],
)
def experiment_id_checklist(row_index, display_level):
    if display_level == 'container':
        ophys_container_id = container_table.iloc[row_index[0]]['ophys_container_id']  # noqa: F841 - Flake8 doesn't recognize the variable being used below
        subset = experiment_table.query('ophys_container_id == @ophys_container_id').sort_values(by='ophys_experiment_id')[['session_type', 'ophys_experiment_id']].reset_index(drop=True)
        options = [{'label': '{} {}'.format(subset.loc[i]['session_type'], subset.loc[i]['ophys_experiment_id']), 'value': subset.loc[i]['ophys_experiment_id']} for i in range(len(subset))]
    elif display_level == 'session':
        experiments = list(np.array(session_table.iloc[row_index[0]]['ophys_experiment_ids, paired']).flatten())
        options = [{'label': oeid, 'value': oeid} for oeid in experiments]
    else:
        options = [{'label': 'NONE', 'value': None}]
    return options

# select all experiments


@app.callback(
    Output('feedback_popup_experiments', 'value'),
    [
        Input('feedback_popup_select_all_experiments', 'n_clicks_timestamp'),
        Input('feedback_popup_unselect_all_experiments', 'n_clicks_timestamp'),
        Input("open_feedback_popup", "n_clicks_timestamp")
    ],
    [State('feedback_popup_experiments', 'options')]
)
def select_all_experiments(select_all_timestamp, unselect_all_timestamp, open_feedback_timestamp, options):
    # value = [v for k,v in options.items()]
    if select_all_timestamp is None:
        select_all_timestamp = 0
    if unselect_all_timestamp is None:
        unselect_all_timestamp = 0
    if open_feedback_timestamp is None:
        open_feedback_timestamp = 0
    if select_all_timestamp > unselect_all_timestamp and select_all_timestamp > open_feedback_timestamp:
        return [d['value'] for d in options]
    else:
        return []

# populate feedback popup qc options

# only enable OK button when all fields are populated


@app.callback(
    Output('feedback_popup_ok', 'disabled'),
    [
        Input('feedback_popup_username', 'value'),
        Input('feedback_popup_experiments', 'value'),
        Input('feedback_popup_qc_dropdown', 'value'),
        Input('feedback_popup_qc_labels', 'value'),
        Input('feedback_popup_motion_present', 'value')
    ],
)
def enable_popup_ok(username, selected_experiments, qc_attribute, qc_label, motion_present):
    # we need a special case for the motion_present flag in "Motion Correction"
    if qc_attribute == 'Motion Correction':
        binary_choice_selected = motion_present is not None
    else:
        binary_choice_selected = True

    anything_missing = (
        username == ''
        or username is None
        or selected_experiments is None
        or qc_attribute == ''
        or qc_attribute is None
        or qc_label is None
        or binary_choice_selected == False
    )
    if anything_missing:
        # return disabled = True if anything is missing
        return True
    else:
        # else return disabled = False
        return False


@app.callback(
    Output('feedback_popup_qc_labels', 'options'),
    [
        Input('feedback_popup_qc_dropdown', 'value'),
        Input('display_level_selection', 'value'),
    ],
)
def populate_qc_options(attribute_to_qc, display_level):
    if display_level == 'container':
        qc_attributes = functions.load_container_qc_definitions()
    elif display_level == 'session':
        qc_attributes = functions.load_session_qc_definitions()
    try:
        return [{'label': v, 'value': v} for v in qc_attributes[attribute_to_qc]['qc_attributes']]
    except KeyError:
        return []

# clear popup text


@app.callback(
    Output("feedback_popup_text", "value"),
    [
        Input("open_feedback_popup", "n_clicks"),
    ],
    [State("plot_qc_popup", "is_open")],
)
def clear_popup_text(n1, is_open):
    return ''


# # clear experiment selections
# @app.callback(
#     Output('feedback_popup_experiments', 'value'),
#     [
#         Input("open_feedback_popup", "n_clicks"),
#     ],
#     [State("plot_qc_popup", "is_open")],
# )
# def clear_experiment_labels(n1, is_open):
#     return None


# clear motion selection
@app.callback(
    Output("feedback_popup_motion_present", "value"),
    [
        Input("open_feedback_popup", "n_clicks"),
    ],
    [State("plot_qc_popup", "is_open")],
)
def clear_motion_label(n1, is_open):
    return None

# clear qc label selections


@app.callback(
    Output("feedback_popup_qc_labels", "value"),
    [
        Input("open_feedback_popup", "n_clicks"),
    ],
    [State("plot_qc_popup", "is_open")],
)
def clear_qc_labels(n1, is_open):
    return None


# populate datetime in feedback popup
@app.callback(
    Output("feedback_popup_datetime", "value"),
    [
        Input("open_feedback_popup", "n_clicks"),
    ],
)
def populate_popup_datetime(n_clicks):
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@app.callback(Output('data_table', 'page_size'), [Input('entries_per_page_input', 'value')])
def change_entries_per_page(entries_per_page):
    return entries_per_page


@app.callback(Output('container_overview_iframe', 'src'), [Input('container_overview_dropdown', 'value')])
def embed_iframe(value):
    print('getting a new iframe')
    print('value: {}'.format(value))
    if value == 'motion_corrected_movies':
        print("going to show URLs!!!!")
        return None
    else:
        return app.get_asset_url('qc_plots/overview_plots/{}'.format(value))


# update container overview options when container checklist state is changed
@app.callback(Output('container_overview_dropdown', 'options'), [Input('container_checklist', 'value')])
def update_container_overview_options(checkbox_values):
    global container_overview_plot_options
    container_overview_plot_options = functions.load_container_overview_plot_options()
    return container_overview_plot_options


# update container plot options when container checklist state is changed
@app.callback(
    Output('plot_selection_dropdown', 'options'),
    [
        Input('container_checklist', 'value'),
        Input('display_level_selection', 'value')
    ]
)
def update_container_plot_options(checkbox_values, display_level_selection):
    if display_level_selection == 'container':
        plot_options = functions.load_container_plot_options()
    elif display_level_selection == 'session':
        plot_options = functions.load_session_plot_options()
    return plot_options


# show/hide container view frame based on 'container_checklist'
@app.callback(Output('container_overview_iframe', 'hidden'), [Input('container_checklist', 'value')])
def show_container_view(checkbox_values):
    if 'show_container_plots' in checkbox_values:
        # retun hidden = False
        return False
    else:
        # return hidden = True
        return True


# repopulate plot inventory frame based on 'container_checklist'
@app.callback(Output('plot_inventory_graph', 'figure'), [Input('container_checklist', 'value')])
def regenerate_plot_inventory(checkbox_values):
    if 'show_plot_inventory' in checkbox_values:

        plot_inventory = functions.generate_plot_inventory()
        plot_inventory_fig = functions.make_plot_inventory_heatmap(plot_inventory)
        temp_fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[1, 8 * np.random.rand(), 2])])
        return plot_inventory_fig
    else:
        # return hidden = True
        temp_fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[0, 0, 0])])
        return temp_fig


# show/hide plot inventory frame based on 'container_checklist'
@app.callback(Output('plot_inventory_container', 'style'), [Input('container_checklist', 'value')])
def show_plot_inventory_checklist(checkbox_values):
    if 'show_plot_inventory' in checkbox_values:
        # retun hidden = False
        print('making plot visible!!')
        return {'display': 'block'}
    else:
        # return hidden = True
        return {'display': 'none'}


# show/hide motion label based on attribute being qc'd
@app.callback(Output('feedback_popup_motion_present', 'style'), [Input('feedback_popup_qc_dropdown', 'value')])
def show_plot_inventory_attribute(attribute_to_qc):
    if attribute_to_qc == "Motion Correction":
        return {'display': True}
    else:
        # return hidden = True
        return {'display': 'none'}

# show/hide motion label based on attribute being qc'd


@app.callback(Output('feedback_popup_motion_present_label', 'style'), [Input('feedback_popup_qc_dropdown', 'value')])
def show_plot_inventory(attribute_to_qc):
    if attribute_to_qc == "Motion Correction":
        return {'display': True}
    else:
        # return hidden = True
        return {'display': 'none'}


# show/hide container dropdown based on 'container_checklist'
@app.callback(Output('container_overview_dropdown', 'style'), [Input('container_checklist', 'value')])
def show_container_dropdown(checkbox_values):
    if 'show_container_plots' in checkbox_values:
        # return hidden = False
        return {'display': 'block'}
    else:
        # return hidden = True
        return {'display': 'none'}


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

    style_data_conditional = [{
        "if": {"row_index": index_to_highlight},
        "backgroundColor": "#3D9970",
        'color': 'white'
    }]
    return style_data_conditional

# set plot titles
# this is just text above the actual plot frame
# Use this loop to determine the correct title to update


def update_plot_title(plot_types, display_level, input_id):
    '''a function to update plot titles'''
    idx = int(input_id.split('plot_title_')[1])
    try:
        return plot_types[idx]
    except IndexError:
        return ''


for i in range(10):
    app.callback(
        Output(f"plot_title_{i}", "children"),
        [
            Input("plot_selection_dropdown", "value"),
            Input('display_level_selection', 'value'),
            Input(f"plot_title_{i}", "id")
        ]  # noqa: F541
    )(update_plot_title)

# image frames callbacks
# generated in a loop


def update_frame_N(row_index, plot_types, display_level, input_id):
    '''
    a function to fill the image frames
    '''
    idx = int(input_id.split('image_frame_')[1])
    try:
        plot_type = plot_types[idx]
        if display_level == 'container':
            _id = container_table.iloc[row_index[0]]['ophys_container_id']
        elif display_level == 'session':
            _id = session_table.iloc[row_index[0]]['ophys_session_id']

        encoded_image = functions.get_plot(_id, plot_type=plot_type, display_level=display_level)
        return 'data:image/png;base64,{}'.format(encoded_image.decode())
    except IndexError:
        return None


for i in range(10):
    app.callback(
        Output(f"image_frame_{i}", "src"),
        [
            Input('data_table', 'selected_rows'),
            Input('plot_selection_dropdown', 'value'),
            Input('display_level_selection', 'value'),
            Input(f"image_frame_{i}", "id")
        ]
    )(update_frame_N)

# update_links


def update_link_text_N(row_index, path_style, input_id):
    '''a function to update plot titles'''
    idx = int(input_id.split('link_')[1])
    ophys_container_id = container_table.iloc[row_index[0]]['ophys_container_id']
    link_list = functions.get_motion_corrected_movie_paths(ophys_container_id)
    if path_style == 'windows':
        link_list = [v.replace('/', '\\') for v in link_list]
    try:
        return link_list[idx]
    except IndexError:
        return 'INVALID LINK'


for i in range(10):
    app.callback(
        Output(f"link_{i}", "children"),
        [
            Input('data_table', 'selected_rows'),
            Input('path_style', 'value'),
            Input(f"link_{i}", "id")
        ]
    )(update_link_text_N)


def update_link_destination_N(row_index, input_id):
    '''a function to update plot titles'''
    idx = int(input_id.split('link_')[1])
    ophys_container_id = container_table.iloc[row_index[0]]['ophys_container_id']
    link_list = functions.get_motion_corrected_movie_paths(ophys_container_id)
    try:
        return 'file:{}'.format(link_list[idx])
    except IndexError:
        return 'https://www.google.com/'


for i in range(10):
    app.callback(
        Output(f"link_{i}", "href"),
        [Input('data_table', 'selected_rows'), Input(f"link_{i}", "id")]
    )(update_link_destination_N)


def update_link_visibility_N(row_index, input_id):
    '''a function to update plot titles'''
    idx = int(input_id.split('link_')[1])
    try:
        print("Returning True, idx = {}".format(idx))
        return {'display': True}
    except IndexError:
        print("Returning None, idx = {}".format(idx))
        return {'display': 'none'}


for i in range(10):
    app.callback(
        Output(f"link_{i}", "style"),
        [Input('data_table', 'selected_rows'), Input(f"link_{i}", "id")]
    )(update_link_visibility_N)

# update session header


@app.callback(
    Output('session_header', 'children'),
    [
        Input('data_table', 'selected_rows'),
        Input('display_level_selection', 'value'),
    ]
)
def update_session_id_header(row_index, display_level_selection):
    if display_level_selection == 'session':
        return 'Experiments associated with Session ID {}'.format(session_table.iloc[row_index[0]]['ophys_session_id'])


def update_exp_link_text_N(row_index, display_level_selection, path_style, input_id):
    if display_level_selection == 'session':
        print('IN update_exp_link_text_N')
        print('\t row_index = {}'.format(row_index))
        print('\t input_id = {}'.format(input_id))
        idx = int(input_id.split('exp_link_')[1])
        print('\t idx = {}'.format(idx))
        session_id = session_table.iloc[row_index[0]]['ophys_session_id']
        exp_list = functions.get_experiment_ids_for_session_id(int(session_id))
        try:
            return exp_list[idx]
        except IndexError:
            return 'INVALID LINK'
    else:
        return "NOT APPLICABLE"


for i in range(8):
    app.callback(
        Output(f"exp_link_{i}", "children"),
        [
            Input('data_table', 'selected_rows'),
            Input('display_level_selection', 'value'),
            Input('path_style', 'value'),
            Input(f"exp_link_{i}", "id")
        ]
    )(update_exp_link_text_N)


@app.callback(Output('link_0', 'children'),
              [Input('data_table', 'selected_rows'),
               Input('plot_selection_dropdown', 'value'),
               ])
def print_movie_paths(row_index, plot_types):
    ophys_container_id = container_table.iloc[row_index[0]]['ophys_container_id']
    output_text = functions.print_motion_corrected_movie_paths(ophys_container_id)
    print(output_text)
    return output_text


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run dash visualization app for VB production data')
    parser.add_argument(
        '--port',
        type=int,
        default='3389',
        metavar='port on which to host. 3389 (Remote desktop port) by default, since it is open over VPN)'
    )
    parser.add_argument(
        '--debug',
        help='boolean, not followed by an argument. Enables debug mode. False by default.',
        action='store_true'
    )
    args = parser.parse_args()
    print("PORT = {}".format(args.port))
    print("DEBUG MODE = {}".format(args.debug))
    app.run_server(debug=args.debug, port=args.port, host='0.0.0.0')
