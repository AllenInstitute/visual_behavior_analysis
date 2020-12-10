#!/usr/bin/env python

import dash_core_components as dcc
import dash_html_components as html
import dash_table
import dash_bootstrap_components as dbc

from functions import generate_plot_inventory, make_plot_inventory_heatmap, load_container_qc_definitions


# dropdown for selecting plots to display
plot_selection_dropdown = dcc.Dropdown(
    id='container_plot_dropdown',
    options=None,
    value=[],
    multi=True
)

# checklist for components to show
show_overview_checklist = dcc.Checklist(
    id='container_checklist',
    options=[
        {'label': 'Show Container Level Summary Plots   |', 'value': 'show_container_plots'},
        {'label': 'Show Plot Inventory   ', 'value': 'show_plot_inventory'},
    # style={'display': 'none'},
    ],
    value=[]
)

# dropdown to select which overview plot to show in iframe
container_overview_dropdown = dcc.Dropdown(
    id='container_overview_dropdown',
    # style={'display': 'none'},
    options=None,
    value='VisualBehavior_containers_chronological.png'
)

# iframe to contain overview plots
container_overview_iframe = html.Iframe(
    id='container_overview_iframe',
    style={'height': '1000px', 'width': '1500px'},
    hidden=True,
    src=None,
)

# iframe to contain overview plots
plot_inventory_iframe = html.Iframe(
    id='plot_inventory_iframe',
    style={'height': '3500px', 'width': '1500px'},
    hidden=True,
    src=None,
)

# selection for number of rows to show in table
table_row_selection = dcc.Input(
    id='entries_per_page_input',
    type='number',
    value=5,
)

# next/previous buttons
next_button = html.Button('Next', id='next_button')
previous_button = html.Button('Previous', id='previous_button')

# data table
container_data_table = dash_table.DataTable(
    id='data_table',
    columns=None,
    data=None,
    row_selectable="single",
    selected_rows=[0],
    page_size=5,
    filter_action='native',
    sort_action='native',
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
    style_cell={'padding': '2px'},
    css=[{'selector': '.row', 'rule': 'margin: 0'}]
)

plot_titles = [html.H4(id='plot_title_{}'.format(N), children='') for N in range(30)]
plot_frames = [html.Img(id='image_frame_{}'.format(N), style={'width': '1500px'}) for N in range(30)]
plot_inventory = generate_plot_inventory()
plot_inventory_fig = make_plot_inventory_heatmap(plot_inventory)

plot_inventory_graph = dcc.Graph(
    id='plot_inventory_graph',
    style={'height': 3000},
    figure=plot_inventory_fig
)
plot_inventory_graph_div = html.Div(
    id="plot_inventory_container",
    style={'display': 'block'},
    children=[
        plot_inventory_graph,
    ]
)

QC_ATTRIBUTES = load_container_qc_definitions()
QC_OPTIONS = [{'label':key, 'value':key} for key in list(QC_ATTRIBUTES.keys())]
feedback_button = html.Div(
    [
        dbc.Button("Provide Feedback", id="open_feedback_popup"),
        dbc.Modal(
            [
                dbc.ModalHeader("Plot Feedback"),
                dbc.ModalBody(
                    [
                        dbc.Label("Timestamp:"),
                        dbc.Input(id="feedback_popup_datetime", type="text", disabled=True),
                        dbc.Label("Username:"),
                        dbc.Input(id="feedback_popup_username", type="text", debounce=True),
                        dbc.Label("Container ID:"),
                        dbc.Input(id="feedback_popup_container_id", type="text", disabled=True),
                        dbc.Label("Experiment ID:"),
                        dbc.Checklist(
                            options=[
                                {"label": "exp0 ", "value": 1},
                            ],
                            value=[],
                            id="feedback_popup_experiments",
                        ),
                        dbc.Label("Attribute being QC'd:"),
                        dcc.Dropdown(
                            id='feedback_popup_qc_dropdown',
                            options=QC_OPTIONS,
                            value=''
                        ),
                        dbc.Label("QC Label:"),
                        dbc.Checklist(
                            options=[],
                            value=[],
                            id="feedback_popup_qc_labels",
                        ),
                        dbc.Label("Input text:"),
                        dbc.Input(id="feedback_popup_text", type="text", debounce=True),
                    ]
                ),
                dbc.ModalFooter(
                    [
                        dbc.Button("OK", color="primary", id="feedback_popup_ok"),
                        dbc.Button("Cancel", id="feedback_popup_cancel"),
                    ]
                ),
            ],
            id="plot_qc_popup",
        ),
    ]
)
