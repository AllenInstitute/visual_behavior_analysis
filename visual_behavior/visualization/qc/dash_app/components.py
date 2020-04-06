#!/usr/bin/env python

import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.graph_objs as go
import dash_bootstrap_components as dbc

from functions import generate_plot_inventory, make_plot_inventory_heatmap


# dropdown for selecting plots to display
plot_selection_dropdown = dcc.Dropdown(
    id='container_plot_dropdown',
    options=None,
    value=['ophys_session_sequence'],
    multi=True
)

# checklist for components to show
show_overview_checklist = dcc.Checklist(
    id='container_checklist',
    options=[
        {'label': 'Show Container Level Summary Plots   |', 'value': 'show_container_plots'},
        {'label': 'Show Plot Inventory   ', 'value': 'show_plot_inventory'},
    ],
    value=[]
)

# dropdown to select which overview plot to show in iframe
container_overview_dropdown = dcc.Dropdown(
    id='container_overview_dropdown',
    style={'display': 'none'},
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

feeback_button = html.Div(
    [
        dbc.Button("Provide Feedback", id="open"),
        dbc.Modal(
            [
                dbc.ModalHeader("Plot Feedback"),
                dbc.ModalBody("This is a prototype popup for plot QC"),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close", className="ml-auto")
                ),
            ],
            id="plot_qc_popup",
        ),
    ]
)


class FeebackButton(object):
    def __init__(plot_number, plot_name, body_text=None):
        self.plot_number = plot_number
        self.plot_name = plot_name
        if body_text is not None:
            self.body_text = body_text
        else:
            self.body_text = 'this is the popup body'

        self.popup = self.make_popup()

    def make_popup():
        feeback_button = html.Div(
            [
                dbc.Button(
                    "{} - Provide Feedback".format(plot_title),
                    id="open_feedback_{}".format(self.plot_number)
                ),
                dbc.Modal(
                    [
                        dbc.ModalHeader("Provide feedback for {}".form(plot_title)),
                        dbc.ModalBody(self.body_text),
                        dbc.ModalFooter(
                            dbc.Button("Close", id="close", className="ml-auto")
                        ),
                    ],
                    id="plot_qc_popup_{}".format(self.plot_number),
                ),
            ]
        )
        return feedback_button

    def update_info(plot_number, plot_name):
        self.popup
