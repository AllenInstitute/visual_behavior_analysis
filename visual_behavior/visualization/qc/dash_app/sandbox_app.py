import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import argparse
import numpy as np
import plotly.graph_objs as go
import time
import dash_bootstrap_components as dbc

# APP SETUP
# app = dash.Dash(__name__,)
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'sandbox'
# app.config['suppress_callback_exceptions'] = True

modal = html.Div(
    [
        dbc.Button("Open modal", id="open"),
        dbc.Modal(
            [
                dbc.ModalHeader("Header"),
                dbc.ModalBody("This is the content of the modal"),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close", className="ml-auto")
                ),
            ],
            id="modal",
        ),
    ]
)

alerts = html.Div(
    [
        dbc.Alert("This is a primary alert", color="primary"),
        dbc.Alert("This is a secondary alert", color="secondary"),
        dbc.Alert("This is a success alert! Well done!", color="success"),
        dbc.Alert("This is a warning alert... be careful...", color="warning"),
        dbc.Alert("This is a danger alert. Scary!", color="danger"),
        dbc.Alert("This is an info alert. Good to know!", color="info"),
        dbc.Alert("This is a light alert", color="light"),
        dbc.Alert("This is a dark alert", color="dark"),
    ]
)

app.layout = html.Div([
    modal,
    alerts,
], className='container')


@app.callback(
    Output("modal", "is_open"),
    [Input("open", "n_clicks"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

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