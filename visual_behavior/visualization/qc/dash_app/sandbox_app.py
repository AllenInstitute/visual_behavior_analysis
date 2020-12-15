import dash
from dash.dependencies import Input, Output
import dash_html_components as html
import argparse
import dash_bootstrap_components as dbc
import dash_core_components as dcc

# APP SETUP
# app = dash.Dash(__name__,)
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'sandbox'
# app.config['suppress_callback_exceptions'] = True

link_list = [
    '//allen/programs/braintv/production/visualbehavior/prod0/specimen_722884882/ophys_session_787661032/ophys_experiment_788490510/processed/motion_preview.10x.mp4',
    '//allen/programs/braintv/production/neuralcoding/prod0/specimen_807248992/ophys_session_882060185/ophys_experiment_882551935/processed/motion_preview.10x.mp4',
]


app.layout = html.Div([
    html.H4(),
    dcc.Input(
        id="input_0",
        type="text",
        placeholder=""
    ),
    html.H4(),
    dcc.Input(
        id="input_1",
        type="text",
        placeholder=""
    ),
    html.Button('Next', id='next_button'),
    html.Div(id='text_output', children='SOME TEST TEXT'),
    dcc.Link(id='link_0', children='', href='', style={'display': True}),
    html.H4(),
    dcc.Link(id='link_1', children='', href='', style={'display': True}),
], className='container')


def pass_text(text, input_id):
    i = input_id.split('_')[1]
    return '{} - {}'.format(text, i)


for i in range(2):
    app.callback(
        Output(f"link_{i}", "children"), [Input(f"input_{i}", "value"), Input(f"link_{i}", "id")]
    )(pass_text)


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
