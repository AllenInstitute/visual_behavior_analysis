import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import argparse
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
                dbc.Input(id="input", placeholder="Type something...", type="text"),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close", className="ml-auto")
                ),
            ],
            id="modal",
        ),
    ]
)


class FeedbackButton(object):
    def __init__(self, id_number=0, plot_name='test', body_text=None):
        self.id_number = id_number
        self.plot_name = plot_name
        if body_text is not None:
            self.body_text = body_text
        else:
            self.body_text = 'this is the popup body'

        self.popup = self.make_popup()

    def make_popup(self):
        feedback_button = html.Div(
            [
                dbc.Button(
                    "{} - Provide Feedback".format(self.plot_name),
                    id="open_feedback_{}".format(self.id_number)
                ),
                dbc.Modal(
                    [
                        dbc.ModalHeader("Provide feedback for {}".format(self.plot_name)),
                        dbc.ModalBody(self.body_text),
                        dbc.ModalFooter(
                            dbc.Button("Close", id="close_{}".format(self.id_number), className="ml-auto")
                        ),
                    ],
                    id="plot_qc_popup_{}".format(self.id_number),
                ),
            ]
        )
        return feedback_button

    def update_button_text(self, new_text):
        self.popup.children[0].children = new_text


fb = [FeedbackButton(id_number=i) for i in range(3)]
fb[0].update_button_text('some new text')
fb[1].update_button_text('some more new text')

app.layout = html.Div([
    modal,
    html.H4(),
    fb[0].popup,
    fb[1].popup,
], className='container')


@app.callback(
    Output("modal", "is_open"),
    [Input("open", "n_clicks"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    global fb1
    if n1 or n2:
        print('current button text = {}'.format(fb1.popup.children[0].children))
        fb1.update_button_text('updated text')
        return not is_open
    return is_open


@app.callback(
    Output("open_feedback_0", "children"),
    [Input("close", "n_clicks")],
)
def toggle_modal_2(n1):
    global fb1
    return fb1.popup.children[0].children


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
