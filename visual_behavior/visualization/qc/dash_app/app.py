#!/usr/bin/env python

import dash
from dash.dependencies import Input, Output
import dash_html_components as html
import argparse

from functions import *
from components import *

# APP SETUP
app = dash.Dash(__name__,)
app.title = 'Visual Behavior Data QC'
app.config['suppress_callback_exceptions'] = True

# FUNCTION CALLS
container_table = load_data().sort_values('first_acquistion_date')
container_plot_options = load_container_plot_options()
container_overview_plot_options = load_container_overview_plot_options()

# COMPONENT SETUP
plot_selection_dropdown.options = container_plot_options
container_overview_dropdown.options = container_overview_plot_options
container_overview_iframe.src = app.get_asset_url('qc_plots/overview_plots/d_prime_container_overview.html')
container_data_table.columns = [{"name": i.replace('_', ' '), "id": i} for i in container_table.columns]
container_data_table.data = container_table.to_dict('records')

# APP LAYOUT
app.layout = html.Div([
    html.H3('Visual Behavior Data'),
    # checklist for components to show
    show_overview_checklist,
    # container level dropdown
    container_overview_dropdown,
    # frame with container level plots
    container_overview_iframe,
    html.H4('Container Summary Data Table:'),
    html.I('Adjust number of rows to display in the data table:'),
    table_row_selection,
    # data table
    container_data_table,
    # dropdown for plot selection
    html.H4('Select plots to generate from the dropdown (max 10)'),
    plot_selection_dropdown,
    plot_titles[0],
    plot_frames[0],
    plot_titles[1],
    plot_frames[1],
    plot_titles[2],
    plot_frames[2],
    plot_titles[3],
    plot_frames[3],
    plot_titles[4],
    plot_frames[4],
    plot_titles[5],
    plot_frames[5],
    plot_titles[6],
    plot_frames[6],
    plot_titles[7],
    plot_frames[7],
    plot_titles[8],
    plot_frames[8],
    plot_titles[9],
    plot_frames[9],
    plot_titles[10],
    plot_frames[10],
    plot_titles[11],
    plot_frames[11],
    plot_titles[12],
    plot_frames[12],
    plot_titles[13],
    plot_frames[13],
    plot_titles[14],
    plot_frames[14],
    plot_titles[15],
    plot_frames[15],
    plot_titles[16],
    plot_frames[16],
    plot_titles[17],
    plot_frames[17],
    plot_titles[18],
    plot_frames[18],
    plot_titles[19],
    plot_frames[19],
], className='container')


@app.callback(Output('data_table', 'page_size'), [Input('entries_per_page_input', 'value')])
def change_entries_per_page(entries_per_page):
    return entries_per_page


@app.callback(Output('container_overview_iframe', 'src'), [Input('container_overview_dropdown', 'value')])
def embed_iframe(value):
    return app.get_asset_url('qc_plots/overview_plots/{}'.format(value))

# update container overview options when container checklist state is changed
@app.callback(Output('container_overview_dropdown', 'options'), [Input('container_checklist', 'value')])
def update_container_overview_options(checkbox_values):
    global container_overview_plot_options
    container_overview_plot_options = load_container_overview_plot_options()
    return container_overview_plot_options

# update container plot options when container checklist state is changed
@app.callback(Output('container_plot_dropdown', 'options'), [Input('container_checklist', 'value')])
def update_container_plot_options(checkbox_values):
    global container_plot_options
    container_plot_options = load_container_plot_options()
    return container_plot_options

# show/hide container view frame based on 'container_checklist'
@app.callback(Output('container_overview_iframe', 'hidden'), [Input('container_checklist', 'value')])
def show_container_view(checkbox_values):
    if 'show_container_plots' in checkbox_values:
        # retun hidden = False
        return False
    else:
        # return hidden = True
        return True

# show/hide container dropdown based on 'container_checklist'
@app.callback(Output('container_overview_dropdown', 'style'), [Input('container_checklist', 'value')])
def show_container_view(checkbox_values):
    if 'show_container_plots' in checkbox_values:
        # return hidden = False
        return {}
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
    print('clicked')
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


@app.callback(Output('plot_title_0', 'children'),
              [Input('container_plot_dropdown', 'value'),
               ])
def update_frame_0(plot_types):
    if len(plot_types) >= 1:
        return plot_types[0]


@app.callback(Output('plot_title_1', 'children'),
              [Input('container_plot_dropdown', 'value'),
               ])
def update_frame_1(plot_types):
    if len(plot_types) >= 2:
        return plot_types[1]


@app.callback(Output('plot_title_2', 'children'),
              [Input('container_plot_dropdown', 'value'),
               ])
def update_frame_2(plot_types):
    if len(plot_types) >= 3:
        return plot_types[2]


@app.callback(Output('plot_title_3', 'children'),
              [Input('container_plot_dropdown', 'value'),
               ])
def update_frame_3(plot_types):
    if len(plot_types) >= 4:
        return plot_types[3]


@app.callback(Output('plot_title_4', 'children'),
              [Input('container_plot_dropdown', 'value'),
               ])
def update_frame_4(plot_types):
    if len(plot_types) >= 5:
        return plot_types[4]


@app.callback(Output('plot_title_5', 'children'),
              [Input('container_plot_dropdown', 'value'),
               ])
def update_frame_5(plot_types):
    if len(plot_types) >= 6:
        return plot_types[5]


@app.callback(Output('plot_title_6', 'children'),
              [Input('container_plot_dropdown', 'value'),
               ])
def update_frame_6(plot_types):
    if len(plot_types) >= 7:
        return plot_types[6]


@app.callback(Output('plot_title_7', 'children'),
              [Input('container_plot_dropdown', 'value'),
               ])
def update_frame_7(plot_types):
    if len(plot_types) >= 8:
        return plot_types[7]


@app.callback(Output('plot_title_8', 'children'),
              [Input('container_plot_dropdown', 'value'),
               ])
def update_frame_8(plot_types):
    if len(plot_types) >= 9:
        return plot_types[8]


@app.callback(Output('plot_title_9', 'children'),
              [Input('container_plot_dropdown', 'value'),
               ])
def update_frame_9(plot_types):
    if len(plot_types) >= 10:
        return plot_types[9]

# image frames callbacks
# (I can't figure out how to make these in a loop!)


@app.callback(Output('image_frame_0', 'src'),
              [Input('data_table', 'selected_rows'),
               Input('container_plot_dropdown', 'value'),
               ])
def update_frame_10(row_index, plot_types):
    print('new row selected')
    if len(plot_types) >= 1:
        plot_type = plot_types[0]
        container_id = container_table.iloc[row_index[0]]['container_id']
        encoded_image = get_container_plot(container_id, plot_type=plot_type)
        return 'data:image/png;base64,{}'.format(encoded_image.decode())


@app.callback(Output('image_frame_1', 'src'),
              [Input('data_table', 'selected_rows'),
               Input('container_plot_dropdown', 'value'),
               ])
def update_frame_11(row_index, plot_types):
    if len(plot_types) >= 2:
        plot_type = plot_types[1]
        container_id = container_table.iloc[row_index[0]]['container_id']
        encoded_image = get_container_plot(container_id, plot_type=plot_type)
        return 'data:image/png;base64,{}'.format(encoded_image.decode())


@app.callback(Output('image_frame_2', 'src'),
              [Input('data_table', 'selected_rows'),
               Input('container_plot_dropdown', 'value'),
               ])
def update_frame_12(row_index, plot_types):
    if len(plot_types) >= 3:
        plot_type = plot_types[2]
        container_id = container_table.iloc[row_index[0]]['container_id']
        encoded_image = get_container_plot(container_id, plot_type=plot_type)
        return 'data:image/png;base64,{}'.format(encoded_image.decode())


@app.callback(Output('image_frame_3', 'src'),
              [Input('data_table', 'selected_rows'),
               Input('container_plot_dropdown', 'value'),
               ])
def update_frame_13(row_index, plot_types):
    if len(plot_types) >= 4:
        plot_type = plot_types[3]
        container_id = container_table.iloc[row_index[0]]['container_id']
        encoded_image = get_container_plot(container_id, plot_type=plot_type)
        return 'data:image/png;base64,{}'.format(encoded_image.decode())


@app.callback(Output('image_frame_4', 'src'),
              [Input('data_table', 'selected_rows'),
               Input('container_plot_dropdown', 'value'),
               ])
def update_frame_14(row_index, plot_types):
    if len(plot_types) >= 5:
        plot_type = plot_types[4]
        container_id = container_table.iloc[row_index[0]]['container_id']
        encoded_image = get_container_plot(container_id, plot_type=plot_type)
        return 'data:image/png;base64,{}'.format(encoded_image.decode())


@app.callback(Output('image_frame_5', 'src'),
              [Input('data_table', 'selected_rows'),
               Input('container_plot_dropdown', 'value'),
               ])
def update_frame_15(row_index, plot_types):
    if len(plot_types) >= 6:
        plot_type = plot_types[5]
        container_id = container_table.iloc[row_index[0]]['container_id']
        encoded_image = get_container_plot(container_id, plot_type=plot_type)
        return 'data:image/png;base64,{}'.format(encoded_image.decode())


@app.callback(Output('image_frame_6', 'src'),
              [Input('data_table', 'selected_rows'),
               Input('container_plot_dropdown', 'value'),
               ])
def update_frame_16(row_index, plot_types):
    if len(plot_types) >= 7:
        plot_type = plot_types[6]
        container_id = container_table.iloc[row_index[0]]['container_id']
        encoded_image = get_container_plot(container_id, plot_type=plot_type)
        return 'data:image/png;base64,{}'.format(encoded_image.decode())


@app.callback(Output('image_frame_7', 'src'),
              [Input('data_table', 'selected_rows'),
               Input('container_plot_dropdown', 'value'),
               ])
def update_frame_17(row_index, plot_types):
    if len(plot_types) >= 8:
        plot_type = plot_types[7]
        container_id = container_table.iloc[row_index[0]]['container_id']
        encoded_image = get_container_plot(container_id, plot_type=plot_type)
        return 'data:image/png;base64,{}'.format(encoded_image.decode())


@app.callback(Output('image_frame_8', 'src'),
              [Input('data_table', 'selected_rows'),
               Input('container_plot_dropdown', 'value'),
               ])
def update_frame_18(row_index, plot_types):
    if len(plot_types) >= 9:
        plot_type = plot_types[8]
        container_id = container_table.iloc[row_index[0]]['container_id']
        encoded_image = get_container_plot(container_id, plot_type=plot_type)
        return 'data:image/png;base64,{}'.format(encoded_image.decode())


@app.callback(Output('image_frame_9', 'src'),
              [Input('data_table', 'selected_rows'),
               Input('container_plot_dropdown', 'value'),
               ])
def update_frame_19(row_index, plot_types):
    if len(plot_types) >= 10:
        plot_type = plot_types[9]
        container_id = container_table.iloc[row_index[0]]['container_id']
        encoded_image = get_container_plot(container_id, plot_type=plot_type)
        return 'data:image/png;base64,{}'.format(encoded_image.decode())


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
