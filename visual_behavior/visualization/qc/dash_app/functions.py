#!/usr/bin/env python

import base64
import os
import yaml
import json
import pandas as pd
import plotly.graph_objs as go
import datetime
import uuid
from visual_behavior import database as db

import visual_behavior.visualization.qc.data_loading as dl
from visual_behavior.data_access import loading


def load_container_data():
    container_df = dl.build_container_df()
    return container_df


def load_session_data():
    session_table = refactor_sessions_table_mesoscope_for_qc()
    session_table['container_id'] = session_table['container_id'].astype(int)
    session_table = session_table.reset_index().rename(columns={'index':'ophys_session_id'})

    session_table['has_decrosstalk_qc'] = False

    columms_to_show = [
        'ophys_session_id',
        'ophys_experiment_ids, paired',
        'date_of_acquisition', 
        'driver_line',
        'equipment_name', 
        'mouse_id',
        'project_code', 
        'session_type',
        'has_decrosstalk_qc',
    ]

    try:
        included_session_list = pd.read_csv('/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/session_plots/session_list.csv')
    except FileNotFoundError:
        pass

    session_table_to_return = session_table[
        session_table['ophys_session_id'].isin(included_session_list['ophys_session_id'])
    ][columms_to_show]

    return session_table_to_return


def load_yaml(yaml_path):
    with open(yaml_path, 'r') as stream:
        yaml_contents = yaml.safe_load(stream)

    options = []
    for k, v in yaml_contents.items():
        options.append({'label': k, 'value': v})
    return options


def get_plot_list(qc_definitions):
    plot_list = []
    for plot_title, attributes in qc_definitions.items():
        if attributes['show_plots']:
            plot_list.append({'label': plot_title, 'value': attributes['plot_folder_name']})
    return plot_list


def load_container_qc_definitions():
    container_qc_definition_path = "/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/container_plots/qc_definitions.json"
    return json.load(open(container_qc_definition_path))


def load_session_qc_definitions():
    container_qc_definition_path = "/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/session_plots/qc_definitions.json"
    return json.load(open(container_qc_definition_path))


def load_container_plot_options():
    container_options = get_plot_list(load_container_qc_definitions())
    print('container_options:')
    print(container_options)
    return container_options


def load_session_plot_options():
    session_options = get_plot_list(load_session_qc_definitions())
    return session_options


def load_container_overview_plot_options():
    plot_definition_path = "/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/overview_plots/plot_definitions.yml"
    container_overview_options = load_yaml(plot_definition_path)
    return container_overview_options


def get_plot_path(_id, plot_type, display_level):
    qc_plot_folder = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots'
    plot_folder = os.path.join(qc_plot_folder, '{}_plots'.format(display_level))

    plot_image_path = os.path.join(
        plot_folder,
        plot_type, '{}_{}.png'.format(display_level, _id)
    )
    return plot_image_path


def get_plot(_id, plot_type, display_level):
    plot_image_path = get_plot_path(_id, plot_type, display_level)
    try:
        encoded_image = base64.b64encode(open(plot_image_path, 'rb').read())
    except FileNotFoundError:
        print('not found, container_id = {}, plot_type = {}'.format(_id, plot_type))
        qc_plot_folder = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots'
        plot_folder = os.path.join(qc_plot_folder, '{}_plots'.format(display_level))

        plot_not_found_path = os.path.join(
            plot_folder,
            'no_cached_plot_small.png'
        )
        encoded_image = base64.b64encode(
            open(plot_not_found_path, 'rb').read())

    return encoded_image


CONTAINER_TABLE = load_container_data().sort_values('first_acquistion_date')


def generate_plot_inventory():
    global CONTAINER_TABLE
    container_table = CONTAINER_TABLE
    plots = load_container_plot_options()
    list_of_dicts = []
    for container_id in container_table['container_id'].values:
        d = {'container_id': container_id}
        for entry in plots:
            plot_type = entry['value']
            d.update({plot_type: os.path.exists(get_plot_path(container_id, plot_type, 'container'))})
        list_of_dicts.append(d)
    return pd.DataFrame(list_of_dicts).set_index('container_id').sort_index()


def make_plot_inventory_heatmap(plot_inventory):
    fig = go.Figure(
        data=go.Heatmap(
            z=plot_inventory.values.astype(int),
            x=plot_inventory.columns,
            y=plot_inventory.index,
            hoverongaps=True,
            showscale=False,
            colorscale='inferno',
            xgap=3,
            ygap=3,
        )
    )

    timestamp = datetime.datetime.now()
    timestamp_string = 'last updated on {} @ {}'.format(timestamp.strftime('%D'), timestamp.strftime('%H:%M:%S'))

    fig.update_layout(
        autosize=False,
        width=1000,
        height=3000,
        margin=dict(
            l=0,  # NOQA E741
            r=0,
            b=0,
            t=50,
            pad=0
        ),
        xaxis_title='plot type',
        yaxis_title='container ID',
        title='Plot Inventory (black = missing) {}'.format(timestamp_string)
    )
    fig.update_yaxes(autorange="reversed", type='category', dtick=1)
    fig.update_xaxes(dtick=1)

    return fig


def get_motion_corrected_movie_paths(container_id):
    et = loading.get_filtered_ophys_experiment_table().reset_index()
    paths = []
    for oeid in et.query('container_id == @container_id').sort_values(by='ophys_experiment_id')['ophys_experiment_id']:
        paths.append(
            loading.get_motion_corrected_movie_h5_location(oeid).replace(
                'motion_corrected_video.h5',
                'motion_preview.10x.mp4'
            )
        )
    return paths


def print_motion_corrected_movie_paths(container_id):
    et = loading.get_filtered_ophys_experiment_table().reset_index()
    lines = []
    for oeid in et.query('container_id == @container_id')['ophys_experiment_id']:
        movie_path = loading.get_motion_corrected_movie_h5_location(oeid).replace('motion_corrected_video.h5', 'motion_preview.10x.mp4')
        lines.append('ophys experiment ID = {}\n'.format(oeid))
        lines.append("LINUX PATH:")
        lines.append('\t<a href="url">{}</a>'.format(movie_path))
        lines.append('WINDOWS PATH')
        lines.append('\t{}'.format(movie_path.replace('/', '\\')))
        lines.append('')
    return '\n'.join(lines)


def to_json(data_to_log, display_level):
    '''log data to filesystem'''
    saveloc = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_records/{}_level'.format(display_level)
    filename = os.path.join(saveloc, '{}.json'.format(data_to_log['_id']))
    json.dump(data_to_log, open(filename, 'w' ))


def to_mongo(data_to_log, display_level):
    '''log data to mongo'''
    conn = db.Database('visual_behavior_data')
    collection = conn['ophys_qc']['{}_qc_records'.format(display_level)]
    collection.insert_one(db.clean_and_timestamp(data_to_log))
    conn.close()


def log_feedback(feedback, display_level):
    '''logs feedback from app to mongo and filesystem'''
    if pd.notnull(feedback['timestamp']):
        random_id = uuid.uuid4().hex
        feedback.update({'_id': random_id})
        to_json(feedback, display_level)
        to_mongo(feedback, display_level)


def update_qc_status(feedback):
    '''
    updates:
        motion_correction_has_qc to True if feedback['qc_attribute'] == 'Motion Correction'
        cell_matching_has_qc to True if feedback['qc_attribute'] == 'Nway Production Warp Summary'
    '''
    if feedback['qc_attribute'] == 'Motion Correction':
        conn = db.Database('visual_behavior_data')
        collection = conn['ophys_qc']['container_qc_records']
        collection.update_or_create(db.clean_and_timestamp({}))
        conn.close()


def get_qcd_oeids(container_id, qc_attribute):
    '''
    get all experiment IDS that have been QC'd for a given container_id and attribute
    '''
    conn = db.Database('visual_behavior_data')
    collection = conn['ophys_qc']['container_qc_records']
    res = pd.DataFrame(list(collection.find({'container_id': container_id, 'qc_attribute': qc_attribute})))
    conn.close
    
    if len(res) > 0:
        oeids = []
        if 'experiment_id' in res.keys():
            for entry in res['experiment_id']:
                print(entry)
                if isinstance(entry, (int, float)) and pd.notnull(entry):
                    oeids.append(int(entry))
        elif 'experiment_ids' in res.keys():
            for entry in res['experiment_ids']:
                if isinstance(entry, int):
                    # if entry is int, append to list
                    oeids.append(entry)
                elif isinstance(entry, list):
                    # if entry is list, iterate over it and append each entry to list
                    for subentry in entry:
                        oeids.append(subentry)
        

        return oeids
    else:
        return []


def qc_for_all_experiments(container_id, qc_attribute):
    '''check to see that all experiments for a given container have been QCd'''
    try:
        oeids_with_qc = set(get_qcd_oeids(container_id, qc_attribute=qc_attribute))
        oeids = set(loading.get_filtered_ophys_experiment_table().query('container_id == @container_id').reset_index()['ophys_experiment_id'].values)

        # symmetric_difference is an empty set if all oeids are in oeids_with_qc
        return len(oeids.symmetric_difference(oeids_with_qc)) == 0
    except TypeError:
        pass


def set_qc_complete_flags(feedback):
    if 'container_id' in feedback.keys():
        container_id = feedback['container_id']
        session_id = None
        feedback_type = 'container'
    elif 'session_id' in feedback.keys():
        session_id = feedback['session_id']
        container_id = None
        feedback_type = 'session'

    qc_attribute = feedback['qc_attribute']

    entry = None
    if feedback_type == 'container':
        ## check to see if every experiment for this container has a qc entry. mark as done if so
        try:
            if qc_for_all_experiments(container_id, qc_attribute):
                if qc_attribute == 'Motion Correction':
                    entry = {
                        'container_id': container_id,
                        'motion_correction_has_qc': True
                    }
                elif qc_attribute == 'Nway Production Warp Summary':
                    entry = {
                        'container_id': container_id,
                        'cell_matching_has_qc': True
                    }
        except ValueError:
            pass

        if entry:
            conn = db.Database('visual_behavior_data')
            collection = conn['ophys_qc']['container_qc']
            db.update_or_create(collection, db.clean_and_timestamp(entry), keys_to_check=['container_id'])
            conn.close()

            print('Updating qc state for {}'.format(qc_attribute))

    # elif feedback_type == 'session':
    #     if qc_attribute == 'Decrosstalking - Session Level':



def get_paired_planes(session_id):
        ''' 
        Get paired experiments for given session. 
        This function will first query LIMS. (Query provided by Wayne)
        But since LIMS does not have this information for a lot of older mesoscope sesions, if the query returns nothing, 
        it will try and parce cell extraction input json to get paired planes
        :param ophys_session_id
        :return: list of two elememnt list, where each sublist is two experiment IDs of a pair of coupled planes. In order of acquisition. 
        '''
        pairs = []
        try:
            query = (f"""SELECT
            os.id as session_id,
            oe.id as exp_id,
            oe.ophys_imaging_plane_group_id as pair_id,
            oipg.group_order
            FROM ophys_sessions os
            JOIN ophys_experiments oe ON oe.ophys_session_id=os.id
            JOIN ophys_imaging_plane_groups oipg ON oipg.id=oe.ophys_imaging_plane_group_id
            WHERE os.id = {session_id}
            ORDER BY exp_id
            """)
            
            pairs_df = db.lims_query(query)
            
        except Exception as e:
            print("Unable to query LIMS database: {}".format(e))
        if len(pairs_df) > 0:
            num_groups = pairs_df['group_order'].drop_duplicates().values
            for i in num_groups:
                pair = [exp_id for exp_id in pairs_df.loc[pairs_df['group_order'] == i].exp_id]
                pairs.append(pair)
        else:
            print(f"Lims returned no group information about session {self.session_id}, using hardcoded splitting json filename")
            splitting_json = self.get_splitting_json()
            with open(splitting_json, "r") as f:
                data = json.load(f)
            for pg in data.get("plane_groups", []):
                pairs.append([p["experiment_id"] for p in pg.get("ophys_experiments", [])])
        return pairs
    
def refactor_sessions_table_mesoscope_for_qc():
    '''
    Refactor the sessions table for emoscope decrosstalking QC.
    :param 
    :return: pandas DataFrame, refactored table
    '''
    session_table = loading.get_filtered_ophys_session_table()
    meso_only_sessions = session_table.loc[session_table.equipment_name == 'MESO.1']
    meso_only_sessions_filtered = meso_only_sessions #.drop(columns=['ophys_experiment_id', 'at_least_one_experiment_passed', 'age_in_days', 'at_least_one_experiment_passed', 'behavior_session_id', 'donor_id', 'full_genotype', 'model_outputs_available', 'reporter_line', 'session_name', 'sex', 'specimen_id'])
    meso_table = pd.concat([meso_only_sessions_filtered, pd.DataFrame(columns=['ophys_experiment_ids, paired'])])
    for session in meso_table.index:
        paired_planes = get_paired_planes(session)
        meso_table.at[session,'ophys_experiment_ids, paired'] = paired_planes
    return meso_table


def get_roi_overlap_plots_links(session_id, plots_dir = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/single_cell_plots/mesoscope_decrosstalk'):
    """
    function to build links ot the roi-level plots given session_id
    session_id : int, session ID form lims
    plots_dir: str, path to outer directory 
    returns dict, where {'pair_0_overlaps' : "path_to_roi_level_dir"}
    """
    session_path = os.path.join(plots_dir, f"session_{session_id}")
    roi_links = {}
    pairs = get_paired_planes(session_id)
    for i, pair in enumerate(pairs):
        pair_dir_path = os.path.join(session_path, f"pair_{i}_overlaps")
        if os.path.isdir(pair_dir_path):
            roi_links[f'pair_{i}'] = pair_dir_path
        else:
            roi_links[f'pair_{i}']  = "roi level plots don't exist"        
    return roi_links


def does_session_have_qc(session_id, attribute):
    
    conn = db.Database('visual_behavior_data')
    collection = conn['ophys_qc']['session_qc_records']
    res = list(collection.find({'session_id':session_id, 'qc_attribute':attribute}))
    conn.close()
    
    return len(res) > 0


def mark_session_as_qcd(session_id, attribute):
    entry = {
        'session_id':session_id,
        attribute:True
    }
    
    conn = db.Database('visual_behavior_data')
    collection = conn['ophys_qc']['session_qc']
    db.update_or_create(collection, db.clean_and_timestamp(entry), keys_to_check=['session_id'])
    conn.close()


def update_session_table(session_table):
    '''
    updates session table with flag to mark 'has_decrosstalk_qc' is True
    '''
    print('UPDATING SESSION LEVEL TABLE')
    for idx,row in session_table.iterrows():
        osid = row['ophys_session_id']
        attribute = 'Decrosstalking - Session Level'
        if does_session_have_qc(osid, attribute):
            session_table.at[idx,'has_decrosstalk_qc'] = True
            print('osid {} is already done'.format(osid))
    return session_table