
import os
import numpy as np
import pandas as pd

from visual_behavior.data import loading

### FUNCTIONS TO REFORMAT DATA LOADED FROM ALLENSDK TO ADDRESS ISSUES WITH MISSING, INCORRECT OR IMPROPERLY STRUCTURED DATA ###

### THESE FUNCTIONS ARE TEMPORARY WORKAROUNDS UNTIL THE ISSUES CAN BE RESOLVED IN THE SDK ###


# REFORMATTING MANIFEST DATA #


def add_mouse_seeks_fail_tags_to_experiments_table(experiments):
    mouse_seeks_report_file_base = r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots'
    report_file = 'ophys_session_log_031820.xlsx'
    vb_report_path = os.path.join(mouse_seeks_report_file_base, report_file)
    vb_report_df = pd.read_excel(vb_report_path)

    def clean_columns(columns):
        return [c.lower().replace(' ', '_') for c in columns]

    vb_report_df.columns = clean_columns(vb_report_df.columns)
    vb_report_df = vb_report_df.rename(columns={'session_id': 'ophys_session_id'})
    # merge fail tags into all_experiments manifest
    experiments = experiments.merge(vb_report_df[['ophys_session_id', 'session_tags', 'failure_tags']],
                                    right_on='ophys_session_id', left_on='ophys_session_id')
    return experiments


def add_location_to_expts(expts):
    expts['location'] = [expts.loc[x]['cre_line'].split('-')[0] + '_' + expts.loc[x]['targeted_structure'] + '_' + str(int(expts.loc[x]['imaging_depth'])) for x in expts.index.values]
    return expts


def get_exposure_number_for_group(group):
    order = np.argsort(group['date_of_acquisition'].values)
    group['exposure_number'] = order
    return group


def add_exposure_number_to_experiments_table(experiments):
    experiments = experiments.groupby(['super_container_id', 'container_id', 'session_type']).apply(get_exposure_number_for_group)
    return experiments


def reformat_experiments_table(experiments):
    experiments = experiments.reset_index()
    experiments['super_container_id'] = experiments['specimen_id'].values
    # clean up cre_line naming
    experiments['cre_line'] = [driver_line[1] if driver_line[0] == 'Camk2a-tTA' else driver_line[0] for driver_line in
                               experiments.driver_line.values]
    experiments = experiments[experiments.cre_line != 'Cux2-CreERT2']  # why is this here?
    # replace session types that are NaN with string None
    experiments.at[experiments[experiments.session_type.isnull()].index.values, 'session_type'] = 'None'
    experiments = add_mouse_seeks_fail_tags_to_experiments_table(experiments)
    experiments = add_exposure_number_to_experiments_table(experiments)
    if 'level_0' in experiments.columns:
        experiments = experiments.drop(columns='level_0')
    if 'index' in experiments.columns:
        experiments = experiments.drop(columns='index')
    experiments = add_location_to_expts(experiments)
    return experiments



def add_all_qc_states_to_ophys_session_table(session_table):
    """ Add 'experiment_workflow_state', 'container_workflow_state', and 'session_workflow_state' to session_table.
            :param session_table: session_table from SDK cache
            :return: session_table: with additional columns added
            """
    experiment_table = loading.get_filtered_ophys_experiment_table(include_failed_data=True)
    session_table = add_session_workflow_state_to_ophys_session_table(session_table, experiment_table)
    session_table = add_container_workflow_state_to_ophys_session_table(session_table, experiment_table)
    return session_table


def add_session_workflow_state_to_ophys_session_table(session_table, experiment_table):
    """
    Define session_workflow_state as 'passing' if at least one of the experiments from that session passed QC.
    If all experiments failed, it is likely a session level failure cause, such as abehavior failure.
    :param session_table: session_table from SDK cache
    :return: session_table: with additional column for session_workflow_state added
    """
    passed_experiments = experiment_table[experiment_table.experiment_workflow_state == 'passed'].copy()
    session_ids = session_table.index.values
    session_table['at_least_one_experiment_passed'] = [any(passed_experiments['ophys_session_id'] == x) for x in
                                                      session_ids]
    session_table['session_workflow_state'] = ['passed' if criterion == True else 'failed' for criterion in
                                               session_table.at_least_one_experiment_passed.values]
    return session_table

def add_container_workflow_state_to_ophys_session_table(session_table, experiment_table):
    """
        Add 'container_workflow_state' to session_table by merging with experiment_table
        :param session_table: session_table from SDK cache
        :return: session_table: with additional column for container_workflow_state added
        """
    session_table = session_table.reset_index()
    session_table = session_table[session_table.ophys_session_id.isin(experiment_table.ophys_session_id.unique())]
    experiments = experiment_table[['ophys_session_id', 'container_id', 'container_workflow_state']].drop_duplicates(
        ['ophys_session_id'])
    session_table = session_table.merge(experiments, left_on='ophys_session_id', right_on='ophys_session_id')
    session_table = session_table.set_index(keys='ophys_session_id')
    return session_table


# REFORMATING SDK SESSION OBJECT DATA #
