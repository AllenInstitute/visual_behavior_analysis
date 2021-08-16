

# FILTERING MANIFEST TABLES #

def limit_to_passed_experiments(table):
    """
        :param table: experiments_table from SDK manifest or sessions_table with 'experiment_workflow_state' added
        :return: table with rows where experiment_workflow_state is 'passed'.
        """
    table = table[table.experiment_workflow_state == 'passed']
    return table


def limit_to_experiments_with_final_qc_state(table):
    """
        :param table: experiments_table from SDK manifest or sessions_table with 'experiment_workflow_state' added
        :param table: experiments_table from SDK manifest or sessions_table with 'experiment_workflow_state' added
        :return: table with rows where experiment_workflow_state is 'passed' or 'failed'.
                Excludes experiment_workflow_state = 'created' or 'qc',
        """
    table = table[table.experiment_workflow_state.isin(['passed', 'failed'])]
    return table


def limit_to_passed_containers(table):
    """
           :param table: experiments_table or sessions_table from SDK manifest with column 'container_workflow_state'
           :return: table rows where container_workflow_state is in one of the states indicating passing QC.
           """
    table = table[table.container_workflow_state.isin(['container_qc', 'completed', 'published'])]
    return table


def remove_failed_containers(table):
    """
           :param table: experiments_table or sessions_table from SDK manifest with column 'container_workflow_state'
           :return: table rows where container_workflow_state is not 'failed'. Includes 'passed' and 'holding' containers.
           """
    table = table[table.container_workflow_state != 'failed']  # include containers in holding
    return table


def limit_to_passed_ophys_sessions(session_table):
    """
        'session_workflow_state' is created according to criteria in reformat.add_session_workflow_state_to_ophys_session_table(session_table)
        :param table: session_table with 'session_workflow_state' added
        :return: session_table with rows where session_workflow_state is 'passed'.
        """
    session_table = session_table[session_table.session_workflow_state == 'passed']
    return session_table


def limit_to_production_project_codes(table):
    """ filter out data where the value of 'project_code' column does not belong to the list of production project codes"""
    table = table[table.project_code.isin(['VisualBehavior', 'VisualBehaviorTask1B',
                                           'VisualBehaviorMultiscope', 'VisualBehaviorMultiscope4areasx2d'])]
    return table


def limit_to_Scientifica_data(table):
    """ filter out data where the value of 'project_code' column does not belong to the list of Scientifica production project codes"""
    table = table[table.project_code.isin(['VisualBehavior', 'VisualBehaviorTask1B'])]
    return table


def limit_to_Multiscope_data(table):
    """ filter out data where the value of 'project_code' column does not belong to the list of Multiscope production project codes"""
    table = table[table.project_code.isin(['VisualBehaviorMultiscope', 'VisualBehaviorMultiscope4areasx2d'])]
    return table


def limit_to_valid_ophys_session_types(table):
    table = table[table.session_type.isin(['OPHYS_1_images_A', 'OPHYS_2_images_A_passive', 'OPHYS_3_images_A',
                                           'OPHYS_4_images_B', 'OPHYS_5_images_B_passive', 'OPHYS_6_images_B',
                                           'OPHYS_1_images_B', 'OPHYS_2_images_B_passive', 'OPHYS_3_images_B',
                                           'OPHYS_4_images_A', 'OPHYS_5_images_A_passive', 'OPHYS_6_images_A',
                                           'OPHYS_1_images_G', 'OPHYS_2_images_G_passive', 'OPHYS_3_images_G',
                                           'OPHYS_4_images_H', 'OPHYS_5_images_H_passive', 'OPHYS_6_images_H'])]
    return table


def get_experiment_ids_that_pass_qc(experiments_table):
    filtered_experiment_ids = experiments_table[experiments_table.experiment_workflow_state == 'passed'].ophys_experiment_id.unique()
    return filtered_experiment_ids


# def get_first_passing_novel_image_exposure_experiment_ids(experiments_table):
    # note: get_passed_experiments_from_experiments_table is not defined. Commenting out whole function to pass linter
    # data = get_passed_experiments_from_experiments_table(experiments_table)
    # data = data[data.session_type.isin(['OPHYS_4_images_B', 'OPHYS_4_images_A', 'OPHYS_4_images_H'])]
    # data = data[data.exposure_number == 0]
    # filtered_experiment_ids = data.ophys_experiment_id.unique()
    # return filtered_experiment_ids


def get_first_novel_image_exposure_experiment_ids(experiments_table):
    data = experiments_table.copy()
    data = data[data.session_type.isin(['OPHYS_4_images_B', 'OPHYS_4_images_A', 'OPHYS_4_images_H'])]
    data = data[data.exposure_number == 0]
    filtered_experiment_ids = data.ophys_experiment_id.unique()
    return filtered_experiment_ids


# def get_first_passing_omission_exposure_experiment_ids(experiments_table):
    # note: get_passed_experiments_from_experiments_table is not defined. Commenting out whole function to pass linter
    # data = get_passed_experiments_from_experiments_table(experiments_table)
    # data = data[data.session_type.isin(['OPHYS_1_images_B', 'OPHYS_1_images_A', 'OPHYS_1_images_G'])]
    # data = data[data.exposure_number == 0]
    # filtered_experiment_ids = data.ophys_experiment_id.unique()
    # return filtered_experiment_ids


def get_first_omission_exposure_experiment_ids(experiments_table):
    data = experiments_table.copy()
    data = data[data.session_type.isin(['OPHYS_1_images_B', 'OPHYS_1_images_A', 'OPHYS_1_images_G'])]
    data = data[data.exposure_number == 0]
    filtered_experiment_ids = data.ophys_experiment_id.unique()
    return filtered_experiment_ids
