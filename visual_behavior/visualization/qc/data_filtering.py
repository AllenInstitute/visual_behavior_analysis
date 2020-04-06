

def get_experiment_ids_that_pass_qc(experiments_table):
    filtered_experiment_ids = experiments_table[experiments_table.experiment_workflow_state == 'passed'].ophys_experiment_id.unique()
    return filtered_experiment_ids


def get_first_passing_novel_image_exposure_experiment_ids(experiments_table):
    data = experiments_table[experiments_table.experiment_workflow_state == 'passed'].copy()
    data = data[data.session_type.isin(['OPHYS_4_images_B', 'OPHYS_4_images_A', 'OPHYS_4_images_H'])]
    data = data[data.exposure_number == 0]
    filtered_experiment_ids = data.ophys_experiment_id.unique()
    return filtered_experiment_ids


def get_first_novel_image_exposure_experiment_ids(experiments_table):
    data = experiments_table.copy()
    data = data[data.session_type.isin(['OPHYS_4_images_B', 'OPHYS_4_images_A', 'OPHYS_4_images_H'])]
    data = data[data.exposure_number == 0]
    filtered_experiment_ids = data.ophys_experiment_id.unique()
    return filtered_experiment_ids


def get_first_passing_omission_exposure_experiment_ids(experiments_table):
    data = experiments_table[experiments_table.experiment_workflow_state == 'passed'].copy()
    data = data[data.session_type.isin(['OPHYS_1_images_B', 'OPHYS_1_images_A', 'OPHYS_1_images_G'])]
    data = data[data.exposure_number == 0]
    filtered_experiment_ids = data.ophys_experiment_id.unique()
    return filtered_experiment_ids


def get_first_omission_exposure_experiment_ids(experiments_table):
    data = experiments_table.copy()
    data = data[data.session_type.isin(['OPHYS_1_images_B', 'OPHYS_1_images_A', 'OPHYS_1_images_G'])]
    data = data[data.exposure_number == 0]
    filtered_experiment_ids = data.ophys_experiment_id.unique()
    return filtered_experiment_ids
