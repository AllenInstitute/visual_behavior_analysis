from visual_behavior.ophys.io.lims_database import LimsDatabase

def test_lims_database(ophys_experiment_id):
    
    lims_data = LimsDatabase(ophys_experiment_id)
    assert lims_data.specimen_id == 652073919
    assert lims_data.session_id == 702013508
    assert lims_data.experiment_container_id == 700821114
    assert lims_data.project_id == 'VisualBehaviorDevelopment'
    assert lims_data.external_specimen_id == 363887
    assert lims_data.experiment_name == '20180524_363887_sessionC'
    assert str(lims_data.experiment_date) == '2018-05-24 14:27:25'
    assert lims_data.targeted_structure == 'VISal'
    assert lims_data.imaging_depth == 175
    assert lims_data.stimulus_type is None
    assert lims_data.operator == 'saharm'
    assert lims_data.rig == 'CAM2P.6'
    assert lims_data.workflow_state == 'qc'
    assert lims_data.data_folder == '/allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/ophys_experiment_702134928/'
    assert lims_data.sync_file == '/allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/702013508_363887_20180524142941_sync.h5'
    assert lims_data.specimen_id == 652073919
    assert lims_data.stim_file == '/allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/702013508_363887_20180524142941_stim.pkl'
    assert 'objectlist.txt' in lims_data.curr_objectlist_file
    assert '_input_extract_traces.json' in lims_data.curr_input_extract_traces_file
    assert lims_data.dff_file == '/allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/ophys_experiment_702134928/702134928_dff.h5'
    assert lims_data.demix_file == '/allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/ophys_experiment_702134928/demix/702134928_demixed_traces.h5'
    assert lims_data.rigid_motion_transform_file == '/allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/ophys_experiment_702134928/processed/702134928_rigid_motion_transform.csv'
    assert lims_data.maxInt_a13_file == '/allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/ophys_experiment_702134928/processed/ophys_cell_segmentation_run_814561221/maxInt_a13a.png'
    assert lims_data.roi_traces_file == '/allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/ophys_experiment_702134928/processed/roi_traces.h5'
    assert lims_data.avgint_a1X_file == '/allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/ophys_experiment_702134928/processed/ophys_cell_segmentation_run_814561221/avgInt_a1X.png'
