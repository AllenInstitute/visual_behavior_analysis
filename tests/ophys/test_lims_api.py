import pytest

from visual_behavior.ophys.io.lims_api import VisualBehaviorLimsAPI


@pytest.mark.parametrize('ophys_experiment_id, driver_line', [
    (789359614, ['Camk2a-tTA', 'Slc17a7-IRES2-Cre']),
    (702134928, ['Vip-IRES-Cre'])])
def test_get_metadata(ophys_experiment_id, driver_line):

    api = VisualBehaviorLimsAPI()
    assert api.get_driver_line(ophys_experiment_id) == driver_line


def test_lims_api():

    oeid = 702134928

    api = VisualBehaviorLimsAPI()

    TD = {'ophys_dir':'/allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/ophys_experiment_702134928/',
         'demix_file':'/allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/ophys_experiment_702134928/demix/702134928_demixed_traces.h5',
         'avgint_a1X_file':'/allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/ophys_experiment_702134928/processed/ophys_cell_segmentation_run_814561221/avgInt_a1X.png',
         'rigid_motion_transform_file':'/allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/ophys_experiment_702134928/processed/702134928_rigid_motion_transform.csv',
         'targeted_structure':'VISal',
         'imaging_depth':175,
         'stimulus_name':None,
         'reporter_line':'Ai148(TIT2L-GC6f-ICL-tTA2)',
         'driver_line':['Vip-IRES-Cre'],
         'LabTracks_ID':'363887',
         'full_genotype':'Vip-IRES-Cre/wt;Ai148(TIT2L-GC6f-ICL-tTA2)/wt'
         }

    assert api.get_ophys_experiment_dir(oeid) == TD['ophys_dir']
    assert api.get_demix_file(oeid) == TD['demix_file']
    assert api.get_avgint_a1X_file(oeid) == TD['avgint_a1X_file']
    assert api.get_rigid_motion_transform_file(oeid) == TD['rigid_motion_transform_file']
    assert api.get_targeted_structure(oeid) == TD['targeted_structure']
    assert api.get_imaging_depth(oeid) == TD['imaging_depth']
    assert api.get_stimulus_name(oeid) == TD['stimulus_name']
    assert str(api.get_experiment_date(oeid)) == '2018-05-24 21:27:25'
    assert api.get_reporter_line(oeid) == TD['reporter_line']
    assert api.get_driver_line(oeid) == TD['driver_line']
    assert api.get_LabTracks_ID(oeid) == TD['LabTracks_ID']
    assert api.get_full_genotype(oeid) == TD['full_genotype']