import pandas as pd
import time
import pytest
import os

from visual_behavior.translator.foraging2 import data_to_change_detection_core
from visual_behavior.translator.core import create_extended_dataframe
from visual_behavior.validation.qc import generate_qc_report
from visual_behavior.validation.qc import define_validation_functions
from visual_behavior.change_detection.trials.summarize import session_level_summary
# some sessions that have been manually validated. These should all pass the QC functions.
sessions = {
    'TRAINING_1_gratings_412629':'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/test_fixtures/181119092559_412629_a3775e3e-e1ca-474a-b413-91cccd6d886f.pkl',
    'TRAINING_1_gratings_421137': '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/test_fixtures/181119102010_421137_c108dc71-ef5e-46ad-8d85-8da0fdaf7d3d.pkl',
    'TRAINING_2_gratings_flashed_416656': '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/test_fixtures/181119150503_416656_2b0893fe-843d-495e-bceb-83b13f2b02dc.pkl',
    'TRAINING_3_images_A_10uL_reward_424460': '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/test_fixtures/181119135416_424460_b6daf247-2caf-4f38-9eb1-ab97825923cd.pkl',
    'TRAINING_4_images_A_handoff_ready_402329': '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/test_fixtures/181119134201_402329_b75a87d0-8178-4171-a3b2-7cea3ae8e118.pkl',
    'OPHYS_IMAGES_A_412364': '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/test_fixtures/778113069_stim.pkl',
    'TRAINING_2_gratings_flashed_428136': '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/test_fixtures/190124093821_428136_d22c9f16-12e6-4b21-b04c-843495560b5a.pkl'
}

# look up pytest skipif, skipif(~os.path.exists(//allen))


class DataCheck(object):
    # a simple class for loading data, running qc
    def __init__(self, pkl_path):
        self.pkl_path = pkl_path
        self.load_data()
        self.run_qc()
        self.make_session_summary()

    def load_data(self):

        self.data = pd.read_pickle(self.pkl_path)
        self.core_data = data_to_change_detection_core(self.data)
        self.trials = create_extended_dataframe(**self.core_data)

    def run_qc(self):
        validation_functions = define_validation_functions(self.core_data)
        for func in validation_functions:
            assert func(*validation_functions[func]) == True, 'failed on {}'.format(func.__name__)
        self.qc = {'passes':True}

    def make_session_summary(self):
        session_summary = session_level_summary(self.trials)
        print(session_summary)




@pytest.mark.parametrize("session_key, filename", sessions.items())
@pytest.mark.skipif(not os.path.exists('//allen/programs/braintv'), reason="no access to network path, skipping test on network PKL files")
def test_sessions(session_key, filename):
    data_result = DataCheck(filename)
    assert(data_result.qc['passes'] == True)


if __name__ == '__main__':
    # test_sessions(*list(sessions.items())[-1])

    for session_key,filename in sessions.items():
        test_sessions(session_key, filename)
    print('all sessions passed')
