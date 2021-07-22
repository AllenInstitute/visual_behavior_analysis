import pandas as pd
from pandas.testing import assert_frame_equal
from visual_behavior.data_access.reformat import add_engagement_state_to_trials_table

def test_add_engagement_state_to_trials_table():
    extended_stimulus_presentations = pd.DataFrame(
        [
            {'stimulus_presentations_id': 188, 'start_time': 450.40741, 'engaged': False, 'engagement_state': 'disengaged'},
            {'stimulus_presentations_id': 189, 'start_time': 451.15798, 'engaged': False, 'engagement_state': 'disengaged'},
            {'stimulus_presentations_id': 190, 'start_time': 451.90859, 'engaged': False, 'engagement_state': 'disengaged'},
            {'stimulus_presentations_id': 191, 'start_time': 452.65921, 'engaged': False, 'engagement_state': 'disengaged'},
            {'stimulus_presentations_id': 192, 'start_time': 453.40984, 'engaged': False, 'engagement_state': 'disengaged'},
            {'stimulus_presentations_id': 193, 'start_time': 454.16046, 'engaged': False, 'engagement_state': 'disengaged'},
            {'stimulus_presentations_id': 194, 'start_time': 454.91107, 'engaged': False, 'engagement_state': 'disengaged'},
            {'stimulus_presentations_id': 195, 'start_time': 455.66172, 'engaged': False, 'engagement_state': 'disengaged'},
            {'stimulus_presentations_id': 196, 'start_time': 456.4123,  'engaged': False, 'engagement_state': 'disengaged'},
            {'stimulus_presentations_id': 197, 'start_time': 457.16291, 'engaged': False, 'engagement_state': 'disengaged'},
            {'stimulus_presentations_id': 198, 'start_time': 457.91351, 'engaged': False, 'engagement_state': 'disengaged'},
            {'stimulus_presentations_id': 199, 'start_time': 458.66411, 'engaged': True, 'engagement_state': 'engaged'},
            {'stimulus_presentations_id': 200, 'start_time': 459.41474, 'engaged': True, 'engagement_state': 'engaged'},
            {'stimulus_presentations_id': 201, 'start_time': 460.16537, 'engaged': True, 'engagement_state': 'engaged'},
            {'stimulus_presentations_id': 202, 'start_time': 460.916, 'engaged': True, 'engagement_state': 'engaged'},
            {'stimulus_presentations_id': 203, 'start_time': 461.66659, 'engaged': True, 'engagement_state': 'engaged'},
            {'stimulus_presentations_id': 204, 'start_time': 462.4172, 'engaged': True, 'engagement_state': 'engaged'},
            {'stimulus_presentations_id': 205, 'start_time': 463.16782, 'engaged': True, 'engagement_state': 'engaged'},
            {'stimulus_presentations_id': 206, 'start_time': 463.91841, 'engaged': True, 'engagement_state': 'engaged'},
            {'stimulus_presentations_id': 207, 'start_time': 464.66905, 'engaged': True, 'engagement_state': 'engaged'}
        ]
    ).set_index('stimulus_presentations_id')

    trials = pd.DataFrame(
        [
            {'start_time': 451.14132},
            {'start_time': 459.39807},
            {'start_time': 461.64992}
        ]
    )
    
    expected_result = pd.DataFrame([
        {'start_time': 451.14132, 'first_stim_presentation_index': 189.0, 'stimulus_presentations_id': 189, 'engaged': False, 'engagement_state': 'disengaged'},
        {'start_time': 459.39807, 'first_stim_presentation_index': 200.0, 'stimulus_presentations_id': 200, 'engaged': True, 'engagement_state': 'engaged'},
        {'start_time': 461.64992, 'first_stim_presentation_index': 203.0, 'stimulus_presentations_id': 203, 'engaged': True, 'engagement_state': 'engaged'}
    ])
    
    assert_frame_equal(
        expected_result,
        add_engagement_state_to_trials_table(trials, extended_stimulus_presentations)
    )