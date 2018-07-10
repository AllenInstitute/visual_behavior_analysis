
from visual_behavior.schemas.core import MetadataSchema, StimulusSchema, \
    RunningSchema, LickSchema, RewardSchema, TrialSchema
from visual_behavior.schemas.extended_trials import ExtendedTrialSchema
from visual_behavior.translator import foraging2, foraging
from visual_behavior.translator.core import create_extended_dataframe


"""test the schemas vs the outputs here
"""

def _test_core_data_schemas(core_data):

    # metadata

    # core dataframes
    dataframe_schemas = (
        (StimulusSchema, core_data['visual_stimuli']),
        (RunningSchema, core_data['running']),
        (LickSchema, core_data['licks']),
        (RewardSchema, core_data['rewards']),
        (TrialSchema, core_data['trials']),
    )

    for Schema, data in dataframe_schemas:
        errors = Schema(many=True).validate(data.to_dict(orient='records'))

        for row, row_errors in errors.items():
            assert len(row_errors)==0, (Schema, data, row_errors)

    extended_trials = create_extended_dataframe(
        trials=core_data['trials'],
        metadata=core_data['metadata'],
        licks=core_data['licks'],
        time=core_data['time'],
    )

    errors = ExtendedTrialSchema(many=True).validate(
        extended_trials.to_dict(orient='records')
    )

    for row, row_errors in errors.items():
        assert len(row_errors)==0, row_errors

    errors = MetadataSchema().validate(core_data['metadata'])
    assert len(errors)==0, errors.keys()


def test_foraging2_translator_schema(foraging2_data_stage4_2018_05_10):
    core_data = foraging2.data_to_change_detection_core(
        foraging2_data_stage4_2018_05_10
    )
    _test_core_data_schemas(core_data)

def test_foraging_translator_schema(behavioral_session_output_fixture):
    core_data = foraging.data_to_change_detection_core(
        behavioral_session_output_fixture
    )
    _test_core_data_schemas(core_data)
