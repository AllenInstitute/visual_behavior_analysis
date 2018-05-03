import pytest

from visual_behavior import validation


@pytest.mark.fixture
def perfect_core_data_fixture():
    return {}


def test_generate_validation_functions(perfect_core_data_fixture):
    assert validation.generate_validation_metrics(perfect_core_data_fixture) == \
        {"passes": True, }
