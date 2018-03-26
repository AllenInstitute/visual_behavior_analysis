from visual_behavior.translator import foraging


def test_data_to_monolith(foraging_data_fixture):
    foraging.data_to_monolith(foraging_data_fixture)
