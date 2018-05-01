from visual_behavior.translator.foraging2.extract_stimuli import get_image_changes, get_grating_changes

IMAGES_CHANGE_LOG = [
    (('im111', 'im111'), ('im053', 'im053'), 4.475209915492957, 229),
    (('im053', 'im053'), ('im111', 'im111'), 16.250224375586853, 965),
    (('im111', 'im111'), ('im037', 'im037'), 22.876219192488264, 1379),
]

EXPECTED_IMAGES_CHANGES = [
    {'frame': 229, 'orientation': 'im053', 'time': 4.475209915492957},
    {'frame': 965, 'orientation': 'im111', 'time': 16.250224375586853},
    {'frame': 1379, 'orientation': 'im037', 'time': 22.876219192488264},
]

GRATING_CHANGE_LOG = [
    (('group0', 0), ('group1', 90), 6.558310159624413, 367),
    (('group1', 90), ('group0', 0), 16.127468169014083, 965),
    (('group0', 0), ('group1', 90), 21.279577840375588, 1287),
]

EXPECTED_GRATING_CHANGES = [
    {'frame': 367, 'orientation': 90, 'time': 6.558310159624413},
    {'frame': 965, 'orientation': 0, 'time': 16.127468169014083},
    {'frame': 1287, 'orientation': 90, 'time': 21.279577840375588},
]


def test_get_image_changes():
    changes = get_image_changes(IMAGES_CHANGE_LOG)
    assert changes == EXPECTED_IMAGES_CHANGES

    
def test_get_grating_changes():
    changes = get_grating_changes(GRATING_CHANGE_LOG)
    assert changes == EXPECTED_GRATING_CHANGES
