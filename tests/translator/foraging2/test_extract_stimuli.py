from visual_behavior.translator.foraging2.extract_stimuli import get_image_changes, get_grating_changes

IMAGES_CHANGE_LOG = [
    (('im111', 'im111'), ('im053', 'im053'), 4.475209915492957, 229),
    (('im053', 'im053'), ('im111', 'im111'), 16.250224375586853, 965),
    (('im111', 'im111'), ('im037', 'im037'), 22.876219192488264, 1379),
]

EXPECTED_IMAGES_CHANGES = [
    {
        'frame': 229,
        'time': 4.475209915492957,
        'image_category': 'im053',
        'image_name': 'im053',
        'prior_image_category': 'im111',
        'prior_image_name': 'im111',
    },
    {
        'frame': 965,
        'time': 16.250224375586853,
        'image_category': 'im111',
        'image_name': 'im111',
        'prior_image_category': 'im053',
        'prior_image_name': 'im053',
    },
    {
        'frame': 1379,
        'time': 22.876219192488264,
        'image_category': 'im037',
        'image_name': 'im037',
        'prior_image_category': 'im111',
        'prior_image_name': 'im111',
    },
]

GRATING_CHANGE_LOG = [
    (('group0', 0), ('group1', 90), 6.558310159624413, 367),
    (('group1', 90), ('group0', 0), 16.127468169014083, 965),
    (('group0', 0), ('group1', 90), 21.279577840375588, 1287),
]

EXPECTED_GRATING_CHANGES = [
    {
        'frame': 367,
        'time': 6.558310159624413,
        'orientation': 90,
        'prior_orientation': 0,
    },
    {
        'frame': 965,
        'time': 16.127468169014083,
        'orientation': 0,
        'prior_orientation': 90,
    },
    {
        'frame': 1287,
        'time': 21.279577840375588,
        'orientation': 90,
        'prior_orientation': 0,
    },
]


def test_get_image_changes():
    changes = get_image_changes(IMAGES_CHANGE_LOG)
    assert changes == EXPECTED_IMAGES_CHANGES


def test_get_grating_changes():
    changes = get_grating_changes(GRATING_CHANGE_LOG)
    assert changes == EXPECTED_GRATING_CHANGES
