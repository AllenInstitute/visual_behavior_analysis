import numpy as np
from numpy.testing import assert_allclose
from visual_behavior.translator.foraging import extract_images

EXPECTED_METADATA = {
    'image_set': 'natural_images_eight_CAM_matched_2017.01.19.pkl',
}

EXPECTED_IMAGE_MEANS = np.array(
    [
    124.760094, 121.14325, 120.433488, 126.095071,
    119.088306, 125.41706, 119.367858, 125.626576,
    ],
)

EXPECTED_IMAGE_META = [
    {'image_category': 0, 'image_name': 'img061_VH.tiff', 'image_index': 0},
    {'image_category': 1, 'image_name': 'img062_VH.tiff', 'image_index': 1},
    {'image_category': 2, 'image_name': 'img063_VH.tiff', 'image_index': 2},
    {'image_category': 3, 'image_name': 'img065_VH.tiff', 'image_index': 3},
    {'image_category': 4, 'image_name': 'img066_VH.tiff', 'image_index': 4},
    {'image_category': 5, 'image_name': 'img069_VH.tiff', 'image_index': 5},
    {'image_category': 6, 'image_name': 'img077_VH.tiff', 'image_index': 6},
    {'image_category': 7, 'image_name': 'img085_VH.tiff', 'image_index': 7},
]

def test_get_image_metadata(foraging_data_fixture):

    image_metadata = extract_images.get_image_metadata(foraging_data_fixture)
    assert image_metadata == EXPECTED_METADATA


def test_get_image_data(foraging_data_fixture):

    image_dict = foraging_data_fixture['image_dict']
    images, images_meta = extract_images.get_image_data(image_dict)
    means = np.array([img.mean() for img in images])

    assert_allclose(means, EXPECTED_IMAGE_MEANS)
    assert images_meta == EXPECTED_IMAGE_META
