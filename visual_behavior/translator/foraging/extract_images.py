

def get_image_metadata(data):

    metadata = dict(
        image_set=data['stimulus'],
    )

    return metadata


def get_image_data(image_dict):

    images = []
    images_meta = []

    ii = 0
    for cat, cat_images in image_dict.items():
        for img_name, img in cat_images.items():
            meta = dict(
                image_category=cat,
                image_name=img_name,
                image_index=ii,
            )

            images.append(img)
            images_meta.append(meta)

            ii += 1
    return images, images_meta
