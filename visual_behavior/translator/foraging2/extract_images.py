def get_image_metadata(data):

    metadata = dict(
        image_set=data["items"]["behavior"]["stimuli"]["images"]["image_path"]
    )

    return metadata
