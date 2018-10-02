def get_image_metadata(data):
    
    metadata = dict(
        image_set=data[b"items"][b"behavior"][b"stimuli"][b"images"][b"image_path"]
    )

    return metadata