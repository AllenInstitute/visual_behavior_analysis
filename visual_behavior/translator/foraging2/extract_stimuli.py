

def get_image_changes(change_log):
    changes = []
    for _, stim_params, time, frame in change_log:
        changes.append(dict(
            image_category=stim_params[0],
            image_name=stim_params[1],
            time=time,
            frame=frame,
        ))
    return changes


def get_grating_changes(change_log):
    changes = []
    for _, stim_params, time, frame in change_log:
        changes.append(dict(
            orientation=stim_params[1],
            time=time,
            frame=frame,
        ))
    return changes
