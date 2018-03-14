import pandas as pd

from .extract import get_trial_log, get_stimuli, get_pre_change_time, annotate_licks,\
    annotate_rewards, annotate_optogenetics, annotate_responses, annotate_schedule_time, \
    annotate_stimuli, get_datetime_info, get_user_id, get_mouse_id, \
    get_blank_duration_range, get_session_id, get_filename, get_device_name, \
    get_session_duration, get_stimulus_duration, get_task_id, get_image_path, \
    get_scheduled_trial_duration, annotate_trials, get_response_window, get_licks, \
    get_running_speed, get_params, get_time


def expand_dict(out_dict, from_dict, index):
    """there is obviously a better way...
    """
    for k, v in from_dict.items():
        if not out_dict.get(k):
            out_dict[k] = {}

        out_dict.get(k)[index] = v


def data_to_licks(data):
    return get_licks(data)


def data_to_monolith(data):
    stimuli = get_stimuli(data)
    trial_log = get_trial_log(data)
    pre_change_time = get_pre_change_time(data)
    experiment_params = {
        "mouse_id": get_mouse_id(data),
        "user_id": get_user_id(data),
        "session_id": get_session_id(data),
        "filename": get_filename(data),
        "device_name": get_device_name(data),
        "session_duration": get_session_duration(data),
        "stimulus_duration": get_stimulus_duration(data),
        "blank_duration_range": get_blank_duration_range(data),
        "pre_change_time": pre_change_time,
        "task_id": get_task_id(data),
        "image_path": get_image_path(data),
        "scheduled_trial_duration": get_scheduled_trial_duration(data),
        "response_window": get_response_window(data),
        **get_datetime_info(data),
    }

    monolith = {}

    for trial in trial_log:
        index = trial["index"]  # trial index

        expand_dict(monolith, annotate_licks(trial), index)
        expand_dict(monolith, annotate_rewards(trial), index)
        expand_dict(monolith, annotate_optogenetics(trial), index)
        expand_dict(monolith, annotate_responses(trial), index)
        expand_dict(monolith, annotate_schedule_time(trial, pre_change_time), index)
        expand_dict(monolith, annotate_stimuli(trial, stimuli), index)
        expand_dict(monolith, annotate_trials(trial), index)
        expand_dict(monolith, experiment_params, index)

    return pd.DataFrame(data=monolith)


def data_to_params(data):
    return get_params(data)


def data_to_running_speed(data):
    return get_running_speed(data)


def data_to_time(data):
    return get_time(data)
