import numpy as np
import pandas as pd
from six import iteritems
from .extract import get_time


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


def change_records_to_dataframe(change_records):
    return pd.DataFrame(
        change_records,
        columns=['frame', 'time', 'image_category', 'image_name', 'orientation'],
    )


def find_change_indices(changes, draw_log):

    indices = np.searchsorted(
        changes['frame'],
        draw_log['frame'],
        side='right',
    )

    return indices - 1


def get_visual_stimuli(stimuli, time):
    visual_stimuli = []

    for stim_category_name, stim_dict in iteritems(stimuli):
        if stim_category_name == 'images':
            changes = get_image_changes(stim_dict['change_log'])
        elif stim_category_name == 'grating':
            changes = get_grating_changes(stim_dict['change_log'])
        else:
            raise NotImplementedError("we don't know how to parse {} stimuli".format(stim_category_name))

        changes = change_records_to_dataframe(changes)
        draw_log = get_draw_log(stim_dict, time)
        draw_log = annotate_end_frames(draw_log)
        draw_log = reduce_to_onsets(draw_log)
        draw_log['change_index'] = find_change_indices(changes, draw_log)
        viz = draw_log.merge(
            changes[['image_name', 'image_category', 'orientation']],
            how='left',
            left_on='change_index',
            right_index=True,
        ).reset_index()[['frame', 'time', 'image_name', 'image_category', 'orientation']]

        visual_stimuli.append(viz)
    visual_stimuli = pd.concat(visual_stimuli)

    return visual_stimuli


def get_draw_log(stim_dict, time=None):

    if time is None:
        time = get_time(data)

    draw_log = stim_dict['draw_log']
    draw_log = (
        pd.DataFrame({'time': time, 'draw': draw_log})
        .reset_index()
        .rename(columns={'index': 'frame'})
    )
    return draw_log


class EndOfRunFinder(object):
    def __init__(self):
        self.end_frame = None

    def check(self, log):
        if log['draw'] == 0:
            self.end_frame = log['frame']
        elif self.end_frame is None:
            self.end_frame = log['frame'] + 1
        return self.end_frame


def annotate_end_frames(draw_log):
    draw_log = draw_log.sort_values('frame', ascending=False)
    draw_log['end_frame'] = draw_log.apply(EndOfRunFinder().check, axis=1)
    draw_log = draw_log.sort_values('frame', ascending=True)
    return draw_log


def reduce_to_onsets(draw_log):
    flash_onsets = draw_log['draw'].diff().fillna(1.0) > 0
    draw_log = draw_log[flash_onsets]
    return draw_log
