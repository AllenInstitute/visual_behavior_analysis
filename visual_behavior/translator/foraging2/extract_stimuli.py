import numpy as np
import pandas as pd
from six import iteritems


def get_image_changes(change_log):
    changes = []
    for source_params, dest_params, time, frame in change_log:
        changes.append(dict(
            prior_image_category=source_params[0],
            prior_image_name=source_params[1],
            image_category=dest_params[0],
            image_name=dest_params[1],
            time=time,
            frame=frame,
        ))
    return changes


def get_grating_changes(change_log):
    changes = []
    for source_params, dest_params, time, frame in change_log:
        changes.append(dict(
            prior_orientation=source_params[1],
            orientation=dest_params[1],
            time=time,
            frame=frame,
        ))
    return changes


def change_records_to_dataframe(change_records):
    first_record = dict(
        frame=0,
        time=None,
    )
    try:
        image_name = change_records[0]['prior_image_name']
        image_category = change_records[0]['prior_image_category']
        first_record.update(image_name=image_name)
        first_record.update(image_category=image_category)
    except KeyError:
        orientation = change_records[0]['prior_orientation']
        first_record.update(orientation=orientation)

    change_records.insert(0, first_record)

    changes = pd.DataFrame(
        change_records,
        columns=[
            'frame',
            'time',
            'image_category',
            'image_name',
            'orientation',
        ],
    )

    return changes


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
            changes[[
                'image_name',
                'image_category',
                'orientation',
            ]],
            how='left',
            left_on='change_index',
            right_index=True,
        ).reset_index()[['frame', 'end_frame', 'time', 'image_name', 'image_category', 'orientation']]

        # NEED CHANGES TO FORAGING2 FOR THIS TO WORK
        # viz['end_time'] = viz['end_frame'].map(lambda fr: time[int(fr)])
        # viz['duration'] = viz['end_time'] - viz['time']
        # del viz['end_time']

        visual_stimuli.append(viz)
    visual_stimuli = pd.concat(visual_stimuli)

    return visual_stimuli


def get_draw_log(stim_dict, time):

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


def _get_static_visual_stimuli(stim_dict, end_time=np.inf):
    """This is a quick hack function to create a visual stimuli dataframe for a
    static change detection task

    Parameters
    ----------
    stim_dict: Mapping
        foraging2 output stim_dict
    end_time: float, default=np.inf
        end time of the last trial in which a stimulus from 'stim_dict' is shown

    Returns
    -------
    pandas.DataFrame
        visual stimuli dataframe

    Notes
    -----
    - as of 05/10/2018, the 'stim_dict' is assumed to be a dictionary in the
    foraging2 output and is expected to be at:
        foraging2 output -> 'items' -> 'behavior' -> 'stimuli' -> <stim name> -> stim_dict
    - for an experiment with only one 'stim_dict', which is all experiments
    currently (05/10/2018), then:
        end_time == <end time of the last trial>
    """
    data = []
    for idx, (attr_name, attr_value, time, frame, ) in enumerate(stim_dict["set_log"]):
        contrast = attr_value if attr_name.lower() == "contrast" else np.nan
        orientation = attr_value if attr_name.lower() == "ori" else np.nan
        image_name = attr_value if attr_name.lower() == "image" else np.nan

        try:
            duration = stim_dict["set_log"][idx + 1][2] - time  # subtract time from the time of the next time stimuli set
        except IndexError:
            duration = end_time - time

        data.append({
            "contrast": contrast,
            "orientation": orientation,
            "image_name": image_name,
            "image_category": image_name,
            "frame": frame,
            "time": time,
            "duration": duration,
        })

    return pd.DataFrame(data=data)
