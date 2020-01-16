import numpy as np
from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession
from visual_behavior_analysis.visual_behavior.translator.allensdk_sessions import attribute_formatting as af

def VisualBehaviorSession():
    def __init__(self):
        print("")
        #  super().__init__(api)

#    def get_licks(self):
#        #  licks_original = api.get_licks()
#        return np.arange(5)





# TODO: update this for allensdk sessions
def data_to_change_detection_core(data, time=None):
    """Core data structure to be used across all analysis code?

    Parameters
    ----------
    data: Mapping
        foraging2 style output data structure

    Returns
    -------
    pandas.DataFrame
        core data structure for the change detection task

    Notes
    -----
    - currently doesn't require or check that the `task` field in the
    experiment data is "DoC" (Detection of Change)
    """

    if time is None:
        time = data_to_time(data)

    log_messages = []
    handler = ListHandler(log_messages)
    handler.setFormatter(
        DoubleColonFormatter
    )
    handler.setLevel(logging.INFO)

    logger.addHandler(
        handler
    )

    core_data = {
        "metadata": data_to_metadata(data),
        "time": time,
        "licks": data_to_licks(data, time=time),
        "trials": data_to_trials(data, time=time),
        "running": data_to_running(data, time=time),
        "rewards": data_to_rewards(data, time=time),
        "visual_stimuli": data_to_visual_stimuli(data, time=time),
        "omitted_stimuli": data_to_omitted_stimuli(data, time=time),
        "image_set": data_to_images(data),
    }

    core_data['log'] = log_messages

    return core_data
