# from collections import MutableMapping

from .extract import get_dx, get_licks, get_params, get_rewards, \
    get_running_speed, get_stimulus_log, get_time, get_trials, get_vsyncs
from .transform import annotate_parameters, annotate_n_rewards, \
    annotate_reward_volume, annotate_change_detect
# from .extract import *
# from .transform import *
# the import order affects what stays and goes...this is so bad...TODO make less bad maybe later if you remember

# resolve liek.... {**extract}
"""name me something meaningful
"""


# class DoCDataFrameTranslator(MutableMapping):  # hopefully we wont use this long...
#
#     _output_structure = {
#         "wut": None,
#     }
#
#     def __init__(self, **values):
#         super(DoCDataFrameTranslator, self).__init__(**values)
#
#     @classmethod
#     def _coercer(self, cls, input):
#         raise NotImplementedError()
#
#     @classmethod
#     def _finalizer(self, cls, state):
#         return cls._output_structure.map(state)
#
#     def coerce(self):
#         self.__state = self.__class__._coercer(self.__state)
#
#     def finalize(self):
#         return self.__class__._finalizer(self.__state)
#
#
# class CreateDataBaseTranslator(CreateDoCDataFrame):
#     pass


def data_to_monolith(exp_data):
    trials = get_trials(exp_data)

    annotate_parameters(trials, exp_data)
    annotate_n_rewards(trials)
    annotate_reward_volume(trials, exp_data)
    annotate_change_detect(trials)

    return trials
