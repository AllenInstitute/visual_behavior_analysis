# from marshmallow import Schema, fields
#
#
# class ChangeDetectionCoreMetadataSchema(Schema):
#
#     startdatetime = fields.Str()
#     rig_id = fields.Str()
#     computer_name = fields.Str()
#     reward_vol = Float()
#     auto_reward_vol = fields.Number()
#     params = fields.Dict()
#     mouse_id = fields.Str()
#     response_window = fields.List()
#     task = fields.Str()
#     stage = fields.Str()
#     stop_time = fields.Number()
#     user_id = fields.Str()
#     lick_detect_training_mode = fields.Boolean()
#     blank_screen_on_timeout = fields.Boolean()
#     stim_duration = fields.Number()
#     blank_duration_range = fields.List()
#     delta_minimum = fields.Number()
#     stimulus_distribution = fields.List()
#     delta_mean = fields.Number()
#     trial_duration = fields.Number()
#     n_stimulus_frames = fields.Integer()
#
#
# class TimeSeriesSchema(Schema):
#
#     time = fields.Number()
#
#
# class StreamSchema(DataFrameSchema):
#
#     frame = fields.Integer()
#     time = fields.Number()
#
#
# class LicksSchema(StreamSchema):
#
#     volume = fields.Number()
#     lickspout = fields.Integer()
#
#
# class RewardsSchema(StreamSchema):
#
#     pass
#
#
# class RunningSchema(StreamSchema):
#
#     speed = fields.Number()
#
#
#
# class ChangeDetectionCoreSchema(Schema):
#
#     time = fields.Nested(TimeSeriesSchema, many=True)
#     licks = fields.Nested()
#
# # class ChangeDetectionCoreSchema
# #
# #
# # class ChangeDetectionCoreSchema(object):
# #
# #     metadata = dict
# #     time = list
# #     licks = Lick
# #     trials = trials
# #     stage =
#
# "metadata": data_to_metadata(data),
# "time": time or data_to_time(data),
# "licks": data_to_licks(data, time),
# "trials": data_to_trials(data, time),
# "running": data_to_running(data, time),
# "rewards": data_to_rewards(data, time),
# "visual_stimuli": None,  # not yet implemented
#
#
# {
#     "startdatetime": start_time_datetime.astimezone(tz.gettz("UTC")).isoformat(),
#     "rig_id": None,  # not obtainable because device_name is not obtainable
#     "computer_name": device_name,
#     "reward_vol": params["reward_volume"],
#     "auto_reward_vol": params["auto_reward_volume"],
#     "params": params,
#     "mouse_id": get_mouse_id(data),
#     "response_window": list(get_response_window(data)),  # tuple to list
#     "task": get_task_id(data),
#     "stage": None,  # not implemented currently
#     "stop_time": get_session_duration(data),
#     "user_id": get_user_id(data),
#     "lick_detect_training_mode": False,  # currently no choice
#     "blank_screen_on_timeout": None,  # not obtainable
#     "stim_duration": get_stimulus_duration(data) * 1000,  # seconds to milliseconds
#     "blank_duration_range": [
#         get_blank_duration_range(data)[0] * 1000,
#         get_blank_duration_range(data)[1] * 1000,
#     ],  # seconds to miliseconds
#     "delta_minimum": get_pre_change_time(data),
#     "stimulus_distribution": None,  # not obtainable
#     "delta_mean": None,  # not obtainable
#     "trial_duration": None,  # not obtainable
#     "n_stimulus_frames": n_stimulus_frames,
# }
