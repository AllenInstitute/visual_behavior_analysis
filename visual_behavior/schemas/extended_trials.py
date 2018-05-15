from marshmallow import fields

from .base import PandasSchemaBase


class TrialSchema(PandasSchemaBase):
    """
    This schema describes the core trial structure
    """
    index = fields.Int(required=True)  # maybe this should be required...

    auto_rewarded = fields.Bool()
    change_contrast = fields.Float(allow_none=True)
    change_frame = fields.Integer(strict=True)
    change_image_category = fields.Raw(allow_none=True)
    change_image_name = fields.Raw(allow_none=True)
    change_ori = fields.Float(allow_none=True)
    change_time = fields.Float()
    cumulative_reward_number = fields.Integer(strict=True)
    cumulative_volume = fields.Float()
    delta_ori = fields.Float(allow_none=True)
    endtime = fields.Float()
    endframe = fields.Integer(strict=True)
    initial_contrast = fields.Float(allow_none=True)
    initial_image_category = fields.Raw(allow_none=True)
    initial_image_name = fields.Raw(allow_none=True)
    initial_ori = fields.Float(allow_none=True)
    lick_times = fields.List(fields.Float)
    optogenetics = fields.Bool(allow_none=True)
    publish_time = fields.String()
    response_time = fields.List(fields.Float(allow_none=True))
    reward_frames = fields.List(fields.Integer(strict=True))
    reward_times = fields.List(fields.Float)
    reward_volume = fields.Float()
    rewarded = fields.Bool()
    scheduled_change_time = fields.Float()
    startframe = fields.Integer(strict=True)
    starttime = fields.Float()
    stim_on_frames = fields.List(fields.Integer(strict=True))


class ExtendedTrialSchema(TrialSchema):
    """Extended trial schema
    """
    blank_duration_range = fields.List(fields.Float)
    blank_screen_timeout = fields.Bool()
    color = fields.String()
    computer_name = fields.String()
    distribution_mean = fields.Float()
    LDT_mode = fields.String()
    lick_frames = fields.List(fields.Integer(strict=True))
    mouse_id = fields.String()
    number_of_rewards = fields.Integer(strict=True)
    prechange_minimum = fields.Float()
    response = fields.Float()
    response_window = fields.List(fields.Float)
    reward_licks = fields.Raw()
    reward_lick_count = fields.Integer(strict=True, allow_none=True)
    reward_lick_latency = fields.Float(allow_none=True)
    reward_rate = fields.Float(allow_none=True)
    response_type = fields.String(allow_none=True)
    response_latency = fields.Float(allow_none=True)
    rig_id = fields.String()
    session_duration = fields.Float()
    stage = fields.String()
    stim_duration = fields.Float()
    stimulus = fields.String()
    stimulus_distribution = fields.String()
    task = fields.String()
    trial_duration = fields.Float()
    trial_length = fields.Float()
    trial_type = fields.String()
    user_id = fields.String()

    startdatetime = fields.DateTime()
    date = fields.Date()
    year = fields.Integer(strict=True)
    month = fields.Integer(strict=True)
    day = fields.Integer(strict=True)
    hour = fields.Integer(strict=True)
    dayofweek = fields.Integer(strict=True)
