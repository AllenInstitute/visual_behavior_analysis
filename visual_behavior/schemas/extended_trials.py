from marshmallow import fields
from datetime import datetime, date
from .core import TrialSchema


class FriendlyDateTime(fields.DateTime):
    def _deserialize(self, value, attr, data):
        if isinstance(value, datetime):
            return value
        result = super(FriendlyDateTime, self)._deserialize(value, attr, data)
        return result


class FriendlyDate(fields.Date):
    def _deserialize(self, value, attr, data):
        if isinstance(value, date):
            return value
        result = super(FriendlyDate, self)._deserialize(value, attr, data)
        return result


class ExtendedTrialSchema(TrialSchema):
    """Extended trial schema
    """
    blank_duration_range = fields.List(
        fields.Float,
        required=True,
    )
    blank_screen_timeout = fields.Bool(
        required=True,
    )
    color = fields.String(
        required=True,
    )
    computer_name = fields.String(
        required=True,
    )
    distribution_mean = fields.Float(
        required=True,
    )
    LDT_mode = fields.String(
        required=True,
    )
    lick_frames = fields.List(
        fields.Integer(strict=True),
        required=True,
    )
    mouse_id = fields.String(
        required=True,
    )
    number_of_rewards = fields.Integer(
        required=True,
        strict=True,
    )
    prechange_minimum = fields.Float(
        required=True,
    )
    response = fields.Float(
        required=True,
    )
    response_type = fields.String(
        required=True,
    )
    response_window = fields.List(
        fields.Float,
        required=True,
    )
    reward_licks = fields.List(
        fields.Float,
        required=True,
        allow_none=True,
    )
    reward_lick_count = fields.Integer(
        required=True,
        # strict=True,
        allow_none=True,
    )
    reward_lick_latency = fields.Float(
        allow_none=True,
        allow_nan=True,
    )
    reward_rate = fields.Float(
        allow_none=True,
        allow_nan=True,
    )
    rig_id = fields.String(
        required=True,
    )
    session_duration = fields.Float(
        required=True,
    )
    stage = fields.String(
        required=True,
    )
    stim_duration = fields.Float(
        required=True,
    )
    stimulus = fields.String(
        required=True,
    )
    stimulus_distribution = fields.String(
        required=True,
    )
    task = fields.String(
        required=True,
    )
    trial_type = fields.String(
        required=True,
    )
    user_id = fields.String(
        required=True,
    )
    startdatetime = FriendlyDateTime(
        required=True,
        strict=True,
    )
    date = FriendlyDate(
        required=True,
    )
    year = fields.Integer(
        strict=True
    )
    month = fields.Integer(
        required=True,
        strict=True,
    )
    day = fields.Integer(
        required=True,
        strict=True,
    )
    hour = fields.Integer(
        required=True,
        strict=True,
    )
    dayofweek = fields.Integer(
        strict=True,
        required=True,
    )
    behavior_session_uuid = fields.UUID(
        required=True,
    )
