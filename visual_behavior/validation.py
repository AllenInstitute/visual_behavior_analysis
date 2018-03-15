from marshmallow import Schema, fields


class TimeSeriesSchema(Schema):
    frame = fields.Int(required=True)
    time = fields.Float(required=True)

    
class RewardSchema(TimeSeriesSchema):
    pass


class LickSchema(TimeSeriesSchema):
    pass


class RunningSchema(TimeSeriesSchema):
    speed = fields.Float(required=True)


class ChangeDetectionTrialSchema(Schema):
    auto_rewarded = fields.Bool(required=True, allow_none=True)
    change_contrast = fields.Int(required=True)
    change_frame = fields.Float(required=True)
    change_image_category = fields.String(required=True, allow_none=True)
    change_image_name = fields.String(required=True, allow_none=True)
    change_ori = fields.Float(required=True)
    change_time = fields.Float(required=True)
    cumulative_reward_number = fields.Int(required=True)
    cumulative_volume = fields.Float(required=True)
    delta_ori = fields.Float(required=True)
    index = fields.Int(required=True)
    initial_contrast = fields.Float(required=True)
    initial_image_category = fields.String(required=True)
    initial_image_name = fields.String(required=True)
    initial_ori = fields.Float(required=True)
    lick_times = fields.List(fields.Float, required=True)
    optogenetics = fields.Bool(required=True)
    publish_time = fields.Str(required=True)
    response_latency = fields.Float(required=True)
    response_time = fields.List(fields.Float, required=True)
    response_type = fields.List(fields.Bool, required=True)
    reward_frames = fields.List(fields.Int, required=True)
    reward_times = fields.List(fields.Float, required=True)
    reward_volume = fields.Float(required=True)
    rewarded = fields.Bool(required=True)
    scheduled_change_time = fields.Float(required=True)
    startframe = fields.Int(required=True)
    starttime = fields.Float(required=True)
    stim_on_frames = fields.List(fields.Bool, required=True)


def dataframe_validator(row,schema=None):
    errors = schema.validate(row)
    return len(errors)==0
