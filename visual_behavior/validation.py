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
    acceleration = fields.Float(required=True)
    jerk = fields.Float(required=True)


def dataframe_validator(row,schema=None):
    errors = schema.validate(row)
    return len(errors)==0
