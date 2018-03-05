from marshmallow import Schema, fields


class RewardSchema(Schema):
    frame = fields.Int()
    time = fields.Float()


def dataframe_validator(row,schema=None):
    errors = schema.validate(row)
    return len(errors)==0
