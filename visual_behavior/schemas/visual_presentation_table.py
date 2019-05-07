from marshmallow import Schema, fields


class VisualPresentationTable(Schema):

    frame = fields.Int(
        description='The stimulus frame when this observation or event occured',
        required=True,
    )

    end_frame = fields.Int(
        description='The last frame of this stimulus, non-inclusive',
        required=True,
    )

    time = fields.Float(
        description=(
            'The time of the experiment (in seconds) when this '
            'observation or event occured'
        ),
        required=True,
    )

    duration = fields.Float(
        description='duration of the movie stimulus display',
        required=True,
    )

    movie_path = fields.String(
        description='The path to the movie stimulus',
        required=True,
        allow_none=True,
    )

    movie_index = fields.Int(
        description='Movie frame index being presented',
        required=True,
        allow_none=True,
    )
