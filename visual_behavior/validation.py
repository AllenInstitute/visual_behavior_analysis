from marshmallow import Schema, fields


class ChangeDetectionSessionCoreSchema(Schema):
    """ This is the set of core data assets in a change detection session.

    """
    metadata = fields.Nested(
        MetadataSchema,
        description='metadata for the session (dict)',
        required=True
    )
    time = fields.List(
        fields.Float,
        description='array of start times for each stimulus frame (list)',
        required=True,
    )
    licks = DataFrameField(
        description='dataframe of observed licks (pandas.DataFrame)',
        row_schema=LickSchema,
        required=True,
    )
    trials = DataFrameField(
        row_schema=TrialSchema,
        required=True,
    )
    running = DataFrameField(
        description='dataframe of running speed'
        row_schema=RunningSchema,
        required=True,
    )
    rewards = DataFrameField(
        description='dataframe of observed licks'
        row_schema=RewardSchema,
        required=True,
    )
    visual_stimuli = DataFrameField(
        description='dataframe of presented stimuli'
        row_schema=StimulusSchema,
        required=True,
    )

class TimeSeriesSchema(Schema):
    """ base schema for all timeseries
    """
    frame = fields.Int(
        description='The stimulus frame when this observation or event occured',
        required=True,
    )
    time = fields.Float(
        description='The time of the experiment (in seconds) when this observation or event occured',
        required=True,
    )


class RewardSchema(TimeSeriesSchema):
    """ schema for water reward presentations

    """
    volume = fields.Float(
        description='Volume of water dispensed on this reward presentation in mL',
        required=True,
    )
    lickspout = fields.Int(
        description='The water line on which this reward was dispensed',
        required=True,
    )


class LickSchema(TimeSeriesSchema):
    """ schema for observed licks
    """
    pass


class RunningSchema(TimeSeriesSchema):
    speed = fields.Float(
        description='The speed of the mouse on the running wheel (in cm/s)',
        required=True,
    )


class TrialSchema(Schema):
    """
    This schema describes the core trial structure
    """

    index = fields.Int(
        description='Trial number in this session',
        required=True,
    )
    startframe = fields.Int(
        description='frame when this trial starts',
        required=True,
    )
    starttime = fields.Float(
        description='time in seconds when this trial starts',
        required=True,
    )


    # timing paramters
    change_frame = fields.Int(
        description='The stimulus frame when the change occured on this trial',
        required=True,
    )
    scheduled_change_time = fields.Float(
        description='The time when the change was scheduled to occur on this trial',
        required=True,
    )
    change_time = fields.Float(
        description='The time when the change occured on this trial',
        required=True,
    )

    # image parameters
    initial_image_category = fields.String(
        description='The category of the initial images on this trial',
        required=True,
    )
    initial_image_name = fields.String(
        description='The name of the last initial image before the change on this trial',
        required=True,
    )
    change_image_category = fields.String(
        description='The category of the change images on this trial',
        required=True,
        allow_none=True,
    )
    change_image_name = fields.String(
        description='The name of the first change image on this trial',
        required=True,
        allow_none=True,
    )

    # oriented gratings paramters
    initial_contrast = fields.Float(
        description='The contrast of the initial orientation on this trial',
        required=True,
    )
    change_contrast = fields.Float(
        description='The contrast of the change orientation on this trial',
        required=True,
    )
    initial_ori = fields.Float(
        description='The orientation of the initial orientation on this trial',
        required=True,
    )
    change_ori = fields.Float(
        description='The orientation of the change orientation on this trial',
        required=True,
    )
    delta_ori = fields.Float(
        description='The difference between the initial and change orientations on this trial',
        required=True,
    )

    # general stimulus info
    stim_on_frames = fields.List(
        fields.Bool,
        description='frames in this trial in which the stimulus was present',
        required=True,
    )

    # licks
    lick_times = fields.List(
        fields.Float,
        description='times of licks on this trial',
        required=True,
    )
    response_latency = fields.Float(
        description='The latency between the change and the first lick on this trial',
        required=True,
    )
    response_time = fields.List(
        fields.Float,
        description='need to check this with Doug',
        required=True,
    )
    response_type = fields.List(
        fields.Bool,
        description='need to check this with Doug',
        required=True,
    )

    # rewards
    reward_frames = fields.List(
        fields.Int,
        required=True,
    )
    reward_times = fields.List(
        fields.Float,
        required=True,
    )
    reward_volume = fields.Float(
        required=True,
    )
    rewarded = fields.Bool(
        required=True,
    )

    auto_rewarded = fields.Bool(
        description='whether this trial was an auto_rewarded trial',
        required=True,
        allow_none=True,
    )
    cumulative_reward_number = fields.Int(
        description='the cumulative number of rewards in the session at trial end',
        required=True,
    )
    cumulative_volume = fields.Float(
        description='the total volume of rewards in the session at trial end',
        required=True,
    )

    # candidates to deprecate
    optogenetics = fields.Bool(
        description='whether optogenetic stimulation was applied on this trial',
        required=True,
        )
    publish_time = fields.Str(
        description='the time that this trial was published',
        required=True,
        )


class StimulusSchema(TimeSeriesSchema):
    duration = fields.Float(
        description='duration of the stimulus',
        required=True,
    )
    image_category = fields.String(
        description='The category of an image stimulus',
        required=True,
        allow_none=True,
    )
    image_name = fields.String(
        description='The name of an image stimulus',
        required=True,
        allow_none=True,
    )
    contrast = fields.Float(
        description='The contrast of a grating stimulus',
        required=True,
        allow_none=True,
    )
    orientation = fields.Float(
        description='The orientation of a grating stimulus',
        required=True,
    )

class ExtendedTrialSchema(TrialSchema):
    pass


class MetadataSchema(Schema):
    startdatetime = fields.String(
        description='Start time of visual behavior session in ISO 8601',
        required=True,
    )
    rig_id = fields.String(
        description='short name of rig',
    )
    computer_name = fields.String(
        description='hostname of stimulus computer',
        required=True,
    )
    rewardvol = fields.Float(
        description='volume of rewards for this session',
        required=True,
    )
    params = fields.Dict(
        description='record of parameters passed into script',
        required=True,
    )
    mouseid = fields.String(
        description='ID of mouse',
        required=True,
    )
    response_window = fields.List(
        fields.Float,
        description='beginning and end of response window',
        required=True,
    )
    task = fields.String(
        description='name of task',
        required=True,
    )
    stage = fields.String(
        description='name of training stage',
        required=True,
    ),
    stoptime = fields.Float(
        description='time when experiment ended (in seconds)',
        required=True,
    )
    userid = fields.String(
        description='active directory username of trainer',
        required=True,
    )
    lick_detect_training_mode = fields.String(
        description='mode for lick detection training',
        required=True,
    )
    blankscreen_on_timeout = fields.Boolean(
        description='whether the screen should go blank during a timeout',
        required=True,
    )
    stim_duration = fields.Float(
        description='duration of stimulus (in ms)',
        required=True,
    )
    blank_duration_range = fields.List(
        fields.Float,
        description='beginning and end of response window (in ms)',
        required=True,
    )
    delta_minimum = fields.Float(
        description='minimum time until change',
        required=True,
    )
    stimulus_distribution = fields.String(
        description='distribution from which to sample change times ("uniform" or "exponential")',
        required=True,
    )
    stimulus = fields.String(
        description='stimulus type',
        required=True,
    )
    delta_mean = fields.Float(
        description='mean time until change',
        required=True,
    )
    trial_duration = fields.Float(
        description='duration of a single trial (unless aborted or extended by licks)',
        required=True,
    )
    n_stimulus_frames = fields.Int(
        description='total number of stimulus frames',
        required=True,
    )


def dataframe_validator(row, schema):
    errors = schema.validate(row)
    return (len(errors) == 0)
