from marshmallow import Schema, fields

# from .base import PandasSchemaBase


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
        required=False,
    )
    lickspout = fields.Int(
        description='The water line on which this reward was dispensed',
        required=False,
        allow_none=True,
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
    dx = fields.Float(
        description='The raw encoder values that the speed is computed from.',
        required=True,
    )
    v_in = fields.Float(
        description='The reference voltage for the encoder.',
        required=True,
        allow_none=True,
        allow_nan=True,
    )
    v_sig = fields.Float(
        description='The input voltage for the encoder.',
        required=True,
        allow_none=True,
        allow_nan=True,
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
    endframe = fields.Int(
        description='frame when this trial ends',
        required=True,
    )
    endtime = fields.Float(
        description='time in seconds when this trial ends',
        required=True,
    )
    trial_length = fields.Float(
        required=True,
    )

    # timing paramters
    change_frame = fields.Float(
        description='The stimulus frame when the change occured on this trial',
        required=True,
        allow_nan=True,
    )
    scheduled_change_time = fields.Float(
        description='The time when the change was scheduled to occur on this trial',
        required=True,
    )
    change_time = fields.Float(
        description='The time when the change occured on this trial',
        required=True,
        allow_nan=True,
    )

    # image parameters
    initial_image_category = fields.String(
        description='The category of the initial images on this trial',
        required=True,
        allow_none=True,
    )
    initial_image_name = fields.String(
        description='The name of the last initial image before the change on this trial',
        required=True,
        allow_none=True,
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
        allow_none=True,
    )
    change_contrast = fields.Float(
        description='The contrast of the change orientation on this trial',
        required=True,
        allow_none=True,
    )
    initial_ori = fields.Float(
        description='The orientation of the initial orientation on this trial',
        required=True,
        allow_none=True,
    )
    change_ori = fields.Float(
        description='The orientation of the change orientation on this trial',
        required=True,
        allow_none=True,
        allow_nan=True,
    )
    delta_ori = fields.Float(
        description='The difference between the initial and change orientations on this trial',
        required=True,
        allow_none=True,
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
        allow_nan=True,
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

    # optogenetics
    optogenetics = fields.Bool(
        description='whether optogenetic stimulation was applied on this trial',
        required=True,
    )


class StimulusSchema(TimeSeriesSchema):
    contrast = fields.Float(
        description='The contrast',
        required=False,
        allow_none=True,
    )
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
    orientation = fields.Float(
        description='The orientation of a grating stimulus',
        required=True,
        allow_none=True,
        allow_nan=True,
    )
    end_frame = fields.Int(
        description='The last frame of this stimulus, non-inclusive',
        required=True,
    )


class MetadataSchema(Schema):
    rig_id = fields.String(
        required=True,
    )
    reward_volume = fields.Float(
        required=False,
    )
    trial_duration = fields.Float(
        required=False,
        allow_none=True,
    )
    startdatetime = fields.String(
        description='Start time of visual behavior session in ISO 8601',
        required=True,
    )
    computer_name = fields.String(
        description='hostname of stimulus computer',
        required=True,
    )
    rewardvol = fields.Float(
        description='volume of rewards for this session',
        required=True,
    )
    reward_vol = fields.Float(
        description='volume of rewards for this session',
        required=False,
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
    )
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
    n_stimulus_frames = fields.Int(
        description='total number of stimulus frames',
        required=True,
    )
    auto_reward_vol = fields.Float(
        description='volume provided during autoreward trials',
        required=True,
    )
    max_session_duration = fields.Float(
        description='maximum duration in minutes of a session',
        required=True,
    )
    min_no_lick_time = fields.Float(
        description='minimum time where there should be no licks before the start of a trial',
        required=True,
    )
    free_reward_trials = fields.Int(
        description='number of free reward trials to start the session',
        required=True,
    )
    abort_on_early_response = fields.Bool(
        description='if True, abort trials on early responses',
        required=True,
    )
    even_sampling_enabled = fields.Bool(
        description='if True, images should be sample evenly from the change matrix',
        required=True,
    )
    failure_repeats = fields.Int(
        description='maximum number of times to repeat parameters after a false alarm',
        required=True,
    )
    initial_blank_duration = fields.Float(
        description='duration of grey screen at start of each trial, in seconds',
        required=True,
    )
    catch_frequency = fields.Float(
        description='fraction of trials that should be catch trials',
        required=True,
    )
    warm_up_trials = fields.Int(
        description='number of warm up trials at start of session',
        required=True,
    )
    stimulus_window = fields.Float(
        description='start and stop times of stimulus window',
        required=True,
    )
    volume_limit = fields.Float(
        description='maximum volume of water to deliver in a session, in mL',
        required=True,
    )
    auto_reward_delay = fields.Float(
        description='delay between change time and reward on autoreward trials',
        required=True,
    )
    periodic_flash = fields.List(
        fields.Float,
        description='duration of flash and grey screen',
        required=True,
        allow_none=True,
    )
    platform_info = fields.Dict(
        description='record of platform information when script ran',
        required=True,
    )
    behavior_session_uuid = fields.UUID(
        required=True,
    )


class ImageSetMetadataSchema(Schema):
    image_set = fields.Str(
        description='name for image set. should be unique (not enforced)',
        # required=True,
    )


class ImageAttributesSchema(Schema):

    image_category = fields.String(
        description='The category of an image stimulus',
        required=True,
    )
    image_name = fields.String(
        description='The name of an image stimulus',
        required=True,
    )
    image_index = fields.Int(
        description='Index of image',
        required=True,
    )


class ImageSetSchema(Schema):
    metadata = fields.Nested(
        ImageSetMetadataSchema,
        description='record of parameters passed into script',
        required=True,
    )
    images = fields.List(
        fields.Raw,
        description='Image data',
        required=True,
    )
    image_attributes = fields.List(
        fields.Nested(ImageAttributesSchema),
        description='Attributes for each of the images in the set',
        required=True,
    )


# class ChangeDetectionSessionCoreSchema(Schema):
#     """ This is the set of core data assets in a change detection session.
#
#     """
#     metadata = fields.Nested(
#         MetadataSchema,
#         description='metadata for the session (dict)',
#         required=True
#     )
#     time = fields.List(
#         fields.Float,
#         description='array of start times for each stimulus frame (list)',
#         required=True,
#     )
#     licks = DataFrameField(description='dataframe of observed licks (pandas.DataFrame)', row_schema=LickSchema, required=True, )  # noqa: F821
#     trials = DataFrameField(row_schema=TrialSchema, required=True, )  # noqa: F821
#     running = DataFrameField(description='dataframe of running speed', row_schema=RunningSchema, required=True, )  # noqa: F821
#     rewards = DataFrameField(description='dataframe of observed licks', row_schema=RewardSchema, required=True, )  # noqa: F821
#     visual_stimuli = DataFrameField(description='dataframe of presented stimuli', row_schema=StimulusSchema, required=True, )  # noqa: F821
