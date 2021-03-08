from __future__ import print_function
from dateutil import parser, tz
from functools import wraps
import logging
import numpy as np
import pandas as pd
from scipy.stats import norm, zscore
from scipy import ndimage
import datetime
import os
import h5py
import cv2
import warnings

from . import database as db

from visual_behavior.ophys.sync.sync_dataset import Dataset


def flatten_list(in_list):
    out_list = []
    for i in range(len(in_list)):
        # check to see if each entry is a list or array.
        if isinstance(in_list[i], list) or isinstance(in_list[i], np.ndarray):
            # if so, iterate over each value and append to out_list
            for entry in in_list[i]:
                out_list.append(entry)
        else:
            # otherwise, append the value itself
            out_list.append(in_list[i])

    return out_list


def get_response_rates(df_in, sliding_window=100, apply_trial_number_limit=False):
    """
    calculates the rolling hit rate, false alarm rate, and dprime value
    Note that the pandas rolling metric deals with NaN values by propogating the previous non-NaN value

    Parameters
    ----------
    sliding_window : int
        Number of trials over which to calculate metrics

    Returns
    -------
    tuple containing hit rate, false alarm rate, d_prime

    """

    from visual_behavior.translator.core.annotate import is_catch, is_hit

    go_responses = df_in.apply(is_hit, axis=1)

    hit_rate = go_responses.rolling(
        window=sliding_window,
        min_periods=0,
    ).mean().values

    catch_responses = df_in.apply(is_catch, axis=1)

    catch_rate = catch_responses.rolling(
        window=sliding_window,
        min_periods=0,
    ).mean().values

    if apply_trial_number_limit:
        # avoid values close to 0 and 1
        go_count = go_responses.rolling(
            window=sliding_window,
            min_periods=0,
        ).count()

        catch_count = catch_responses.rolling(
            window=sliding_window,
            min_periods=0,
        ).count()

        hit_rate = np.vectorize(trial_number_limit)(hit_rate, go_count)
        catch_rate = np.vectorize(trial_number_limit)(catch_rate, catch_count)

    d_prime = dprime(hit_rate, catch_rate)

    return hit_rate, catch_rate, d_prime


class RisingEdge():
    """
    This object implements a "rising edge" detector on a boolean array.

    It takes advantage of how pandas applies functions in order.

    For example, if the "criteria" column in the `df` dataframe consists of booleans indicating
    whether the row meets a criterion, we can detect the first run of three rows above criterion
    with the following

        first_run_of_three = (
            df['criteria']
            .rolling(center=False,window=3)
            .apply(func=RisingEdge().check)
            )

    ```

    """

    def __init__(self):
        self.firstall = False

    def check(self, arr):
        if arr.all():
            self.firstall = True
        return self.firstall


# -> metrics
def trial_number_limit(p, N):
    if N == 0:
        return np.nan
    if not pd.isnull(p):
        p = np.max((p, 1. / (2 * N)))
        p = np.min((p, 1 - 1. / (2 * N)))
    return p


def dprime(hit_rate, fa_rate, limits=(0.01, 0.99)):
    """ calculates the d-prime for a given hit rate and false alarm rate

    https://en.wikipedia.org/wiki/Sensitivity_index

    Parameters
    ----------
    hit_rate : float
        rate of hits in the True class
    fa_rate : float
        rate of false alarms in the False class
    limits : tuple, optional
        limits on extreme values, which distort. default: (0.01,0.99)

    Returns
    -------
    d_prime

    """
    Z = norm.ppf

    # Limit values in order to avoid d' infinity
    hit_rate = np.clip(hit_rate, limits[0], limits[1])
    fa_rate = np.clip(fa_rate, limits[0], limits[1])

    # keep track of nan locations
    hit_rate = pd.Series(hit_rate)
    fa_rate = pd.Series(fa_rate)
    hit_rate_nan_locs = list(hit_rate[pd.isnull(hit_rate)].index)
    fa_rate_nan_locs = list(fa_rate[pd.isnull(fa_rate)].index)

    # fill nans with 0.0 to avoid warning about nans
    d_prime = Z(hit_rate.fillna(0)) - Z(fa_rate.fillna(0))

    # for every location in hit_rate and fa_rate with a nan, fill d_prime with a nan
    for nan_locs in [hit_rate_nan_locs, fa_rate_nan_locs]:
        d_prime[nan_locs] = np.nan

    if len(d_prime) == 1:
        # if the result is a 1-length vector, return as a scalar
        return d_prime[0]
    else:
        return d_prime


def calc_deriv(x, time):
    dx = np.diff(x)
    dt = np.diff(time)
    dxdt_rt = np.hstack((np.nan, dx / dt))
    dxdt_lt = np.hstack((dx / dt, np.nan))

    dxdt = np.vstack((dxdt_rt, dxdt_lt))

    dxdt = np.nanmean(dxdt, axis=0)

    return dxdt


def deg_to_dist(speed_deg_per_s):
    '''
    takes speed in degrees per second
    converts to radians
    multiplies by radius (in cm) to get linear speed in cm/s
    '''
    wheel_diameter = 6.5 * 2.54  # 6.5" wheel diameter
    running_radius = 0.5 * (
        2.0 * wheel_diameter / 3.0)  # assume the animal runs at 2/3 the distance from the wheel center
    running_speed_cm_per_sec = np.pi * speed_deg_per_s * running_radius / 180.
    return running_speed_cm_per_sec


def local_time(iso_timestamp, timezone=None):
    if isinstance(iso_timestamp, datetime.datetime):
        dt = iso_timestamp
    else:
        dt = parser.parse(iso_timestamp)

    if not dt.tzinfo:
        dt = dt.replace(tzinfo=tz.gettz('America/Los_Angeles'))
    return dt.isoformat()


class ListHandler(logging.Handler):
    """docstring for ListHandler."""

    def __init__(self, log_list):
        super(ListHandler, self).__init__()
        self.log_list = log_list

    def emit(self, record):
        entry = self.format(record)
        self.log_list.append(entry)


DoubleColonFormatter = logging.Formatter(
    "%(levelname)s::%(name)s::%(message)s",
)


def inplace(func):
    """ decorator which allows functions that modify a dataframe inplace
    to use a copy instead
    """

    @wraps(func)
    def df_wrapper(df, *args, **kwargs):

        try:
            inplace = kwargs.pop('inplace')
        except KeyError:
            inplace = False

        if inplace is False:
            df = df.copy()

        func(df, *args, **kwargs)

        if inplace is False:
            return df
        else:
            return None

    return df_wrapper


def find_nearest_index(val, time_array):
    '''
    Takes an input (can be a scalar, list, or array) and a time_array
    Returns the index or indices of the time points in time_array that are closest to val
    '''
    if hasattr(val, "__len__"):
        idx = np.zeros(len(val), dtype=int)
        for i, v in enumerate(val):
            idx[i] = np.argmin(np.abs(v - np.array(time_array)))
    else:
        idx = np.argmin(np.abs(val - np.array(time_array)))
    return idx


class Movie(object):
    '''
    a class for loading movies captured with videomon

    Args
    ----------
    filepath (string):
        path to the movie file
    sync_timestamps (array), optional:
        array of timestamps acquired by sync. None by default
    h5_filename (string), optional:
        path to h5 file. assumes by default that filename matches movie filename, but with .h5 extension
    lazy_load (boolean), defaults True:
        when True, each frame is loaded from disk when requested. When False, the entire movie is loaded into memory on intialization (can be very slow)

    Attributes:
    ------------
    frame_count (int):
        number of frames in movie
    width (int):
        width of each frame
    height (int):
        height of each frame

    Todo:
    --------------
    - non-lazy-load a defined interval (would this be useful?)
    '''

    def __init__(self, filepath, sync_timestamps=None, h5_filename=None, lazy_load=True):

        self.cap = cv2.VideoCapture(filepath)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.filepath = filepath

        if sync_timestamps is not None:
            self.sync_timestamps = sync_timestamps
        else:
            self.sync_timestamps = None

        if not h5_filename:
            h5_filename = filepath.replace('.avi', '.h5')
        if os.path.exists(h5_filename):
            timestamp_file = h5py.File(filepath.replace('.avi', '.h5'))

            # videomon saves an h5 file with frame intervals. Take cumulative sum to get timestamps
            self.timestamps_from_file = np.hstack((0, np.cumsum(timestamp_file['frame_intervals'])))
            if self.sync_timestamps is not None and len(self.sync_timestamps) != len(self.timestamps_from_file):
                warnings.warn('NONMATCHING timestamp counts\nThere are {} timestamps in sync and {} timestamps in the associated camera file\nthese should match'.format(
                    len(self.sync_timestamps), len(self.timestamps_from_file)))
        else:
            # warnings.warn('Movies often have a companion h5 file with a corresponding name. None found for this movie. Expected {}'.format(h5_filename))
            self.timestamps_from_file = None

        self._lazy_load = lazy_load
        if self._lazy_load == False:
            self._get_array()
        else:
            self.array = None

    def get_frame(self, frame=None, time=None, timestamps='sync'):
        if time and timestamps == 'sync':
            assert self.sync_timestamps is not None, 'sync timestamps do not exist'
            timestamps = self.sync_timestamps
        elif time and timestamps == 'file':
            assert self.timestamps_from_file is not None, 'timestamps from file do not exist'
            timestamps = self.timestamps_from_file
        else:
            timestamps = None

        if time is not None and frame is None:
            assert timestamps is not None, 'must pass a timestamp array if referencing by time'
            frame = find_nearest_index(time, timestamps)

        # use open CV to get the frame from disk if lazy mode is True
        if self._lazy_load:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            found_frame, frame_array = self.cap.read()
            if found_frame == True:
                return frame_array
            else:
                warnings.warn("Couldn't find frame {}, returning None".format(frame))
                return None
        # or get the frame the preloaded array
        else:
            return self.array[frame, :, :]

    def _get_array(self, dtype='uint8'):
        '''iterate over movie, load frames into an in-memory numpy array one at a time (slow and memory intensive)'''
        self.array = np.empty((self.frame_count, self.height, self.width), np.dtype(dtype))

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for N in range(self.frame_count):
            found_frame, frame = self.cap.read()
            if not found_frame:
                print('something went wrong on frame {}, stopping'.format(frame))
                break
            self.array[N, :, :] = frame[:, :, 0]


def get_sync_data(sync_path):
    sync_data = Dataset(sync_path)

    sample_freq = sync_data.meta_data['ni_daq']['counter_output_freq']
    line_labels = [label for label in sync_data.meta_data['line_labels'] if label != '']
    sd = {}
    for line_label in line_labels:
        sd.update({line_label + '_rising': sync_data.get_rising_edges(line_label) / sample_freq})
        sd.update({line_label + '_falling': sync_data.get_falling_edges(line_label) / sample_freq})

    return sd


class EyeTrackingData(object):
    def __init__(self, ophys_session_id, data_source='filesystem', filter_outliers=True, filter_blinks=True,
                 pupil_color=[0, 255, 247], eye_color=[255, 107, 66], cr_color=[0, 255, 0],
                 filepaths={}, ellipse_fit_path=None):

        # colors of ellipses:
        self.pupil_color = pupil_color
        self.eye_color = eye_color
        self.cr_color = cr_color

        self.ophys_session_id = ophys_session_id
        self.ophys_experiment_id = db.convert_id({'ophys_session_id': ophys_session_id}, 'ophys_experiment_id')
        self.foraging_id = db.get_value_from_table('id', ophys_session_id, 'ophys_sessions', 'foraging_id')

        # get paths of well known files
        well_known_files = db.get_well_known_files(ophys_session_id)
        self.filepaths = {}
        wkf_to_variable_map = {
            "RawEyeTrackingVideo": 'eye_movie',
            "RawBehaviorTrackingVideo": 'behavior_movie',
            "OphysRigSync": 'sync',
            "EyeTracking Ellipses": 'ellipse_fits',
            "EyeDlcOutputFile": 'raw_tracking_points',
        }
        for name in wkf_to_variable_map.keys():
            if name in filepaths:
                self.filepaths[wkf_to_variable_map[name]] = filepaths[name]
            elif wkf_to_variable_map[name] in filepaths:
                self.filepaths[wkf_to_variable_map[name]] = filepaths[wkf_to_variable_map[name]]
            elif name in well_known_files.index:
                self.filepaths[wkf_to_variable_map[name]] = ''.join(well_known_files.loc[name][['storage_directory', 'filename']].tolist())
            else:
                self.filepaths[wkf_to_variable_map[name]] = None

        # allow the ellipse fit path to be overwritten (useful for troubleshooting)
        if ellipse_fit_path is not None:
            self.filepaths['ellipse_fits'] = ellipse_fit_path

        # open and process sync data if sync path exists
        if self.filepaths['sync']:
            self.sync_data = get_sync_data(self.filepaths['sync'])

            # assign timestamps from sync
            self.sync_timestamps = {}
            for movie_label, movie_path in zip(['eye', 'behavior'], [self.filepaths['eye_movie'], self.filepaths['behavior_movie']]):
                movie = Movie(movie_path)
                sync_line = self.get_matching_sync_line(movie, self.sync_data)

                if sync_line is not None:
                    self.sync_timestamps[movie_label] = self.sync_data[sync_line]
                else:
                    self.sync_timestamps[movie_label] = None
                    warnings.warn('no matching sync line for {}'.format(movie_label))

        # get ellipse fits if path exists
        if self.filepaths['ellipse_fits']:
            self.ellipse_fits = {}
            if data_source == 'filesystem':
                # get ellipse fits from h5 files
                for dataset in ['pupil', 'eye', 'cr']:
                    self.ellipse_fits[dataset] = self.get_eye_data_from_file(self.filepaths['ellipse_fits'], dataset=dataset, timestamps=self.sync_timestamps['eye'])
                # replace the 'cr' key with 'corneal_reflection for clarity
                self.ellipse_fits['corneal_reflection'] = self.ellipse_fits.pop('cr')

            elif data_source == 'mongodb':
                mongo_db = db.Database('visual_behavior_data')

                for dataset in ['pupil', 'eye', 'corneal_reflection']:
                    res = list(mongo_db['eyetracking'][dataset].find({'ophys_session_id': self.ophys_session_id}))
                    self.ellipse_fits[dataset] = pd.concat([pd.DataFrame(r['data']) for r in res]).reset_index()

                mongo_db.close()

            if filter_outliers:
                self.filter_outliers()

            if filter_blinks:
                self.filter_blinks()

    def filter_outliers(self, outlier_threshold=3):
        '''
        zscore columns: 'center_x', 'center_y', 'width', 'height'
        flag any rows with a value that exceeds the threshold
        '''
        cols_to_check = ['center_x', 'center_y', 'width', 'height']
        for dataset in self.ellipse_fits.keys():
            df = self.ellipse_fits[dataset]
            df_outliers = pd.DataFrame({parameter: abs(zscore(df[parameter].fillna(df[parameter].mean()))) > outlier_threshold for parameter in cols_to_check})
            df['likely_outlier'] = df_outliers.any(axis=1)

    def get_matching_sync_line(self, movie, sync_data):
        '''determine which sync line matches the frame count of a given movie'''
        nframes = movie.frame_count
        for candidate_line in ['cam1_exposure_rising', 'cam2_exposure_rising', 'behavior_monitoring_rising', 'eye_tracking_rising']:
            if candidate_line in sync_data.keys() and nframes == len(sync_data[candidate_line]):
                return candidate_line
        return None

    def filter_blinks(self, interpolate_over_blinks=True, dilation_frames=2):
        '''
        identify and remove fits near blinks
        '''
        merged = self.ellipse_fits['pupil'].merge(self.ellipse_fits['eye'].rename(columns={'area': 'eye_area'})['eye_area'], left_index=True, right_index=True)
        # detect blinks as frames with either a missing eye fit or a missing pupil fit
        # dilate by 2 frames, which will also label the frames before/after a blink as likely blinks. This should avoid the borderline cases where the eye is just shutting and the fit is bad
        merged['likely_blinks'] = ndimage.binary_dilation(pd.isnull(merged['area']) | pd.isnull(merged['eye_area']) | merged['likely_outlier'] == True, iterations=dilation_frames)
        for fit_data in self.ellipse_fits.keys():
            self.ellipse_fits[fit_data]['likely_blinks'] = merged['likely_blinks']

        fit_parameters = [c for c in self.ellipse_fits['pupil'] if c not in ['frame', 'time', 'likely_blinks']]
        for fit_parameter in fit_parameters:
            # make a new 'blink_corrected' column
            self.ellipse_fits['pupil']['blink_corrected_{}'.format(fit_parameter)] = self.ellipse_fits['pupil'][fit_parameter]

            # fill likely blinks with nans
            self.ellipse_fits['pupil'].loc[self.ellipse_fits['pupil'].query('likely_blinks == True').index, 'blink_corrected_{}'.format(fit_parameter)] = np.nan

            if interpolate_over_blinks:
                # make a 'blink corrected' column that interpolates over the blinks
                data = self.ellipse_fits['pupil']['blink_corrected_{}'.format(fit_parameter)].interpolate()
                zs = pd.Series(zscore(data.fillna(data.mean())))
                data.loc[zs[zs > 5].index] = np.nan
                self.ellipse_fits['pupil']['blink_corrected_{}'.format(fit_parameter)] = data

        # add a column with area normalized relative to 99th percentile area
        area_99_percentile = np.percentile(self.ellipse_fits['pupil']['blink_corrected_area'].fillna(self.ellipse_fits['pupil']['blink_corrected_area'].median()), 99)
        self.ellipse_fits['pupil']['normalized_blink_corrected_area'] = self.ellipse_fits['pupil']['blink_corrected_area'] / area_99_percentile

    def get_eye_data_from_file(self, eye_tracking_path, dataset='pupil', timestamps=None):
        '''open ellipse fit. try to match sync data if possible'''

        df = pd.read_hdf(eye_tracking_path, dataset)

        def area(row):
            # calculate the area as a circle using the max of the height/width as radius
            max_dim = max(row['height'], row['width'])
            return np.pi * max_dim**2

        df['area'] = df[['height', 'width']].apply(area, axis=1)

        if timestamps is not None:
            df['time'] = timestamps

        df['frame'] = np.arange(len(df)).astype(int)

        # imaginary numbers sometimes show up in the ellipse fits. I'm not sure why, but I'm assuming it's an artifact of the fitting process.
        # Convert them to real numbers
        for col in df.columns:
            df[col] = np.real(df[col])

        return df

    def add_ellipse(self, image, ellipse_fit_row, color=[1, 1, 1], linewidth=4):
        '''adds an ellipse fit to an eye tracking video frame'''
        if pd.notnull(ellipse_fit_row['center_x'].item()):
            center_coordinates = (
                int(ellipse_fit_row['center_x'].item()),
                int(ellipse_fit_row['center_y'].item())
            )

            axesLength = (
                int(ellipse_fit_row['width'].item()),
                int(ellipse_fit_row['height'].item())
            )

            angle = ellipse_fit_row['phi']
            startAngle = 0
            endAngle = 360

            # Line thickness of 5 px
            thickness = linewidth

            # Using cv2.ellipse() method
            # Draw a ellipse with red line borders of thickness of 5 px
            image = cv2.ellipse(image, center_coordinates, axesLength,
                                angle, startAngle, endAngle, color, thickness)

        return image

    def get_annotated_frame(self, frame=None, time=None, pupil=True, eye=True, corneal_reflection=False, linewidth=3):
        '''get a particular eye video frame with ellipses drawn'''
        if time is None and frame is None:
            warnings.warn('must specify either a frame or time')
            return None
        elif time is not None and frame is not None:
            warnings.warn('cannot specify both frame and time')
            return None

        eye_movie = Movie(self.filepaths['eye_movie'])
        eye_movie.sync_timestamps = self.sync_timestamps['eye']

        if time is not None:
            frame = np.argmin(np.abs(time - eye_movie.sync_timestamps))

        image = eye_movie.get_frame(frame=frame)

        if pupil:
            image = self.add_ellipse(image, self.ellipse_fits['pupil'].query('frame == @frame'), color=self.pupil_color)

        if eye:
            image = self.add_ellipse(image, self.ellipse_fits['eye'].query('frame == @frame'), color=self.eye_color)

        if corneal_reflection:
            image = self.add_ellipse(image, self.ellipse_fits['corneal_reflection'].query('frame == @frame'), color=self.cr_color)

        return image


def convert_to_fraction(df_in, baseline_conditional=None):
    '''
    converts all columns of an input dataframe, excluding the columns labeled 't' or 'time' to a fractional change

    baseline conditional can be a string used to subselect rows for baseline calculation
        (e.g. 'time <= 0')
    '''
    df = df_in.copy()
    cols = [col for col in df.columns if col not in ['t', 'time', 'timestamps']]
    for col in cols:
        s = df[col]
        if baseline_conditional is None:
            # use the entire timeseries as baseline
            s0 = df[col].mean(axis=0)
        else:
            s0 = df.query(baseline_conditional)[col].mean(axis=0)
        df[col] = (s - s0) / s0
    return df


def event_triggered_response(df, parameter, event_times, time_key=None, t_before=10, t_after=10, sampling_rate=60, output_format='tidy'):
    '''
    build event triggered response around a given set of events
    required inputs:
      df: dataframe of input data
      parameter: column of input dataframe to extract around events
      event_times: times of events of interest
    optional inputs:
      time_key: key to use for time (if None (default), will search for either 't' or 'time'. if 'index', use indices)
      t_before: time before each of event of interest
      t_after: time after each event of interest
      sampling_rate: desired sampling rate of output (input data will be interpolated)
      output_format: 'wide' or 'tidy' (default = 'tidy')
    output:
      if output_format == 'wide':
        dataframe with one time column ('t') and one column of data for each event
      if output_format == 'tidy':
        dataframe with columns representing:
            time
            output value
            event number
            event time

    An example use case, recover a sinousoid from noise:
        (also see https://gist.github.com/dougollerenshaw/628c63375cc68f869a28933bd5e2cbe5)
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        # generate some sample data¶
        # a sinousoid corrupted by noise
        # make a time vector
        # go from -10 to 110 so we have a 10 second overage if we analyze from 0 to 100 seconds
        t = np.arange(-10,110,0.001)

        # now build a dataframe
        df = pd.DataFrame({
            'time': t,
            'noisy_sinusoid': np.sin(2*np.pi*t) + np.random.randn(len(t))*3
        })

        # use the event_triggered_response function to get a tidy dataframe of the signal around every event
        # events will simply be generated as every 1 second interval starting at 0.5, since our period here is 1
        etr = event_triggered_response(
            df,
            parameter = 'noisy_sinusoid',
            event_times = [C+0.5 for C in range(0,99)],
            t_before = 1,
            t_after = 1,
            sampling_rate = 100
        )

        # use seaborn to view the result
        # We're able to recover the sinusoid through averaging
        fig, ax = plt.subplots()
        sns.lineplot(
            data = etr,
            x='time',
            y='noisy_sinusoid',
            ax=ax
        )

    '''
    if time_key is None:
        if 't' in df.columns:
            time_key = 't'
        elif 'timestamps' in df.columns:
            time_key = 'timestamps'
        else:
            time_key = 'time'

    _d = {'time': np.arange(-t_before, t_after, 1 / sampling_rate)}
    for ii, event_time in enumerate(np.array(event_times)):

        if time_key == 'index':
            df_local = df.loc[(event_time - t_before):(event_time + t_after)]
            t = df_local.index.values - event_time
        else:
            df_local = df.query(
                "{0} > (@event_time - @t_before) and {0} < (@event_time + @t_after)".format(time_key))
            t = df_local[time_key] - event_time
        y = df_local[parameter]

        _d.update({'event_{}_t={}'.format(ii, event_time): np.interp(_d['time'], t, y)})
    if output_format == 'wide':
        return pd.DataFrame(_d)
    elif output_format == 'tidy':
        df = pd.DataFrame(_d)
        melted = df.melt(id_vars='time')
        melted['event_number'] = melted['variable'].map(lambda s: s.split('event_')[1].split('_')[0])
        melted['event_time'] = melted['variable'].map(lambda s: s.split('t=')[1])
        return melted.drop(columns=['variable']).rename(columns={'value': parameter})


def annotate_licks(dataset, inplace=False, lick_bout_ili=2):
    '''
    annotates the licks dataframe with some additional columns

    arguments:
        dataset (BehaviorSession or BehaviorOphysSession object): an SDK session object
        inplace (boolean): If True, operates in place (default = False)
        lick_bout_ili (float): interval between licks required to label a lick as the start/end of a licking bout

    returns (only if inplace=False):
        pandas.DataFrame with columns:
            timestamps (float): timestamp of every lick
            frame (int): frame of every lick
            pre_ili (float): time without any licks before current lick
            post_ili (float): time without any licks after current lick
            bout_start (boolean): True if licks is first in bout, False otherwise
            bout_end (boolean): True if licks is last in bout, False otherwise
            licks_in_bout (int): Number of licks in current lick bout
    '''
    if inplace:
        licks_df = dataset.licks
    else:
        licks_df = dataset.licks.copy()

    licks_df['pre_ili'] = licks_df['timestamps'] - licks_df['timestamps'].shift(fill_value=0)
    licks_df['post_ili'] = licks_df['timestamps'].shift(periods=-1, fill_value=np.inf) - licks_df['timestamps']
    licks_df['bout_start'] = licks_df['pre_ili'] > lick_bout_ili
    licks_df['bout_end'] = licks_df['post_ili'] > lick_bout_ili

    # count licks in every bout, add a lick bout number
    licks_df['licks_in_bout'] = np.nan
    licks_df['lick_bout_number'] = np.nan
    licks_df['bout_rewarded'] = np.nan
    lick_bout_number = 0
    for bout_start_index, row in licks_df.query('bout_start').iterrows():
        bout_end_index = licks_df.iloc[bout_start_index:].query('bout_end').index[0]
        licks_df.at[bout_start_index, 'licks_in_bout'] = bout_end_index - bout_start_index + 1

        licks_df.at[bout_start_index, 'lick_bout_number'] = lick_bout_number
        lick_bout_number += 1

        bout_start_time = licks_df.loc[bout_start_index]['timestamps']
        bout_end_time = licks_df.loc[bout_end_index]['timestamps']
        licks_df.at[bout_start_index, 'bout_rewarded'] = float(len(dataset.rewards.query('timestamps >= @bout_start_time and timestamps <= @bout_end_time')) >= 1)

    licks_df['licks_in_bout'] = licks_df['licks_in_bout'].fillna(method='ffill').astype(int)
    licks_df['lick_bout_number'] = licks_df['lick_bout_number'].fillna(method='ffill').astype(int)
    licks_df['bout_rewarded'] = licks_df['bout_rewarded'].fillna(method='ffill').astype(bool)

    if inplace == False:
        return licks_df
