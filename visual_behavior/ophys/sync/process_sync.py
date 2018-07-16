import numpy as np


def filter_digital(rising, falling, threshold=0.0001):
    """
    Removes short transients from digital signal.

    Rising and falling should be same length and units
        in seconds.

    Kwargs:
        threshold (float): transient width
    """
    # forwards (removes low-to-high transients)
    dif_f = falling - rising
    falling_f = falling[np.abs(dif_f) > threshold]
    rising_f = rising[np.abs(dif_f) > threshold]
    # backwards (removes high-to-low transients )
    dif_b = rising_f[1:] - falling_f[:-1]
    dif_br = np.append([threshold * 2], dif_b)
    dif_bf = np.append(dif_b, [threshold * 2])
    rising_f = rising_f[np.abs(dif_br) > threshold]
    falling_f = falling_f[np.abs(dif_bf) > threshold]

    return rising_f, falling_f


def calculate_delay(sync_data, stim_vsync_fall, sample_frequency):
    # from http://stash.corp.alleninstitute.org/projects/INF/repos/lims2_modules/browse/CAM/ophys_time_sync/ophys_time_sync.py
    ASSUMED_DELAY = 0.0351
    DELAY_THRESHOLD = 0.001
    FIRST_ELEMENT_INDEX = 0
    ROUND_PRECISION = 4
    ONE = 1

    print 'calculating monitor delay' # flake8: noqa: E999

    try:
        # photodiode transitions
        photodiode_rise = sync_data.get_rising_edges('stim_photodiode') / sample_frequency

        # Find start and stop of stimulus
        # test and correct for photodiode transition errors
        photodiode_rise_diff = np.ediff1d(photodiode_rise)
        min_short_photodiode_rise = 0.1
        max_short_photodiode_rise = 0.3
        min_medium_photodiode_rise = 0.5
        max_medium_photodiode_rise = 1.5

        # find the short and medium length photodiode rises
        short_rise_indexes = np.where(np.logical_and(photodiode_rise_diff > min_short_photodiode_rise,
                                                     photodiode_rise_diff < max_short_photodiode_rise))[
            FIRST_ELEMENT_INDEX]
        medium_rise_indexes = np.where(np.logical_and(photodiode_rise_diff > min_medium_photodiode_rise,
                                                      photodiode_rise_diff < max_medium_photodiode_rise))[
            FIRST_ELEMENT_INDEX]

        short_set = set(short_rise_indexes)

        # iterate through the medium photodiode rise indexes to find the start and stop indexes
        # lookng for three rise pattern
        next_frame = ONE
        start_pattern_index = 2
        end_pattern_index = 3
        ptd_start = None
        ptd_end = None

        for medium_rise_index in medium_rise_indexes:
            if set(range(medium_rise_index - start_pattern_index, medium_rise_index)) <= short_set:
                ptd_start = medium_rise_index + next_frame
            elif set(range(medium_rise_index + next_frame, medium_rise_index + end_pattern_index)) <= short_set:
                ptd_end = medium_rise_index

        # if the photodiode signal exists
        if ptd_start is not None and ptd_end is not None:
            # check to make sure there are no there are no photodiode errors
            # sometimes two consecutive photodiode events take place close to each other
            # correct this case if it happens
            photodiode_rise_error_threshold = 1.8
            last_frame_index = -1

            # iterate until all of the errors have been corrected
            while any(photodiode_rise_diff[ptd_start:ptd_end] < photodiode_rise_error_threshold):
                error_frames = np.where(photodiode_rise_diff[ptd_start:ptd_end] < photodiode_rise_error_threshold)[
                    FIRST_ELEMENT_INDEX] + ptd_start
                # remove the bad photodiode event
                photodiode_rise = np.delete(photodiode_rise, error_frames[last_frame_index])
                ptd_end -= 1
                photodiode_rise_diff = np.ediff1d(photodiode_rise)

            # Find the delay
            # calculate monitor delay
            first_pulse = ptd_start
            number_of_photodiode_rises = ptd_end - ptd_start
            half_vsync_fall_events_per_photodiode_rise = 60
            vsync_fall_events_per_photodiode_rise = half_vsync_fall_events_per_photodiode_rise * 2

            delay_rise = np.empty(number_of_photodiode_rises)
            for photodiode_rise_index in range(number_of_photodiode_rises):
                delay_rise[photodiode_rise_index] = photodiode_rise[photodiode_rise_index + first_pulse] - \
                    stim_vsync_fall[(photodiode_rise_index * vsync_fall_events_per_photodiode_rise) + half_vsync_fall_events_per_photodiode_rise]

            # get a single delay value by finding the mean of all of the delays - skip the last element in the array (the end of the experimenet)
            delay = np.mean(delay_rise[:last_frame_index])

            if (delay > DELAY_THRESHOLD or np.isnan(delay)):
                print "Sync photodiode error needs to be fixed"
                delay = ASSUMED_DELAY
                print "Using assumed monitor delay:", round(delay, ROUND_PRECISION)

        # assume delay
        else:
            delay = ASSUMED_DELAY
    except Exception as e:
        print e
        print "Process without photodiode signal"
        delay = ASSUMED_DELAY
        print "Assumed delay:", round(delay, ROUND_PRECISION)

    return delay


def get_sync_data(lims_id):
    print 'getting sync data'
    from visual_behavior.ophys.sync.sync_dataset import Dataset
    import visual_behavior.ophys.io.convert_level_1_to_level_2 as io
    sync_path = io.get_sync_path(lims_id)
    sync_dataset = Dataset(sync_path)
    meta_data = sync_dataset.meta_data
    sample_freq = meta_data['ni_daq']['counter_output_freq']
    # 2P vsyncs
    vs2p_r = sync_dataset.get_rising_edges('2p_vsync')
    vs2p_f = sync_dataset.get_falling_edges(
        '2p_vsync', )  # new sync may be able to do units = 'sec', so conversion can be skipped
    vs2p_rsec = vs2p_r / sample_freq
    vs2p_fsec = vs2p_f / sample_freq
    vs2p_r_filtered, vs2p_f_filtered = filter_digital(vs2p_rsec, vs2p_fsec, threshold=0.01)
    frames_2p = vs2p_f_filtered  # used to be using rising edges, CAM scripts use falling
    # Convert to seconds - skip if using units in get_falling_edges, otherwise convert before doing filter digital
    # vs2p_rsec = vs2p_r / sample_freq
    # frames_2p = vs2p_rsec
    # stimulus vsyncs
    # vs_r = d.get_rising_edges('stim_vsync')
    vs_f = sync_dataset.get_falling_edges('stim_vsync')
    # convert to seconds
    # vs_r_sec = vs_r / sample_freq
    vs_f_sec = vs_f / sample_freq
    # vsyncs = vs_f_sec
    # add display lag
    monitor_delay = calculate_delay(sync_dataset, vs_f_sec, sample_freq)
    vsyncs = vs_f_sec + monitor_delay  # this should be added, right!?
    # add lick data
    lick_1 = sync_dataset.get_rising_edges('lick_1') / sample_freq
    trigger = sync_dataset.get_rising_edges('2p_trigger') / sample_freq
    cam1_exposure = sync_dataset.get_rising_edges('cam1_exposure') / sample_freq
    cam2_exposure = sync_dataset.get_rising_edges('cam2_exposure') / sample_freq
    stim_photodiode = sync_dataset.get_rising_edges('stim_photodiode') / sample_freq
    # some experiments have 2P frames prior to stimulus start - restrict to timestamps after trigger
    frames_2p = frames_2p[frames_2p > trigger[0]]
    print 'stimulus frames detected in sync: ', len(vsyncs)
    print 'ophys frames detected in sync: ', len(frames_2p)
    # put sync data in dphys format to be compatible with downstream analysis
    times_2p = {'timestamps': frames_2p}
    times_vsync = {'timestamps': vsyncs}
    times_lick_1 = {'timestamps': lick_1}
    times_trigger = {'timestamps': trigger}
    times_cam1_exposure = {'timestamps': cam1_exposure}
    times_cam2_exposure = {'timestamps': cam2_exposure}
    times_stim_photodiode = {'timestamps': stim_photodiode}
    sync_data = {'ophys_frames': times_2p,
                 'stimulus_frames': times_vsync,
                 'lick_times': times_lick_1,
                 'cam1_exposure': times_cam1_exposure,
                 'cam2_exposure': times_cam2_exposure,
                 'stim_photodiode': times_stim_photodiode,
                 'ophys_trigger': times_trigger,
                 }
    return sync_data
