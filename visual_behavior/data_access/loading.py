import warnings
from allensdk.internal.api import PostgresQueryMixin
from allensdk.brain_observatory.behavior.behavior_session import BehaviorSession
from allensdk.brain_observatory.behavior.behavior_ophys_experiment import BehaviorOphysExperiment
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache as bpc
from visual_behavior.data_access import filtering
from visual_behavior.data_access import reformat
from visual_behavior.data_access import utilities
from visual_behavior.data_access import from_lims
import visual_behavior.database as db

import os
import glob
import h5py  # for loading motion corrected movie
import numpy as np
import pandas as pd
import configparser as configp  # for parsing scientifica ini files
config = configp.ConfigParser()


try:
    lims_dbname = os.environ["LIMS_DBNAME"]
    lims_user = os.environ["LIMS_USER"]
    lims_host = os.environ["LIMS_HOST"]
    lims_password = os.environ["LIMS_PASSWORD"]
    lims_port = os.environ["LIMS_PORT"]

    mtrain_dbname = os.environ["MTRAIN_DBNAME"]
    mtrain_user = os.environ["MTRAIN_USER"]
    mtrain_host = os.environ["MTRAIN_HOST"]
    mtrain_password = os.environ["MTRAIN_PASSWORD"]
    mtrain_port = os.environ["MTRAIN_PORT"]

    lims_engine = PostgresQueryMixin(
        dbname=lims_dbname,
        user=lims_user,
        host=lims_host,
        password=lims_password,
        port=lims_port
    )

    mtrain_engine = PostgresQueryMixin(
        dbname=mtrain_dbname,
        user=mtrain_user,
        host=mtrain_host,
        password=mtrain_password,
        port=mtrain_port
    )

except Exception as e:
    warn_string = 'failed to set up LIMS/mtrain credentials\n{}\n\ninternal AIBS users should set up environment variables appropriately\nfunctions requiring database access will fail'.format(
        e)
    warnings.warn(warn_string)


# function inputs
# ophys_experiment_id
# ophys_session_id
# behavior_session_id
# ophys_container_id


#  RELEVANT DIRECTORIES


def get_platform_analysis_cache_dir():
    """
    This is the cache directory to use for all platform paper analysis
    This cache contains NWB files downloaded directly from AWS
    """
    return r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/platform_paper_cache'


def get_production_cache_dir():
    """Get directory containing a manifest file that includes all VB production data, including failed experiments"""
    cache_dir = r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/2020_cache/production_cache'
    return cache_dir


def get_qc_plots_dir():
    return r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots'


def get_super_container_plots_dir():
    return r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/super_container_plots'


def get_container_plots_dir():
    return r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/container_plots'


def get_session_plots_dir():
    return r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/session_plots'


def get_experiment_plots_dir():
    return r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/experiment_plots'


def get_single_cell_plots_dir():
    return r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/single_cell_plots'


def get_analysis_cache_dir():
    return r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_production_analysis'


def get_events_dir():
    return r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/event_detection'


def get_behavior_model_outputs_dir():
    return r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/behavior_model_output'


def get_decoding_analysis_dir():
    return r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/decoding'


def get_ophys_glm_dir():
    return r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/ophys_glm'


def get_stimulus_response_df_dir(interpolate=True, output_sampling_rate=30):
    base_dir = r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/platform_paper_cache/stimulus_response_dfs'
    if interpolate:
        save_dir = os.path.join(base_dir, 'interpolate_' + str(output_sampling_rate) + 'Hz')
    else:
        save_dir = os.path.join(base_dir, 'original_frame_rate')
    return save_dir


def get_multi_session_df_df_dir(interpolate=True, output_sampling_rate=30):
    base_dir = get_platform_analysis_cache_dir()
    if interpolate:
        save_dir = os.path.join(base_dir, 'multi_session_mean_response_dfs', 'interpolate_' + str(output_sampling_rate) + 'Hz')
    else:
        save_dir = os.path.join(base_dir, 'multi_session_mean_response_dfs', 'original_frame_rate')
    return save_dir


def get_manifest_path():
    """
    Get path to default manifest file for analysis
    Default location of manifest is the production cache directory at /visual_behavior/2020_cache/production_cache'
    This includes all VB production data and is not the same thing as the platform paper cache
    """
    manifest_path = os.path.join(get_production_cache_dir(), "manifest.json")
    return manifest_path


def get_visual_behavior_cache(from_s3=True, release_data_only=True, cache_dir=None):
    """
    Gets the visual behavior dataset cache object from s3 or lims
    :param from_s3: If True, loads manifest from s3 and saves to provided cache_dir (or default cache_dir if None provided)
    :param release_data_only: limits to data released on March 25th and August 12th when loading from lims
    :param cache_dir: directory where to save manifest & data files if using s3
    :return: SDK cache object
    """
    if from_s3:
        if cache_dir is None:
            cache_dir = get_platform_analysis_cache_dir()
            print(cache_dir)
        cache = bpc.from_s3_cache(cache_dir=cache_dir)
    else:
        if release_data_only:
            cache = bpc.from_lims(data_release_date=['2021-03-25', '2021-08-12'])
        else:
            cache = bpc.from_lims()
    return cache


def get_released_ophys_experiment_table(exclude_ai94=True):
    '''
    gets the released ophys experiment table from AWS

    Keyword Arguments:
        exclude_ai94 {bool} -- If True, exclude data from mice with Ai94(GCaMP6s) as the reporter line. (default: {True})

    Returns:
        experiment_table -- returns a dataframe with ophys_experiment_id as the index and metadata as columns.
    '''
    print('getting experiment table from lims, NOT AWS')

    cache = bpc.from_lims(data_release_date=['2021-03-25', '2021-08-12'])

    experiment_table = cache.get_ophys_experiment_table()

    if exclude_ai94:
        experiment_table = experiment_table.query('reporter_line != "Ai94(TITL-GCaMP6s)"')

    return experiment_table


def get_platform_paper_experiment_table(add_extra_columns=True):
    """
    loads the experiment table that was downloaded from AWS and saved to the the platform paper cache dir.
    Then filter out VisualBehaviorMultiscope4areasx2d and Ai94 data.
    And add cell_type column (values = ['Excitatory', 'Sst Inhibitory', 'Vip Inhibitory']
    Set add_extra_columns to False if you dont need things like 'cell_type', 'binned_depth', or 'add_last_familiar'
    """
    cache_dir = get_platform_analysis_cache_dir()
    cache = bpc.from_s3_cache(cache_dir=cache_dir)
    experiment_table = cache.get_ophys_experiment_table()

    # remove 4x2 and Ai94 data
    experiment_table = experiment_table[(experiment_table.project_code != 'VisualBehaviorMultiscope4areasx2d') &
                                        (experiment_table.reporter_line != 'Ai94(TITL-GCaMP6s)')].copy()

    # overwrite session number and passive columns to patch for bug flagged in this SDK issue:
    # https://github.com/AllenInstitute/AllenSDK/issues/2251
    experiment_table = utilities.add_session_number_to_experiment_table(experiment_table)
    experiment_table = utilities.add_passive_flag_to_ophys_experiment_table(experiment_table)

    if add_extra_columns == True:
        # add cell type and binned depth columms for plot labels
        experiment_table = utilities.add_cell_type_column(experiment_table)
        experiment_table = utilities.add_average_depth_across_container(experiment_table)
        experiment_table = utilities.add_binned_depth_column(experiment_table)
        experiment_table = utilities.add_area_depth_column(experiment_table)
        # add other columns indicating whether a session was the last familiar before the first novel session,
        # or the second passing novel session after the first truly novel one
        experiment_table = utilities.add_date_string(experiment_table)  # add simplified date string for sorting
        experiment_table = utilities.add_first_novel_column(experiment_table)
        experiment_table = utilities.add_n_relative_to_first_novel_column(experiment_table)
        experiment_table = utilities.add_last_familiar_column(experiment_table)
        experiment_table = utilities.add_last_familiar_active_column(experiment_table)
        experiment_table = utilities.add_second_novel_column(experiment_table)
        experiment_table = utilities.add_second_novel_active_column(experiment_table)
        # add column that has a combination of experience level and exposure to omissions for familiar sessions,
        # or exposure to image set for novel sessions
        experiment_table = utilities.add_experience_exposure_column(experiment_table)

    return experiment_table


def get_platform_paper_behavior_session_table():
    """
    loads the behavior sessions table that was downloaded from AWS and saved to the the platform paper cache dir.
    Then filter out VisualBehaviorMultiscope4areasx2d and Ai94 data.
    And add cell_type column (values = ['Excitatory', 'Sst Inhibitory', 'Vip Inhibitory']
    """
    cache_dir = get_platform_analysis_cache_dir()
    cache = bpc.from_s3_cache(cache_dir=cache_dir)
    behavior_sessions = cache.get_behavior_session_table()

    # get rid of NaNs, documented in SDK#2218
    behavior_sessions = behavior_sessions[behavior_sessions.session_type.isnull() == False]

    # remove 4x2 and Ai94 data
    behavior_sessions = behavior_sessions[(behavior_sessions.project_code != 'VisualBehaviorMultiscope4areasx2d') &
                                          (behavior_sessions.reporter_line != 'Ai94(TITL-GCaMP6s)')].copy()

    # overwrite session number and passive columns to patch for bug flagged in this SDK issue:
    # https://github.com/AllenInstitute/AllenSDK/issues/2251
    behavior_sessions = utilities.add_session_number_to_experiment_table(behavior_sessions)
    behavior_sessions = utilities.add_passive_flag_to_ophys_experiment_table(behavior_sessions)
    behavior_sessions = utilities.add_cell_type_column(behavior_sessions)

    return behavior_sessions


def get_filtered_ophys_experiment_table(include_failed_data=False, release_data_only=True, exclude_ai94=True,
                                        add_extra_columns=False, from_cached_file=False, overwrite_cached_file=False):
    """
    Loads a list of available ophys experiments FROM LIMS (not S3 cache) and adds additional useful columns to the table.
    By default, loads from a saved cached file.
    If cached file does not exist, loads list of available experiments directly from lims using SDK BehaviorProjectCache, and saves the reformatted table to the default Visual Behavior data cache location.

    Keyword Arguments:

        include_failed_data {bool} -- If True, return all experiments including those from failed containers and receptive field mapping experiments.
                                      If False, returns only experiments that have passed experiment level QC.
                                      Setting include_failed_data to True will automatically set release_data_only to False
                                      There is no guarantee on data quality or reprocessing for these experiments.
        release_data_only {bool} -- If True, return only experiments that were released on March 25th, 2021 and August 12, 2021.
                                    Fail tags and other extra columns will not be added if this is set to True.
                                    Release data includes project_codes = ['VisualBehavior', 'VisualBehaviorTask1B', 'VisualBehaviorMultiscope'].
                                    If False, return all Visual Behavior ophys experiments that have been collected, including data from project_code = 'VisualBehaviorMultiscope4areasx2d'.
                                    Note, if False, there is no guarantee on data quality or processing for these experiments.
        add_extra_columns {bool} -- Additional columns will be added, including fail tags, model availability and location string
        exclude_ai94 {bool} -- If True, exclude data from mice with Ai94(GCaMP6s) as the reporter line. (default: {True})
        from_cached_file {bool} -- If True, loads experiments table from saved file in default cache location (returned by get_production_cache_dir())
        overwrite_cached_file {bool} -- If True, saves experiment_table to default cache folder, overwrites existing file

    Returns:
        experiment_table -- returns a dataframe with ophys_experiment_id as the index and metadata as columns.
    """
    if include_failed_data is True:
        release_data_only = False
    if release_data_only:
        # get cache from lims for data released on March 25th
        print('getting experiment table for March and August releases from lims')
        cache = bpc.from_lims(data_release_date=['2021-03-25', '2021-08-12'])
        experiments = cache.get_ophys_experiment_table()
    if not release_data_only:
        if from_cached_file == True:
            if 'filtered_ophys_experiment_table.csv' in os.listdir(get_production_cache_dir()):
                filepath = os.path.join(get_production_cache_dir(), 'filtered_ophys_experiment_table.csv')
                print('loading cached experiment_table')
                print('last updated on:')
                import time
                print(time.ctime(os.path.getctime(filepath)))
                # load the cached file
                experiments = pd.read_csv(filepath)
            else:
                print('there is no filtered_ophys_experiment_table.csv', get_production_cache_dir())
        else:
            print('getting up-to-date experiment_table from lims')
            # get everything in lims
            cache = bpc.from_lims()
            experiments = cache.get_ophys_experiment_table(passed_only=False)
            # limit to the 4 VisualBehavior project codes
            experiments = filtering.limit_to_production_project_codes(experiments)
            if add_extra_columns:
                print('adding extra columns')
                print('NOTE: this is slow. set from_cached_file to True to load cached version of experiments_table at:')
                print(get_production_cache_dir())
                # create cre_line column, set NaN session_types to None, add model output availability and location columns
                experiments = reformat.reformat_experiments_table(experiments)
        if include_failed_data:
            print('including failed data')
            pass
        else:
            print('limiting to passed experiments')
            experiments = filtering.limit_to_passed_experiments(experiments)
            experiments = filtering.remove_failed_containers(experiments)  # container_workflow_state can be anything other than 'failed'
            # limit to sessions that start with OPHYS
            print('limiting to sessions that start with OPHYS')
            experiments = filtering.limit_to_valid_ophys_session_types(experiments)
    if experiments.index.name != 'ophys_experiment_id':
        experiments = experiments.drop_duplicates(subset='ophys_experiment_id')
        experiments = experiments.set_index('ophys_experiment_id')
    if exclude_ai94:
        print('excluding Ai94 data')
        experiments = experiments[experiments.full_genotype != 'Slc17a7-IRES2-Cre/wt;Camk2a-tTA/wt;Ai94(TITL-GCaMP6s)/wt']
    if 'cre_line' not in experiments.keys():
        experiments['cre_line'] = [full_genotype.split('/')[0] for full_genotype in experiments.full_genotype.values]
    # filter one more time on load to restrict to Visual Behavior project experiments ###
    experiments = filtering.limit_to_production_project_codes(experiments)

    # add new columns for conditions to analyze for platform paper ###
    experiments = utilities.add_cell_type_column(experiments)

    if overwrite_cached_file == True:
        print('overwriting pre-saved experiments table file')
        experiments.to_csv(os.path.join(get_production_cache_dir(), 'filtered_ophys_experiment_table.csv'))
    return experiments


def get_filtered_ophys_session_table(release_data_only=True, include_failed_data=False):
    """Get ophys sessions table from SDK, and add container_id and container_workflow_state to table,
        add session_workflow_state to table (defined as >1 experiment within session passing),
        and return only sessions where container and session workflow states are 'passed'.
        Includes Multiscope data.
            filtering criteria:
                project codes: VisualBehavior, VisualBehaviorTask1B,
                            VisualBehaviorMultiscope, VisualBehaviorMultiscope4areasx2d
                session workflow state: "passed"
                session_type: OPHYS_1_images_A', 'OPHYS_1_images_B',  'OPHYS_1_images_G',
                            'OPHYS_2_images_A_passive',  'OPHYS_2_images_B_passive',  'OPHYS_2_images_G_passive'
                            'OPHYS_3_images_A',  'OPHYS_3_images_B', 'OPHYS_3_images_G',
                            'OPHYS_4_images_A', 'OPHYS_4_images_B',  'OPHYS_4_images_H'
                            'OPHYS_5_images_A_passive', 'OPHYS_5_images_B_passive', 'OPHYS_5_images_H_passive'
                            'OPHYS_6_images_A',  'OPHYS_6_images_B',   'OPHYS_6_images_H'
    Returns:
        dataframe -- filtered version of the ophys sessions table(filtering criteria above) with the
                        following columns:
                        "ophys_session_id": df index, 9 digit unique identifier for an ophys session
                        "ophys_experiment_id": 9 digit unique identifier for an ophys experiment
                        "project_code": project code associated with the experiment and session
                        "session_name":
                        "session_type":
                        "equipment_name":
                        "date_of_acquisition":
                        "specimen_id":
                        "reporter_line":
                        "driver_line":
                        "at_least_one_experiment_passed":
                        "session_workflow_state":
                        "container_id":
                        "container_workflow_state":
    """
    cache = bpc.from_lims()
    sessions = cache.get_ophys_session_table()
    if release_data_only == False:
        from_cached_file = True
    else:
        from_cached_file = False
    experiment_table = get_filtered_ophys_experiment_table(release_data_only=release_data_only,
                                                           include_failed_data=include_failed_data,
                                                           from_cached_file=from_cached_file)
    sessions = filtering.limit_to_production_project_codes(sessions)
    sessions = reformat.add_all_qc_states_to_ophys_session_table(sessions, experiment_table)
    sessions = filtering.limit_to_valid_ophys_session_types(sessions)
    sessions = filtering.limit_to_passed_ophys_sessions(sessions)
    sessions = filtering.remove_failed_containers(sessions)
    sessions = reformat.add_model_outputs_availability_to_table(sessions)

    return sessions


def get_filtered_behavior_session_table(release_data_only=True):
    """
    Loads list of behavior sessions from SDK BehaviorProjectCache, and does some basic filtering and addition of columns, such as changing mouse_id from str to int and adding project code.

    Keyword Arguments:
        release_data_only {bool} -- If True, only return behavior sessions for mice with ophys data that will be included in the March data release.
        This does not include OPHYS_7_receptive_field_mapping or sessions with an unexpected session_type (i.e. where session_type is NaN)

    Returns:
        behavior_sessions -- Dataframe with behavior_session_id as the index and metadata as columns.
    """
    cache = bpc.from_lims()
    behavior_sessions = cache.get_behavior_session_table()
    behavior_sessions = behavior_sessions.reset_index()
    # make mouse_id an int not string
    behavior_sessions['mouse_id'] = [int(mouse_id) for mouse_id in behavior_sessions.mouse_id.values]
    # add project code from experiments table
    all_experiments = cache.get_ophys_experiment_table()
    all_experiments['mouse_id'] = [int(mouse_id) for mouse_id in all_experiments.mouse_id.values]
    behavior_sessions = behavior_sessions.merge(all_experiments[['mouse_id']], on='mouse_id')
    if release_data_only:
        # limit to mice that are in the data release & have a valid session_type
        release_experiments = get_filtered_ophys_experiment_table(release_data_only=True)
        release_mice = release_experiments.mouse_id.unique()
        behavior_sessions = behavior_sessions[behavior_sessions.mouse_id.isin(release_mice)]
        behavior_sessions = behavior_sessions[behavior_sessions.session_type.isnull() == False]
        behavior_sessions = behavior_sessions[behavior_sessions.session_type != 'OPHYS_7_receptive_field_mapping']
        behavior_sessions['has_passing_ophys_data'] = [True if behavior_session_id in release_experiments.behavior_session_id.values
                                                       else False for behavior_session_id in behavior_sessions.behavior_session_id]
    behavior_sessions = behavior_sessions.drop_duplicates(subset=['behavior_session_id'])
    behavior_sessions = behavior_sessions.set_index('behavior_session_id')
    return behavior_sessions


def get_second_release_candidates():
    """
    Preliminary function to get candidates for August release. Will be revised.
    :return:
    """
    full_cache = bpc.from_lims()
    full_experiment_table = full_cache.get_ophys_experiment_table()

    unreleased_complete_multiscope = full_experiment_table[
        full_experiment_table.project_code.isin(['VisualBehaviorMultiscope']) &
        (full_experiment_table.container_workflow_state.isin(['completed', 'container_qc'])) &
        (full_experiment_table.experiment_workflow_state == 'passed')]
    print(len(unreleased_complete_multiscope), 'un-released VisualBehaviorMultiscope where container_workflow_state == completed')

    unreleased_not_failed_4x2 = full_experiment_table[(full_experiment_table.project_code.isin(['VisualBehaviorMultiscope4areasx2d']) &
                                                       (full_experiment_table.container_workflow_state != 'failed') &
                                                       (full_experiment_table.experiment_workflow_state == 'passed'))]
    print(len(unreleased_not_failed_4x2), 'un-released VisualBehaviorMultiscope4areasx2d where container_workflow_state != failed')

    release_candidates = pd.concat([unreleased_complete_multiscope, unreleased_not_failed_4x2])
    print(len(release_candidates), 'release candidates')

    return release_candidates


def get_extended_stimulus_presentations_table(stimulus_presentations, licks, rewards, running_speed, eye_tracking=None, behavior_session_id=None):
    """
    Takes SDK stimulus presentations table and adds a bunch of useful columns by incorporating data from other tables
    and reformatting existing column data
    Additional columns include epoch #s for 10 minute bins in the session, whether a flash was a pre or post change or omission,
    the mean running speed per flash, mean pupil area per flash, licks per flash, rewards per flash, lick rate, reward rate,
    time since last change, time since last omission, time since last lick

    Set eye_tracking to None by default so that things still run for behavior only sessions
    If behavior_session_id is provided, will load metrics from behavior model outputs file
    """
    if 'time' in licks.keys():
        licks = licks.rename(columns={'time': 'timestamps'})
    if 'orientation' in stimulus_presentations.columns:
        stimulus_presentations = stimulus_presentations.drop(columns=['orientation', 'image_set', 'index',
                                                                      'phase', 'spatial_frequency'])
    stimulus_presentations = reformat.add_change_each_flash(stimulus_presentations)
    stimulus_presentations['pre_change'] = stimulus_presentations['change'].shift(-1)
    stimulus_presentations['pre_omitted'] = stimulus_presentations['omitted'].shift(-1)
    stimulus_presentations = reformat.add_epoch_times(stimulus_presentations)
    stimulus_presentations = reformat.add_mean_running_speed(stimulus_presentations, running_speed)
    if eye_tracking is not None:
        try:  # if eye tracking data is not present or cant be loaded
            stimulus_presentations = reformat.add_mean_pupil_area(stimulus_presentations, eye_tracking)
        except BaseException:  # set to NaN
            stimulus_presentations['mean_pupil_area'] = np.nan
    stimulus_presentations = reformat.add_licks_each_flash(stimulus_presentations, licks)
    stimulus_presentations = reformat.add_response_latency(stimulus_presentations)
    stimulus_presentations = reformat.add_rewards_each_flash(stimulus_presentations, rewards)
    stimulus_presentations['licked'] = [True if len(licks) > 0 else False for licks in
                                        stimulus_presentations.licks.values]
    # lick rate per second
    stimulus_presentations['lick_rate'] = stimulus_presentations['licked'].rolling(window=320, min_periods=1,
                                                                                   win_type='triang').mean() / .75
    stimulus_presentations['rewarded'] = [True if len(rewards) > 0 else False for rewards in stimulus_presentations.rewards.values]
    # (rewards/stimulus)*(1 stimulus/.750s) = rewards/second
    stimulus_presentations['reward_rate_per_second'] = stimulus_presentations['rewarded'].rolling(window=320, min_periods=1,
                                                                                                  win_type='triang').mean() / .75  # units of rewards per second
    # (rewards/stimulus)*(1 stimulus/.750s)*(60s/min) = rewards/min
    stimulus_presentations['reward_rate'] = stimulus_presentations['rewarded'].rolling(window=320, min_periods=1, win_type='triang').mean() * (60 / .75)  # units of rewards/min

    reward_threshold = 2 / 3  # 2/3 rewards per minute = 1/90 rewards/second
    stimulus_presentations['engaged'] = [x > reward_threshold for x in stimulus_presentations['reward_rate']]
    stimulus_presentations['engagement_state'] = ['engaged' if True else 'disengaged' for engaged in stimulus_presentations['engaged'].values]
    stimulus_presentations = reformat.add_response_latency(stimulus_presentations)
    stimulus_presentations = reformat.add_image_contrast_to_stimulus_presentations(stimulus_presentations)
    stimulus_presentations = reformat.add_time_from_last_lick(stimulus_presentations, licks)
    stimulus_presentations = reformat.add_time_from_last_reward(stimulus_presentations, rewards)
    stimulus_presentations = reformat.add_time_from_last_change(stimulus_presentations)
    try:  # behavior only sessions dont have omissions
        stimulus_presentations = reformat.add_time_from_last_omission(stimulus_presentations)
        stimulus_presentations['flash_after_omitted'] = stimulus_presentations['omitted'].shift(1)
    except BaseException:
        pass
    stimulus_presentations['flash_after_change'] = stimulus_presentations['change'].shift(1)
    stimulus_presentations['image_name_next_flash'] = stimulus_presentations['image_name'].shift(-1)
    stimulus_presentations['image_index_next_flash'] = stimulus_presentations['image_index'].shift(-1)
    stimulus_presentations['image_name_previous_flash'] = stimulus_presentations['image_name'].shift(1)
    stimulus_presentations['image_index_previous_flash'] = stimulus_presentations['image_index'].shift(1)
    stimulus_presentations['lick_on_next_flash'] = stimulus_presentations['licked'].shift(-1)
    stimulus_presentations['lick_rate_next_flash'] = stimulus_presentations['lick_rate'].shift(-1)
    stimulus_presentations['lick_on_previous_flash'] = stimulus_presentations['licked'].shift(1)
    stimulus_presentations['lick_rate_previous_flash'] = stimulus_presentations['lick_rate'].shift(1)
    # if behavior_session_id:
    #     if check_if_model_output_available(behavior_session_id):
    #         stimulus_presentations = add_model_outputs_to_stimulus_presentations(
    #             stimulus_presentations, behavior_session_id)
    #     else:
    #         print('model outputs not available')
    return stimulus_presentations


def get_stimulus_response_df(dataset, time_window=[-1, 2.1], interpolate=True, output_sampling_rate=30.,
                             data_type='filtered_events', load_from_file=True):
    """
    load stimulus response df using mindscope_utilities and merge with stimulus_presentations that has trials metadata added
    inputs:
        dataset: BehaviorOphysExperiment instance
        time_window: window over which to extract the event triggered response around each stimulus presentation time
        interpolate: Boolean, whether or not to interpolate traces
        output_sampling_rate: sampling rate for interpolation, only used if interpolate is True
        data_type: which timeseries to get event triggered responses for
                    options: 'filtered_events', 'events', 'dff', 'running_speed', 'pupil_diameter', 'lick_rate'
    """
    import mindscope_utilities.visual_behavior_ophys.data_formatting as vb_ophys
    # load stimulus response df from file if it exists otherwise generate it
    ophys_experiment_id = dataset.ophys_experiment_id
    filepath = os.path.join(get_stimulus_response_df_dir(interpolate, int(output_sampling_rate)), str(ophys_experiment_id) + '_' + data_type + '.h5')
    if load_from_file:
        if os.path.exists(filepath):
            print('loading response df from file for', ophys_experiment_id, data_type)
            sdf = pd.read_hdf(filepath)
        else:
            print('stimulus_response_df does not exist for', filepath)
            print('set load_from_file to False to generate new stimulus_response_df')
            sdf = pd.DataFrame(columns=['stimulus_presentations_id'])
    else:
        print('generating response df')
        sdf = vb_ophys.get_stimulus_response_df(dataset, data_type=data_type, event_type='all',
                                                time_window=time_window, interpolate=interpolate,
                                                output_sampling_rate=output_sampling_rate)

    # if extended_stimulus_presentations is an attribute of the dataset object, use it, otherwise get regular stimulus_presentations
    if 'extended_stimulus_presentations' in dir(dataset):
        stimulus_presentations = dataset.extended_stimulus_presentations.copy()
    else:
        stimulus_presentations = vb_ophys.get_annotated_stimulus_presentations(dataset)
    sdf = sdf.merge(stimulus_presentations, on='stimulus_presentations_id')

    return sdf


# LOAD OPHYS DATA FROM SDK AND EDIT OR ADD METHODS/ATTRIBUTES WITH BUGS OR INCOMPLETE FEATURES #


class BehaviorOphysDataset(BehaviorOphysExperiment):
    """
    Loads SDK ophys experiment object and 1) optionally filters out invalid ROIs, 2) adds extended_stimulus_presentations table, 3) adds extended_trials table, 4) adds behavior movie PCs and timestamps

    Returns:
        BehaviorOphysDataset {class} -- object with attributes & methods to access ophys and behavior data
                                            associated with an ophys_experiment_id (single imaging plane)
    """

    def __init__(self, api, include_invalid_rois=False,
                 eye_tracking_z_threshold: float = 3.0, eye_tracking_dilation_frames: int = 2,
                 events_filter_scale: float = 2.0, events_filter_n_time_steps: int = 20):
        """
        :param session: BehaviorOphysExperiment {class} -- instance of allenSDK BehaviorOphysExperiment object for one ophys_experiment_id
        :param _include_invalid_rois: if True, do not filter out invalid ROIs from cell_specimens_table and dff_traces
        """
        super().__init__(
            api=api,
            eye_tracking_z_threshold=eye_tracking_z_threshold,
            eye_tracking_dilation_frames=eye_tracking_dilation_frames,
            events_filter_scale=events_filter_scale,
            events_filter_n_time_steps=events_filter_n_time_steps
        )

        self._include_invalid_rois = include_invalid_rois

    @property
    def cell_specimen_table(self):
        cell_specimen_table = super().cell_specimen_table.copy()
        if self._include_invalid_rois == False:
            cell_specimen_table = cell_specimen_table[cell_specimen_table.valid_roi == True]
        self._cell_specimen_table = cell_specimen_table
        return self._cell_specimen_table

    @property
    def corrected_fluorescence_traces(self):
        if self._include_invalid_rois == False:
            corrected_fluorescence_traces = super().corrected_fluorescence_traces
            cell_specimen_table = super().cell_specimen_table[super().cell_specimen_table.valid_roi == True]
            valid_cells = cell_specimen_table.cell_roi_id.values
            self._corrected_fluorescence_traces = corrected_fluorescence_traces[
                corrected_fluorescence_traces.cell_roi_id.isin(valid_cells)]
        else:
            self._corrected_fluorescence_traces = super().corrected_fluorescence_traces
        return self._corrected_fluorescence_traces

    @property
    def dff_traces(self):
        if self._include_invalid_rois == False:
            dff_traces = super().dff_traces
            cell_specimen_table = super().cell_specimen_table[super().cell_specimen_table.valid_roi == True]
            valid_cells = cell_specimen_table.cell_roi_id.values
            self._dff_traces = dff_traces[dff_traces.cell_roi_id.isin(valid_cells)]
        else:
            self._dff_traces = super().dff_traces
        return self._dff_traces

    @property
    def events(self):
        if self._include_invalid_rois == False:
            events = super().events
            cell_specimen_table = super().cell_specimen_table[super().cell_specimen_table.valid_roi == True]
            valid_cells = cell_specimen_table.cell_roi_id.values
            self._events = events[events.cell_roi_id.isin(valid_cells)]
        else:
            self._events = super().events
        return self._events

    @property
    def metadata(self):
        # for figure titles & filenames
        metadata = super().metadata
        # metadata['mouse_id'] = metadata['LabTracks_ID']
        # metadata['equipment_name'] = metadata['rig_name']
        # metadata['date_of_acquisition'] = metadata['experiment_datetime']
        self._metadata = metadata
        return self._metadata

    @property
    def metadata_string(self):
        # for figure titles & filenames
        m = self.metadata
        rig_name = m['equipment_name'].split('.')[0] + m['equipment_name'].split('.')[1]
        self._metadata_string = str(m['mouse_id']) + '_' + str(m['ophys_experiment_id']) + '_' + m['driver_line'][
            0] + '_' + m['targeted_structure'] + '_' + str(m['imaging_depth']) + '_' + m['session_type'] + '_' + rig_name
        return self._metadata_string

    @property
    def extended_stimulus_presentations(self):
        extended_stimulus_presentations = get_extended_stimulus_presentations_table(self.stimulus_presentations.copy(),
                                                                                    self.licks, self.rewards, self.running_speed, self.eye_tracking)
        self._extended_stimulus_presentations = extended_stimulus_presentations
        return self._extended_stimulus_presentations

    @property
    def extended_trials(self):
        trials = super().trials.copy()
        trials = reformat.add_epoch_times(trials)
        trials = reformat.add_trial_type_to_trials_table(trials)
        trials = reformat.add_reward_rate_to_trials_table(trials, self.extended_stimulus_presentations)
        trials = reformat.add_engagement_state_to_trials_table(trials, self.extended_stimulus_presentations)
        self._extended_trials = trials
        return self._extended_trials

    @property
    def behavior_movie_timestamps(self):
        lims_data = utilities.get_lims_data(self.ophys_experiment_id)
        timestamps = utilities.get_timestamps(lims_data)
        self._behavior_movie_timestamps = timestamps['behavior_monitoring']['timestamps'].copy()
        return self._behavior_movie_timestamps

    @property
    def behavior_movie_pc_masks(self):
        ophys_session_id = from_lims.get_ophys_session_id_for_ophys_experiment_id(self.ophys_experiment_id)
        self._behavior_movie_pc_masks = get_pc_masks_for_session(ophys_session_id)
        return self._behavior_movie_pc_masks

    @property
    def behavior_movie_pc_activations(self):
        ophys_session_id = from_lims.get_ophys_session_id_for_ophys_experiment_id(self.ophys_experiment_id)
        self._behavior_movie_pc_activations = get_pc_activations_for_session(ophys_session_id)
        return self._behavior_movie_pc_activations

    @property
    def behavior_movie_predictions(self):
        ophys_session_id = from_lims.get_ophys_session_id_for_ophys_experiment_id(self.ophys_experiment_id)
        movie_predictions = get_behavior_movie_predictions_for_session(ophys_session_id)
        movie_predictions.index.name = 'frame_index'
        movie_predictions['timestamps'] = self.behavior_movie_timestamps[:len(
            movie_predictions)]  # length check will trim off spurious timestamps at the end
        self._behavior_movie_predictions = movie_predictions
        return self._behavior_movie_predictions

    def get_cell_specimen_id_for_cell_index(self, cell_index):
        cell_specimen_table = self.cell_specimen_table.copy()
        cell_specimen_id = cell_specimen_table[cell_specimen_table.cell_index == cell_index].index.values[0]
        return cell_specimen_id

    def get_cell_index_for_cell_specimen_id(self, cell_specimen_id):
        cell_specimen_table = self.cell_specimen_table.copy()
        cell_index = cell_specimen_table[cell_specimen_table.index == cell_specimen_id].cell_index.values[0]
        return cell_index

    def get_cell_specimen_id_for_cell_roi_id(self, cell_roi_id):
        cell_specimen_table = self.cell_specimen_table.copy()
        cell_specimen_id = cell_specimen_table[cell_specimen_table.cell_roi_id == cell_roi_id].index.values[0]
        return cell_specimen_id


def get_ophys_dataset(ophys_experiment_id, include_invalid_rois=False, load_from_lims=False, load_from_nwb=True,
                      get_extended_stimulus_presentations=False, get_behavior_movie_timestamps=False):
    """
    Gets behavior + ophys data for one experiment (single imaging plane), either using the SDK LIMS API,
    SDK NWB API, or using BehaviorOphysDataset wrapper which inherits the LIMS API BehaviorOphysSession object,
    and adds functionality including invalid ROI filtering, extended stimulus_presentations and trials, and behavior movie data.

    Arguments:
        ophys_experiment_id {int} -- 9 digit ophys experiment ID
        include_invalid_rois {Boolean} -- if True, return all ROIs including invalid. If False, filter out invalid ROIs
        load_from_lims -- if True, loads dataset directly from BehaviorOphysSession.from_lims(). Invalid ROIs will be included.
        load_from_nwb -- if True, loads dataset directly from BehaviorOphysSession.from_nwb_path(). Invalid ROIs will not be included.
        get_extended_stimulus_presentations -- if True, adds an attribute "extended_stimulus_presentations" to the dataset object

        If both from_lims and from_nwb are set to False, an exception will be raised

    Returns:
        object -- BehaviorOphysSession or BehaviorOphysDataset instance, which inherits attributes & methods from SDK BehaviorOphysSession
    """

    id_type = from_lims.get_id_type(ophys_experiment_id)
    if id_type != 'ophys_experiment_id':
        warnings.warn('It looks like you passed an id of type {} instead of an ophys_experiment_id'.format(id_type))

    assert id_type == 'ophys_experiment_id', "The passed ID type is {}. It must be an ophys_experiment_id".format(id_type)

    if load_from_lims:
        dataset = BehaviorOphysExperiment.from_lims(int(ophys_experiment_id))
    elif load_from_nwb:
        cache_dir = get_platform_analysis_cache_dir()
        cache = bpc.from_s3_cache(cache_dir=cache_dir)
        dataset = cache.get_behavior_ophys_experiment(ophys_experiment_id)
    else:
        raise Exception('Set load_from_lims or load_from_nwb to True')

    if get_extended_stimulus_presentations:
        # add extended stimulus presentations
        dataset.extended_stimulus_presentations = get_extended_stimulus_presentations_table(
            dataset.stimulus_presentations.copy(),
            dataset.licks, dataset.rewards,
            dataset.running_speed, dataset.eye_tracking)
    if get_behavior_movie_timestamps:
        # add behavior movie timestamps
        lims_data = utilities.get_lims_data(ophys_experiment_id)
        timestamps = utilities.get_timestamps(lims_data)
        dataset.behavior_movie_timestamps = timestamps['behavior_monitoring']['timestamps'].copy()

    return dataset


class BehaviorDataset(BehaviorSession):
    """
    Loads SDK behavior session object and adds extended_stimulus_presentations and extended trials tables.

    Returns:
        BehaviorDataset {class} -- object with attributes & methods to access behavior data associated with a behavior_session_id
    """

    def __init__(self, api):
        """
        :param session: BehaviorSession {class} -- instance of allenSDK BehaviorSession object for one behavior_session_id
        """
        super().__init__(api)

    @property
    def metadata(self):
        metadata = super().metadata
        self._metadata = metadata
        return self._metadata

    @property
    def metadata_string(self):
        # for figure titles & filenames
        m = self.metadata
        rig_name = m['equipment_name'].split('.')[0] + m['equipment_name'].split('.')[1]
        self._metadata_string = str(m['mouse_id']) + '_' + str(m['behavior_session_id']) + '_' + m['driver_line'][
            0] + '_' + m['session_type'] + '_' + rig_name
        return self._metadata_string

    @property
    def extended_stimulus_presentations(self):
        stimulus_presentations = self.stimulus_presentations.copy()
        stimulus_presentations = reformat.add_change_each_flash(stimulus_presentations)
        stimulus_presentations['pre_change'] = stimulus_presentations['change'].shift(-1)
        stimulus_presentations['pre_omitted'] = stimulus_presentations['omitted'].shift(-1)
        stimulus_presentations = reformat.add_epoch_times(stimulus_presentations)
        stimulus_presentations = reformat.add_mean_running_speed(stimulus_presentations, self.running_speed)
        stimulus_presentations = reformat.add_licks_each_flash(stimulus_presentations, self.licks)
        stimulus_presentations = reformat.add_response_latency(stimulus_presentations)
        stimulus_presentations = reformat.add_rewards_each_flash(stimulus_presentations, self.rewards)
        stimulus_presentations['licked'] = [True if len(licks) > 0 else False for licks in
                                            stimulus_presentations.licks.values]
        stimulus_presentations['lick_rate'] = stimulus_presentations['licked'].rolling(window=320, min_periods=1,
                                                                                       win_type='triang').mean() / .75
        stimulus_presentations['rewarded'] = [True if len(rewards) > 0 else False for rewards in
                                              stimulus_presentations.rewards.values]
        stimulus_presentations['reward_rate'] = stimulus_presentations['rewarded'].rolling(window=320, min_periods=1,
                                                                                           win_type='triang').mean()
        stimulus_presentations = reformat.add_response_latency(stimulus_presentations)
        stimulus_presentations = reformat.add_image_contrast_to_stimulus_presentations(stimulus_presentations)
        stimulus_presentations = reformat.add_time_from_last_lick(stimulus_presentations, self.licks)
        stimulus_presentations = reformat.add_time_from_last_reward(stimulus_presentations, self.rewards)
        stimulus_presentations = reformat.add_time_from_last_change(stimulus_presentations)
        stimulus_presentations['flash_after_change'] = stimulus_presentations['change'].shift(1)
        stimulus_presentations['image_name_next_flash'] = stimulus_presentations['image_name'].shift(-1)
        stimulus_presentations['image_index_next_flash'] = stimulus_presentations['image_index'].shift(-1)
        stimulus_presentations['image_name_previous_flash'] = stimulus_presentations['image_name'].shift(1)
        stimulus_presentations['image_index_previous_flash'] = stimulus_presentations['image_index'].shift(1)
        stimulus_presentations['lick_on_next_flash'] = stimulus_presentations['licked'].shift(-1)
        stimulus_presentations['lick_rate_next_flash'] = stimulus_presentations['lick_rate'].shift(-1)
        stimulus_presentations['lick_on_previous_flash'] = stimulus_presentations['licked'].shift(1)
        stimulus_presentations['lick_rate_previous_flash'] = stimulus_presentations['lick_rate'].shift(1)
        if check_if_model_output_available(self.metadata['behavior_session_id']):
            stimulus_presentations = add_model_outputs_to_stimulus_presentations(
                stimulus_presentations, self.metadata['behavior_session_id'])
        else:
            print('model outputs not available')
        self._extended_stimulus_presentations = stimulus_presentations
        return self._extended_stimulus_presentations

    @property
    def extended_trials(self):
        trials = super().trials.copy()
        trials = reformat.add_epoch_times(trials)
        trials = reformat.add_trial_type_to_trials_table(trials)
        trials = reformat.add_reward_rate_to_trials_table(trials, self.extended_stimulus_presentations)
        trials = reformat.add_engagement_state_to_trials_table(trials, self.extended_stimulus_presentations)
        self._extended_trials = trials
        return self._extended_trials


def get_extended_trials_table(trials, extended_stimulus_presentations):
    extended_trials = reformat.add_epoch_times(trials)
    extended_trials = reformat.add_trial_type_to_trials_table(extended_trials)
    extended_trials = reformat.add_reward_rate_to_trials_table(extended_trials, extended_stimulus_presentations)
    extended_trials = reformat.add_engagement_state_to_trials_table(extended_trials, extended_stimulus_presentations)
    return extended_trials


def get_behavior_dataset(behavior_session_id, from_lims=False, from_nwb=True,
                         get_extended_stimulus_presentations=False, get_extended_trials=True):
    """
    Gets behavior data for one session, either using the SDK LIMS API, SDK NWB API, or using BehaviorDataset wrapper which inherits the LIMS API BehaviorSession object, and adds access to extended stimulus_presentations and trials.

    Arguments:
        behavior_session_id {int} -- 9 digit behavior session ID
        from_lims -- if True, loads dataset directly from BehaviorSession.from_lims()
        from_nwb -- if True, loads dataset directly from BehaviorSession.from_nwb_path(), after converting behavior_session_id to nwb_path via lims query

        If both from_lims and from_nwb are set to False, an exception will be raised

    Returns:
        object -- BehaviorSession or BehaviorDataset instance
    """
    if from_lims:
        dataset = BehaviorSession.from_lims(behavior_session_id)
    elif from_nwb:
        cache_dir = get_platform_analysis_cache_dir()
        cache = bpc.from_s3_cache(cache_dir=cache_dir)
        dataset = cache.get_behavior_session(behavior_session_id)
    else:
        raise Exception('Set load_from_lims or load_from_nwb to True')

    if get_extended_stimulus_presentations:
        # add extended stimulus presentations
        dataset.extended_stimulus_presentations = get_extended_stimulus_presentations_table(
            dataset.stimulus_presentations.copy(),
            dataset.licks, dataset.rewards,
            dataset.running_speed, behavior_session_id=dataset.metadata['behavior_session_id'])

    if get_extended_trials and get_extended_stimulus_presentations is False:
        dataset.extended_stimulus_presentations = get_extended_stimulus_presentations_table(
            dataset.stimulus_presentations.copy(),
            dataset.licks, dataset.rewards,
            dataset.running_speed, behavior_session_id=dataset.metadata['behavior_session_id'])
        dataset.extended_trials = get_extended_trials_table(dataset.trials, dataset.extended_stimulus_presentations)
    elif get_extended_trials:
        dataset.extended_trials = get_extended_trials_table(dataset.trials, dataset.extended_stimulus_presentations)

    return dataset

#
# def get_ophys_container_ids(include_failed_data=False, release_data_only=False, exclude_ai94=True, add_extra_columns=False,
#                             from_cached_file=True, overwrite_cached_file=False):
#     """Get container_ids that meet the criteria indicated by flags, which are identical to those in get_filtered_ophys_experiment_table() """
#     experiments = get_filtered_ophys_experiment_table(include_failed_data=include_failed_data, release_data_only=release_data_only,
#                                                       exclude_ai94=exclude_ai94, add_extra_columns=add_extra_columns,
#                                                       from_cached_file=from_cached_file, overwrite_cached_file=overwrite_cached_file)
#     container_ids = np.sort(experiments.ophys_container_id.unique())
#     return container_ids


def get_ophys_container_ids(platform_paper_only=False, add_extra_columns=True):
    """
    Gets ophys_container_ids for all published datasets by default, or limits to platform paper containers if platform_paper_only is True
    :return:
    """
    if platform_paper_only:
        experiments = get_platform_paper_experiment_table(add_extra_columns=add_extra_columns)
    else:
        cache_dir = get_platform_analysis_cache_dir()
        cache = bpc.from_s3_cache(cache_dir)
        experiments = cache.get_ophys_experiment_table()
    container_ids = np.sort(experiments.ophys_container_id.unique())
    return container_ids


def get_ophys_session_ids_for_ophys_container_id(ophys_container_id):
    """Get ophys_session_ids belonging to a given ophys_container_id. Ophys session must pass QC.

            Arguments:
                ophys_container_id -- must be in get_ophys_container_ids()

            Returns:
                ophys_session_ids -- list of ophys_session_ids that meet filtering criteria
            """
    experiments = get_filtered_ophys_experiment_table()
    ophys_session_ids = np.sort(experiments[(experiments.ophys_container_id == ophys_container_id)].ophys_session_id.unique())
    return ophys_session_ids


def get_ophys_experiment_ids_for_ophys_container_id(ophys_container_id):
    """Get ophys_experiment_ids belonging to a given ophys_container_id. ophys container must meet the criteria in
        sdk_utils.get_filtered_session_table()

                Arguments:
                    ophys_container_id -- must be in get_filtered_ophys_container_ids()

                Returns:
                    ophys_experiment_ids -- list of ophys_experiment_ids that meet filtering criteria
                """
    experiments = get_filtered_ophys_experiment_table()
    ophys_experiment_ids = np.sort(experiments[(experiments.ophys_container_id == ophys_container_id)].index.values)
    return ophys_experiment_ids


def get_session_type_for_ophys_experiment_id(ophys_experiment_id):
    experiments = get_filtered_ophys_experiment_table()
    session_type = experiments.loc[ophys_experiment_id].session_type
    return session_type


def get_session_type_for_ophys_session_id(ophys_session_id):
    sessions = get_filtered_ophys_session_table()
    session_type = sessions.loc[ophys_session_id].session_type
    return session_type


def get_ophys_experiment_id_for_ophys_session_id(ophys_session_id):
    experiments = get_filtered_ophys_experiment_table()
    ophys_experiment_id = experiments[(experiments.ophys_session_id == ophys_session_id)].index.values[0]
    return ophys_experiment_id


def get_ophys_session_id_for_ophys_experiment_id(ophys_experiment_id):
    experiments = get_filtered_ophys_experiment_table()
    ophys_session_id = experiments.loc[ophys_experiment_id].ophys_session_id
    return ophys_session_id


def get_behavior_session_id_for_ophys_experiment_id(ophys_experiment_id):
    experiments = get_filtered_ophys_experiment_table(include_failed_data=True)
    behavior_session_id = experiments.loc[ophys_experiment_id].behavior_session_id
    return behavior_session_id


def get_pc_masks_for_session(ophys_session_id):
    facemap_output_dir = r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/facemap_results'
    session_file = [file for file in os.listdir(facemap_output_dir) if
                    (str(ophys_session_id) in file) and ('motMask' in file)]
    try:
        pc_masks = np.load(os.path.join(facemap_output_dir, session_file[0]))
    except Exception as e:
        print('could not load PC masks for ophys_session_id', ophys_session_id)
        print(e)
        pc_masks = []
    return pc_masks


def get_pc_activations_for_session(ophys_session_id):
    facemap_output_dir = r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/facemap_results'
    session_file = [file for file in os.listdir(facemap_output_dir) if
                    (str(ophys_session_id) in file) and ('motSVD' in file)]
    try:
        pc_activations = np.load(os.path.join(facemap_output_dir, session_file[0]))
    except Exception as e:
        print('could not load PC activations for ophys_session_id', ophys_session_id)
        print(e)
        pc_activations = []
    return pc_activations


def get_extended_stimulus_presentations_for_session(session):
    '''
    Calculates additional information for each stimulus presentation
    '''
    import visual_behavior.ophys.dataset.stimulus_processing as sp
    stimulus_presentations_pre = session.stimulus_presentations
    change_times = session.trials['change_time'].values
    change_times = change_times[~np.isnan(change_times)]
    extended_stimulus_presentations = sp.get_extended_stimulus_presentations(
        stimulus_presentations_df=stimulus_presentations_pre,
        licks=session.licks,
        rewards=session.rewards,
        change_times=change_times,
        running_speed_df=session.running_data_df,
        pupil_area=session.pupil_area
    )
    return extended_stimulus_presentations


def get_model_output_file(behavior_session_id):
    model_output_dir = get_behavior_model_outputs_dir()
    model_output_file = [file for file in os.listdir(model_output_dir) if
                         (str(behavior_session_id) in file) and ('training' not in file)]
    return model_output_file


def check_if_model_output_available(behavior_session_id):
    model_output_file = get_model_output_file(behavior_session_id)
    if len(model_output_file) > 0:
        return True
    else:
        return False


def load_behavior_model_outputs(behavior_session_id):
    '''
    loads the behavior model outputs from their default save location on the /allen filesystem

    Parameters:
    -----------
    behavior_session_id : int
        desired behavior session ID

    Returns:
    --------
    Pandas.DataFrame
        dataframe containing behavior model outputs
    '''

    # cast ID to int
    behavior_session_id = int(behavior_session_id)

    # check ID type to ensure that it is a behavior_session_id
    id_type = from_lims.get_id_type(behavior_session_id)
    assert id_type == 'behavior_session_id', "passed ID must be a behavior_session_id. A {} was passed instead".format(id_type)

    if check_if_model_output_available(behavior_session_id):
        model_outputs = pd.read_csv(
            os.path.join(
                get_behavior_model_outputs_dir(),
                get_model_output_file(behavior_session_id)[0]
            )
        )
        cols_to_drop = [
            'image_index',
            'image_name',
            'omitted',
            'change',
            'licked',
            'lick_rate',
            'rewarded',
            'reward_rate',
            'is_change'
        ]
        for col in cols_to_drop:
            try:
                model_outputs.drop(columns=[col], inplace=True)
            except KeyError:
                pass

    else:
        warnings.warn('no model outputs saved for behavior_session_id: {}'.format(behavior_session_id))
        model_outputs = None

    return model_outputs


def add_model_outputs_to_stimulus_presentations(stimulus_presentations, behavior_session_id):
    '''
       Adds additional columns to stimulus table for model weights and related metrics
    '''

    model_outputs = load_behavior_model_outputs(behavior_session_id)

    if model_outputs is not None:
        stimulus_presentations = stimulus_presentations.merge(model_outputs, right_on='stimulus_presentations_id',
                                                              left_on='stimulus_presentations_id').set_index(
            'stimulus_presentations_id')
        stimulus_presentations['engagement_state'] = ['engaged' if x else 'disengaged' for x in stimulus_presentations['engaged']]
        stimulus_presentations = stimulus_presentations.drop(
            columns=['hit_rate', 'miss_rate', 'false_alarm_rate', 'correct_reject_rate', 'd_prime', 'criterion'])
        return stimulus_presentations

    return stimulus_presentations


def get_behavior_model_summary_table():
    data_dir = get_behavior_model_outputs_dir()
    data = pd.read_pickle(os.path.join(data_dir, '_summary_table.pkl'))
    return data


def check_for_events_file(ophys_experiment_id):
    # events_folder = os.path.join(get_analysis_cache_dir(), 'events')
    events_folder = os.path.join(get_events_dir())
    if os.path.exists(events_folder):
        events_file = [file for file in os.listdir(events_folder) if
                       str(ophys_experiment_id) in file]
        if len(events_file) > 0:
            return True
        else:
            return False


def get_behavior_movie_predictions_for_session(ophys_session_id):
    """
    Loads model predictions from behavior movie classifier and returns a dictionary with keys =
    ['groom_reach_with_contact', 'groom_reach_without_contact', 'lick_with_contact', 'lick_without_contact', 'no_contact', 'paw_contact']
    :param ophys_session_id: ophys_session_id
    :return: dictionary of behavior prediction values
    """
    model_output_dir = '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/lick_detection_validation/model_predictions'
    predictions_path = glob.glob(os.path.join(model_output_dir, 'predictions*osid={}*'.format(ophys_session_id)))[0]
    try:
        movie_predictions = pd.read_csv(predictions_path)
    except Exception as e:
        print('could not behavior movie model predictions for ophys_session_id', ophys_session_id)
        print(e)
        movie_predictions = []
    return movie_predictions


def get_sdk_max_projection(ophys_experiment_id):
    """ uses SDK to return 2d max projection image of the microscope field of view


    Arguments:
        ophys_experiment_id {int} -- 9 digit ophys experiment ID

    Returns:
        image -- can be visualized via plt.imshow(max_projection)
    """
    session = get_ophys_dataset(ophys_experiment_id)
    max_projection = session.max_projection
    return max_projection


def get_sdk_ave_projection(ophys_experiment_id):
    """uses SDK to return 2d image of the 2-photon microscope filed of view, averaged
        across the experiment.

    Arguments:
        ophys_experiment_id {int} -- 9 digit ophys experiment ID

    Returns:
        image -- can be visualized via plt.imshow(ave_projection)
    """
    session = get_ophys_dataset(ophys_experiment_id)
    ave_projection = session.average_projection
    return ave_projection


def get_sdk_segmentation_mask_image(ophys_experiment_id):
    """uses SDK to return an array containing the masks of all cell ROIS

    Arguments:
       ophys_experiment_id {int} -- 9 digit ophys experiment ID

    Returns:
       array -- a 2D boolean array
                visualized via plt.imshow(seg_mask_image)
    """
    session = get_ophys_dataset(ophys_experiment_id)
    seg_mask_image = session.segmentation_mask_image.data
    return seg_mask_image


def get_sdk_roi_masks(cell_specimen_table):
    """uses sdk to return a dictionary with individual ROI
        masks for each cell specimen ID.

    Arguments:
        cell_specimen_table {DataFrame} -- cell_specimen_table from SDK session object

    Returns:
        dictonary -- keys are cell specimen ids(ints)
                    values are 2d numpy arrays(binary array
                    the size of the motion corrected 2photon
                    FOV where 1's are the ROI/Cell mask).
                    specific cell masks can be visualized via
                    plt.imshow(roi_masks[cell_specimen_id])
    """

    roi_masks = {}
    for cell_roi_id in cell_specimen_table.cell_roi_id.values:
        mask = cell_specimen_table[cell_specimen_table.cell_roi_id == cell_roi_id]['roi_mask'].values[0]
        binary_mask = np.zeros(mask.shape)
        binary_mask[mask == True] = 1
        roi_masks[cell_roi_id] = binary_mask
    return roi_masks


def get_segmentation_mask(ophys_experiment_id, valid_only=True):
    dataset = get_ophys_dataset(ophys_experiment_id, include_invalid_rois=True)
    cell_specimen_table = dataset.cell_specimen_table.copy()
    if valid_only == True:
        roi_masks = get_sdk_roi_masks(cell_specimen_table[cell_specimen_table.valid_roi == True])
    else:
        roi_masks = get_sdk_roi_masks(cell_specimen_table)
    # flatten
    segmentation_mask = np.sum(np.asarray(list(roi_masks.values())), axis=0)
    segmentation_mask[segmentation_mask > 0] = 1
    return segmentation_mask


def get_metrics_df(experiment_id):
    metrics_df = load_current_objectlisttxt_file(experiment_id)
    # ROI locations from lims, including cell_roi_id
    roi_loc = roi_locations_from_cell_rois_table(experiment_id)
    # limit to current segmentation run, otherwise gives old ROIs
    run_id = get_current_segmentation_run_id(experiment_id)
    roi_loc = roi_loc[roi_loc.ophys_cell_segmentation_run_id == run_id]
    # link ROI metrics with cell_roi_id from ROI locations dict using ROI location
    metrics_df = metrics_df.merge(roi_loc, on=['bbox_min_x', 'bbox_min_y'])
    return metrics_df


def get_roi_mask_and_metrics_dict(cell_table, metrics_df, metric):
    roi_mask_dict = {}
    metrics_dict = {}
    for cell_roi_id in cell_table.cell_roi_id.values:
        metrics_dict[cell_roi_id] = metrics_df[metrics_df.cell_roi_id == cell_roi_id][metric].values[0]
        roi_mask = cell_table[cell_table.cell_roi_id == cell_roi_id].roi_mask.values[0]
        mask = np.zeros(roi_mask.shape)
        mask[:] = np.nan
        mask[roi_mask == True] = 1
        roi_mask_dict[cell_roi_id] = mask
    return roi_mask_dict, metrics_dict


def get_sdk_cell_specimen_table(ophys_experiment_id):
    """[summary]

    Arguments:
        ophys_experiment_id {[type]} -- [description]

    Returns:
        Dataframe -- dataframe with the following columns:
                    "cell_specimen_id": index
                    "cell_roi_id"
                    "height"
                    "image_mask"
                    "mask_image_plane"
                    "max_correction_down"
                    "max_correction_left"
                    "max_correction_right"
                    "max_correction_up"
                    "valid_roi"
                    "width"
                    "x"
                    "y"
    """
    session = get_ophys_dataset(ophys_experiment_id)
    cell_specimen_table = session.cell_specimen_table
    return cell_specimen_table


def get_sdk_dff_traces(ophys_experiment_id):
    session = get_ophys_dataset(ophys_experiment_id)
    dff_traces = session.dff_traces
    return dff_traces


def get_sdk_dff_traces_array(ophys_experiment_id):
    dff_traces = get_sdk_dff_traces(ophys_experiment_id)
    dff_traces_array = np.vstack(dff_traces.dff.values)
    return dff_traces_array


def get_sdk_running_speed(ophys_session_id):
    session = get_ophys_dataset(get_ophys_experiment_id_for_ophys_session_id(ophys_session_id))
    running_speed = session.running_data_df['speed']
    return running_speed


def get_sdk_trials(ophys_session_id):
    session = get_ophys_dataset(get_ophys_experiment_id_for_ophys_session_id(ophys_session_id))
    trials = session.trials.reset_index()
    return trials


def get_stim_metrics_summary(behavior_session_id, load_location='from_file'):
    '''
    gets flashwise stimulus presentation summary including behavior model weights

    inputs:
        behavior_session_id (int): LIMS behavior_session_id
        load_location (int): location from which to load data
            'from_file' (default) loads from a CSV on disk
            'from_database' loads from a Mongo database

    returns:
        a pandas dataframe containing columns describing stimulus
        information for each stimulus presentation
    '''
    if load_location == 'from_file':
        stim_metrics_summary_path = "/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/" \
                                    + "flashwise_metric_summary_2020.04.14.csv"
        stim_metrics_summary = pd.read_csv(stim_metrics_summary_path)
        return stim_metrics_summary.query('behavior_session_id == @behavior_session_id').copy()
    elif load_location == 'from_database':
        conn = db.Database('visual_behavior_data')
        collection = conn['behavior_analysis']['annotated_stimulus_presentations']
        df = pd.DataFrame(list(collection.find({'behavior_session_id': int(behavior_session_id)})))
        conn.close()
        return df.sort_values(by=['behavior_session_id', 'flash_index'])


# FROM LIMS DATABASE
# this portion is depreciated, please use functions in from_lims.py instead
gen_depr_str = 'this function is deprecated and will be removed on October 29th 2021, ' \
               + 'please use {}.{} instead'

# EXPERIMENT LEVEL


def get_lims_experiment_info(ophys_experiment_id):
    """uses an sqlite query to retrieve ophys experiment information
        from the lims2 database

    Arguments:
        ophys_experiment_id {int} -- 9 digit ophys experiment ID

    Returns:
        table -- table with the following columns:
                    "ophys_experiment_id":
                    "experiment_workflow_state":
                    "ophys_session_id":
                    "ophys_container_id":
                    "date_of_acquisition":
                    "stage_name_lims":
                    "foraging_id":
                    "mouse_info":
                    "mouse_donor_id":
                    "targeted_structure":
                    "depth":
                    "rig":

    """
    warn_str = gen_depr_str.format('from_lims',
                                   'get_general_info_for_ophys_experiment_id')
    warnings.warn(warn_str)

    ophys_experiment_id = int(ophys_experiment_id)
    mixin = lims_engine
    # build query
    query = '''
    select

    oe.id as ophys_experiment_id,
    oe.workflow_state,
    oe.ophys_session_id,

    container.visual_behavior_experiment_container_id as ophys_container_id,

    os.date_of_acquisition,
    os.stimulus_name as stage_name_lims,
    os.foraging_id,
    oe.workflow_state as experiment_workflow_state,

    specimens.name as mouse_info,
    specimens.donor_id as mouse_donor_id,
    structures.acronym as targeted_structure,
    imaging_depths.depth,
    equipment.name as rig

    from
    ophys_experiments_visual_behavior_experiment_containers container
    join ophys_experiments oe on oe.id = container.ophys_experiment_id
    join ophys_sessions os on os.id = oe.ophys_session_id
    join specimens on specimens.id = os.specimen_id
    join structures on structures.id = oe.targeted_structure_id
    join imaging_depths on imaging_depths.id = oe.imaging_depth_id
    join equipment on equipment.id = os.equipment_id

    where oe.id = {}'''.format(ophys_experiment_id)

    lims_experiment_info = mixin.select(query)

    return lims_experiment_info


def get_current_segmentation_run_id(ophys_experiment_id):
    """gets the id for the current cell segmentation run for a given experiment.
        Queries LIMS via AllenSDK PostgresQuery function.

    Arguments:
       ophys_experiment_id {int} -- 9 digit ophys experiment ID

    Returns:
        int -- current cell segmentation run id
    """
    warn_str = gen_depr_str.format('from_lims',
                                   'get_current_segmentation_run_id_for_ophys_experiment_id')
    warnings.warn(warn_str)

    segmentation_run_table = get_lims_cell_segmentation_run_info(ophys_experiment_id)
    current_segmentation_run_id = segmentation_run_table.loc[segmentation_run_table["current"] == True, ["id"][0]].values[0]
    return current_segmentation_run_id


def get_lims_cell_segmentation_run_info(experiment_id):
    """Queries LIMS via AllenSDK PostgresQuery function to
    retrieve information on all segmentations run in the
    ophys_cell_segmenatation_runs table for a given experiment

    Returns:
        dataframe -- dataframe with the following columns:
            id {int}:  9 digit segmentation run id
            run_number {int}: segmentation run number
            ophys_experiment_id{int}: 9 digit ophys experiment id
            current{boolean}: True/False True: most current segmentation run;
                              False: not the most current segmentation run
            created_at{timestamp}:
            updated_at{timestamp}:
    """
    warn_str = gen_depr_str.format('from_lims',
                                   'get_cell_segmentation_runs_table')
    warnings.warn(warn_str)

    mixin = lims_engine
    query = '''
    select *
    FROM ophys_cell_segmentation_runs
    WHERE ophys_experiment_id = {} '''.format(experiment_id)
    return mixin.select(query)


def get_lims_cell_exclusion_labels(experiment_id):
    warn_str = gen_depr_str.format('from_lims',
                                   'get_cell_exclusion_labels')
    warnings.warn(warn_str)

    mixin = lims_engine
    query = '''
    SELECT oe.id AS oe_id, cr.id AS cr_id , rel.name AS excl_label
    FROM ophys_experiments oe
    JOIN ophys_cell_segmentation_runs ocsr ON ocsr.ophys_experiment_id=oe.id
    AND ocsr.current = 't'
    JOIN cell_rois cr ON cr.ophys_cell_segmentation_run_id=ocsr.id
    JOIN cell_rois_roi_exclusion_labels crrel ON crrel.cell_roi_id=cr.id
    JOIN roi_exclusion_labels rel ON rel.id=crrel.roi_exclusion_label_id
    WHERE oe.id = {} '''.format(experiment_id)
    return mixin.select(query)


def get_lims_cell_rois_table(ophys_experiment_id):
    """Queries LIMS via AllenSDK PostgresQuery function to retrieve
        everything in the cell_rois table for a given experiment

    Arguments:
        experiment_id {int} -- 9 digit unique identifier for an ophys experiment

    Returns:
        dataframe -- returns dataframe with the following columns:
            id: a temporary id assigned before cell matching has occured. Same as cell_roi_id in
                the objectlist.txt file

            cell_specimen_id: a permanent id that is assigned after cell matching has occured.
                            this id can be found in multiple experiments in a container if a
                            cell is matched across experiments.
                            experiments that fail qc are not assigned cell_specimen_id s
            ophys_experiment_id:
            x: roi bounding box min x or "bbox_min_x" in objectlist.txt file
            y: roi bounding box min y or "bbox_min_y" in objectlist.txt file
            width:
            height:
            valid_roi: boolean(true/false), whether the roi passes or fails roi filtering
            mask_matrix: boolean mask of just roi
            max_correction_up:
            max_correction_down:
            max_correction_right:
            max_correction_left:
            mask_image_plane:
            ophys_cell_segmentation_run_id:

    """
    # query from AllenSDK
    warn_str = gen_depr_str.format('from_lims',
                                   'get_cell_segmentation_runs_table')
    warnings.warn(warn_str)

    mixin = lims_engine
    query = '''
    select cell_rois.*
    FROM

    ophys_experiments oe
    JOIN cell_rois on oe.id = cell_rois.ophys_experiment_id

    WHERE oe.id = {}'''.format(ophys_experiment_id)
    lims_cell_rois_table = mixin.select(query)
    return lims_cell_rois_table


def roi_locations_from_cell_rois_table(experiment_id):
    """takes the lims_cell_rois_table and pares it down to just what's relevent to join with the objectlist.txt table.
       Renames columns to maintain continuity between the tables.

    Arguments:
        lims_cell_rois_table {dataframe} -- dataframe from LIMS with roi location information

    Returns:
        dataframe -- pared down dataframe
    """
    import visual_behavior.ophys.io.convert_level_1_to_level_2 as convert
    # get_lims_data = convert.get_lims_data
    # get the cell_rois_table for that experiment
    lims_data = convert.get_lims_data(experiment_id)
    exp_cell_rois_table = get_lims_cell_rois_table(lims_data['lims_id'].values[0])

    # select only the relevent columns
    # roi_locations = exp_cell_rois_table[["id", "x", "y", "width", "height", "valid_roi", "mask_matrix"]]
    roi_locations = exp_cell_rois_table
    # rename columns
    roi_locations = clean_roi_locations_column_labels(roi_locations)
    return roi_locations


###########

def get_failed_roi_exclusion_labels(experiment_id):
    """Queries LIMS  roi_exclusion_labels table via AllenSDK PostgresQuery function to retrieve and build a
         table of all failed ROIS for a particular experiment, and their exclusion labels.

         Failed rois will be listed multiple times/in multiple rows depending upon how many exclusion
         labels they have.

    Arguments:
        experiment_id {int} -- [9 digit unique identifier for the experiment]

    Returns:
        dataframe -- returns a dataframe with the following columns:
            ophys_experiment_id: 9 digit unique identifier for the experiment
            cell_roi_id:unique identifier for each roi (created after segmentation, before cell matching)
            cell_specimen_id: unique identifier for each roi (created after cell matching, so could be blank for some experiments/rois depending on processing step)
            valid_roi: boolean true/false, should be false for all entries, since dataframe should only list failed/invalid rois
            exclusion_label_name: label/tag for why the roi was deemed invalid
    """
    warn_str = gen_depr_str.format('from_lims',
                                   'get_cell_exclusion_labels')
    warnings.warn(warn_str)

    # query from AllenSDK
    experiment_id = int(experiment_id)
    mixin = lims_engine
    # build query
    query = '''
    select

    oe.id as ophys_experiment_id,
    cell_rois.id as cell_roi_id,
    cell_rois.valid_roi,
    cell_rois.cell_specimen_id,
    el.name as exclusion_label_name

    from

    ophys_experiments oe
    join cell_rois on oe.id = cell_rois.ophys_experiment_id
    join cell_rois_roi_exclusion_labels crel on crel.cell_roi_id = cell_rois.id
    join roi_exclusion_labels el on el.id = crel.roi_exclusion_label_id

    where oe.id = {}'''.format(experiment_id)

    failed_roi_exclusion_labels = mixin.select(query)
    return failed_roi_exclusion_labels


def gen_roi_exclusion_labels_lists(experiment_id):
    """[summary]

    Arguments:
        experiment_id {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    roi_exclusion_table = get_failed_roi_exclusion_labels(experiment_id)
    roi_exclusion_table = roi_exclusion_table[["cell_roi_id", "exclusion_label_name"]]
    exclusion_list_per_invalid_roi = roi_exclusion_table.groupby(["cell_roi_id"]).agg(lambda x: tuple(x)).applymap(
        list).reset_index()
    return exclusion_list_per_invalid_roi


def clean_roi_locations_column_labels(roi_locations_dataframe):
    """takes some column labels from the roi_locations dataframe and renames them to be more explicit and descriptive, and to match the column labels
        from the objectlist dataframe.

    Arguments:
        roi_locations_dataframe {dataframe} -- dataframe with roi id and location information

    Returns:
        dataframe -- [description]
    """
    warn_str = gen_depr_str.format('from_lims',
                                   'get_cell_rois_table')
    warnings.warn(warn_str)
    roi_locations_dataframe = roi_locations_dataframe.rename(columns={"id": "cell_roi_id",
                                                                      "mask_matrix": "roi_mask",
                                                                      "x": "bbox_min_x",
                                                                      "y": "bbox_min_y"})
    return roi_locations_dataframe


def get_objectlisttxt_location(segmentation_run_id):
    """use SQL and the LIMS well known file system to get the location information for the objectlist.txt file
        for a given cell segmentation run

    Arguments:
        segmentation_run_id {int} -- 9 digit segmentation run id

    Returns:
        list -- list with storage directory and filename
    """
    warn_str = gen_depr_str.format('from_lims',
                                   'get_segmentation_objects_filepath')
    warnings.warn(warn_str)

    QUERY = '''
    SELECT wkf.storage_directory, wkf.filename
    FROM well_known_files wkf
    JOIN well_known_file_types wkft on wkf.well_known_file_type_id = wkft.id
    JOIN ophys_cell_segmentation_runs ocsr on wkf.attachable_id = ocsr.id
    WHERE wkft.name = 'OphysSegmentationObjects'
    AND wkf.attachable_type = 'OphysCellSegmentationRun'
    AND ocsr.id = {0}
    '''
    lims_cursor = db.get_psql_dict_cursor()
    lims_cursor.execute(QUERY.format(segmentation_run_id))
    objecttxt_info = (lims_cursor.fetchall())
    return objecttxt_info


def load_current_objectlisttxt_file(experiment_id):
    """loads the objectlist.txt file for the current segmentation run, then "cleans" the column names and returns a dataframe

    Arguments:
        experiment_id {[int]} -- 9 digit unique identifier for the experiment

    Returns:
        dataframe -- dataframe with the following columns: (from http://confluence.corp.alleninstitute.org/display/IT/Ophys+Segmentation)
            trace_index:The index to the corresponding trace plot computed  (order of the computed traces in file _somaneuropiltraces.h5)
            center_x: The x coordinate of the centroid of the object
                      in image pixels
            center_y: The y coordinate of the centroid of the object
                      in image pixels
            frame_of_max_intensity_masks_file: The frame the object
                                               mask is in maxInt_masks2.tif
            frame_of_enhanced_movie: The frame in the movie enhimgseq.tif
                                      that best shows the object
            layer_of_max_intensity_file: The layer of the maxInt file
                                         where the object can be seen
            bbox_min_x: coordinates delineating a bounding box that
                        contains the object, in image pixels (upper left corner)
            bbox_min_y: coordinates delineating a bounding box that
                        contains the object, in image pixels (upper left corner)
            bbox_max_x: coordinates delineating a bounding box that
                        contains the object, in image pixels (bottom right corner)
            bbox_max_y: coordinates delineating a bounding box that
                        contains the object, in image pixels (bottom right corner)
            area: Total area of the segmented object
            ellipseness: The "ellipticalness" of the object, i.e.
                        length of long axis divided by length of short axis
            compactness: Compactness :  perimeter^2 divided by area
            exclude_code: A non-zero value indicates the object should be excluded from further analysis.
                        Based on measurements in objectlist.txt
                        0 = not excluded
                        1 = doublet cell
                        2 = boundary cell
                        Others = classified as not complete soma, apical dendrite, ....
            mean_intensity: Correlates with delta F/F.  Mean brightness of the object
            mean_enhanced_intensity: Mean enhanced brightness of the object
            max_intensity: Max brightness of the object
            max_enhanced_intensity: Max enhanced brightness of the object
            intensity_ratio: (max_enhanced_intensity - mean_enhanced_intensity) / mean_enhanced_intensity, for detecting dendrite objects
            soma_minus_np_mean: mean of (soma trace – its neuropil trace)
            soma_minus_np_std: 1-sided stdv of (soma trace – its neuropil trace)
            sig_active_frames_2_5:# frames with significant detected activity (spiking):
                                  Sum ( soma_trace > (np_trace + Snpoffsetmean+ 2.5 * Snpoffsetstdv)
            sig_active_frames_4: # frames with significant detected activity (spiking)
                                 Sum ( soma_trace > (np_trace + Snpoffsetmean+ 4.0 * Snpoffsetstdv)
            overlap_count: 	Number of other objects the object overlaps with
            percent_area_overlap: the percentage of total object area that overlaps with other objects
            overlap_obj0_index: The index of the first object with which this object overlaps
            overlap_obj1_index: The index of the second object with which this object overlaps
            soma_obj0_overlap_trace_corr: trace correlation coefficient between soma and overlap soma0
                                          (-1.0:  excluded cell,  0.0 : NA)
            soma_obj1_overlap_trace_corr: trace correlation coefficient between soma and overlap soma1
    """
    warn_str = gen_depr_str.format('from_lims',
                                   'load_objectlist')
    warnings.warn(warn_str)

    current_segmentation_run_id = get_current_segmentation_run_id(experiment_id)
    objectlist_location_info = get_objectlisttxt_location(current_segmentation_run_id)
    objectlist_path = objectlist_location_info[0]['storage_directory']
    objectlist_file = objectlist_location_info[0]["filename"]
    full_name = os.path.join(objectlist_path, objectlist_file).replace('/allen',
                                                                       '//allen')  # works with windows and linux filepaths
    objectlist_dataframe = pd.read_csv(full_name)
    objectlist_dataframe = clean_objectlist_col_labels(objectlist_dataframe)  # "clean" columns names to be more meaningful
    return objectlist_dataframe


def clean_objectlist_col_labels(objectlist_dataframe):
    """take the roi metrics from the objectlist.txt file and renames them to be more explicit and descriptive.
        -removes single blank space at the beginning of column names
        -enforced naming scheme(no capitolization, added _)
        -renamed columns to be more descriptive/reflect contents of column

    Arguments:
        objectlist_dataframe {pandas dataframe} -- [roi metrics dataframe or dataframe generated from the objectlist.txt file]

    Returns:
        [pandas dataframe] -- [same dataframe with same information but with more informative column names
    """
    warn_str = gen_depr_str.format('from_lims_utilities',
                                   'update_objectlist_column_labels')
    warnings.warn(warn_str)

    objectlist_dataframe = objectlist_dataframe.rename(index=str, columns={' traceindex': "trace_index",
                                                                           ' cx': 'center_x',
                                                                           ' cy': 'center_y',
                                                                           ' mask2Frame': 'frame_of_max_intensity_masks_file',
                                                                           ' frame': 'frame_of_enhanced_movie',
                                                                           ' object': 'layer_of_max_intensity_file',
                                                                           ' minx': 'bbox_min_x',
                                                                           ' miny': 'bbox_min_y',
                                                                           ' maxx': 'bbox_max_x',
                                                                           ' maxy': 'bbox_max_y',
                                                                           ' area': 'area',
                                                                           ' shape0': 'ellipseness',
                                                                           ' shape1': "compactness",
                                                                           ' eXcluded': "exclude_code",
                                                                           ' meanInt0': "mean_intensity",
                                                                           ' meanInt1': "mean_enhanced_intensity",
                                                                           ' maxInt0': "max_intensity",
                                                                           ' maxInt1': "max_enhanced_intensity",
                                                                           ' maxMeanRatio': "intensity_ratio",
                                                                           ' snpoffsetmean': "soma_minus_np_mean",
                                                                           ' snpoffsetstdv': "soma_minus_np_std",
                                                                           ' act2': "sig_active_frames_2_5",
                                                                           ' act3': "sig_active_frames_4",
                                                                           ' OvlpCount': "overlap_count",
                                                                           ' OvlpAreaPer': "percent_area_overlap",
                                                                           ' OvlpObj0': "overlap_obj0_index",
                                                                           ' OvlpObj1': "overlap_obj1_index",
                                                                           ' corcoef0': "soma_obj0_overlap_trace_corr",
                                                                           ' corcoef1': "soma_obj1_overlap_trace_corr"})
    return objectlist_dataframe


def get_average_depth_image(experiment_id):
    """
    quick and dirty function to load 16x depth image from lims
    file path location depends on whether it is scientifica or mesoscope, and which version of the pipeline was run
    function iterates through all possible options of file locations
    """
    import visual_behavior.data_access.utilities as utilities
    import matplotlib.pyplot as plt

    expt_dir = utilities.get_ophys_experiment_dir(utilities.get_lims_data(experiment_id))
    session_dir = utilities.get_ophys_session_dir(utilities.get_lims_data(experiment_id))
    experiment_table = get_filtered_ophys_experiment_table(include_failed_data=True)
    session_id = experiment_table.loc[experiment_id].ophys_session_id

    # try all combinations of potential file path locations...
    if os.path.isfile(os.path.join(session_dir, str(experiment_id) + '_averaged_depth.tif')):
        im = plt.imread(os.path.join(session_dir, str(experiment_id) + '_averaged_depth.tif'))
    elif os.path.isfile(os.path.join(session_dir, str(experiment_id) + '_depth.tif')):
        im = plt.imread(os.path.join(session_dir, str(experiment_id) + '_depth.tif'))
    elif os.path.isfile(os.path.join(session_dir, str(session_id) + '_averaged_depth.tif')):
        im = plt.imread(os.path.join(session_dir, str(session_id) + '_averaged_depth.tif'))
    elif os.path.isfile(os.path.join(expt_dir, str(experiment_id) + '_averaged_depth.tif')):
        im = plt.imread(os.path.join(expt_dir, str(experiment_id) + '_averaged_depth.tif'))
    elif os.path.isfile(os.path.join(expt_dir, str(experiment_id) + '_depth.tif')):
        im = plt.imread(os.path.join(expt_dir, str(experiment_id) + '_depth.tif'))
    else:
        print('problem for', experiment_id)
        print(session_dir)
    return im

# CONTAINER  LEVEL


def get_lims_container_info(ophys_container_id):
    """"uses an sqlite query to retrieve container level information
        from the lims2 database. Each row is an experiment within the container.

    Arguments:
        ophys_container_id {[type]} -- [description]

    Returns:
       table -- table with the following columns:
                    "ophys_container_id":
                    "container_workflow_state":
                    "ophys_experiment_id":
                    "ophys_session_id":
                    "stage_name_lims":
                    "foraging_id":
                    "experiment_workflow_state":
                    "mouse_info":
                    "mouse_donor_id":
                    "targeted_structure":
                    "depth":
                    "rig":
                    "date_of_acquisition":
    """
    warn_str = gen_depr_str.format('from_lims',
                                   'get_general_info_for_ophys_container_id')
    warnings.warn(warn_str)

    ophys_container_id = int(ophys_container_id)

    mixin = lims_engine
    # build query
    query = '''
    SELECT
    container.visual_behavior_experiment_container_id as ophys_container_id,
    vbec.workflow_state as container_workflow_state,
    oe.id as ophys_experiment_id,
    oe.ophys_session_id,
    os.stimulus_name as stage_name_lims,
    os.foraging_id,
    oe.workflow_state as experiment_workflow_state,
    specimens.name as mouse_info,
    specimens.donor_id as mouse_donor_id,
    structures.acronym as targeted_structure,
    imaging_depths.depth,
    equipment.name as rig,
    os.date_of_acquisition

    FROM
    ophys_experiments_visual_behavior_experiment_containers container
    join visual_behavior_experiment_containers vbec on vbec.id = container.visual_behavior_experiment_container_id
    join ophys_experiments oe on oe.id = container.ophys_experiment_id
    join ophys_sessions os on os.id = oe.ophys_session_id
    join structures on structures.id = oe.targeted_structure_id
    join imaging_depths on imaging_depths.id = oe.imaging_depth_id
    join specimens on specimens.id = os.specimen_id
    join equipment on equipment.id = os.equipment_id

    where
    container.visual_behavior_experiment_container_id ={}'''.format(ophys_container_id)

    lims_container_info = mixin.select(query)
    return lims_container_info


# FROM LIMS WELL KNOWN FILES

def get_release_behavior_nwb_file_path(behavior_session_id):
    """"    --LIMS SQL to get behavior Ophys NWB files that are ready now
    """
    mixin = lims_engine
    # build query
    query = '''
    SELECT wkfb.storage_directory || wkfb.filename
    FROM behavior_sessions bs
    JOIN well_known_files wkfb ON wkfb.attachable_id=bs.id AND wkfb.attachable_type = 'BehaviorSession' AND wkfb.well_known_file_type_id = (SELECT id FROM well_known_file_types WHERE name = 'BehaviorNwb')
    WHERE bs.id = {0}
    '''.format(behavior_session_id)
    file_path = mixin.select(query)
    return file_path


def get_release_ophys_nwb_file_paths():
    """"    --LIMS SQL to get behavior Ophys NWB files that are ready now
    """
    mixin = lims_engine
    # build query
    query = '''
    SELECT wkf.storage_directory || wkf.filename AS nwb_file
    FROM ophys_experiments oe
    JOIN ophys_sessions os ON os.id=oe.ophys_session_id JOIN equipment e ON e.id=os.equipment_id
    JOIN jobs j ON j.enqueued_object_id=oe.id AND j.archived = 'f' AND j.job_state_id = 3 --IN (1,2,4,5)
    JOIN job_queues jq ON jq.id=j.job_queue_id AND jq.name = 'BEHAVIOR_OPHYS_WRITE_NWB_QUEUE_NO_STATE_CHANGE'
    JOIN well_known_files wkf ON wkf.attachable_id=oe.id JOIN well_known_file_types wkft ON wkft.id=wkf.well_known_file_type_id AND wkft.name = 'BehaviorOphysNwb'
    JOIN ophys_experiments_visual_behavior_experiment_containers oevbec ON oevbec.ophys_experiment_id=oe.id
    JOIN visual_behavior_experiment_containers vbec ON vbec.id=oevbec.visual_behavior_experiment_container_id
    WHERE oe.workflow_state = 'passed' AND vbec.workflow_state = 'published'
    '''
    file_paths = mixin.select(query)
    return file_paths


def get_release_behavior_nwb_file_paths():
    """"    --LIMS SQL to get behavior-only NWB files that are ready now

    """
    warn_str = gen_depr_str.format('from_lims',
                                   'get_BehaviorOphysNWB_filepath')
    warnings.warn(warn_str)

    mixin = lims_engine
    # build query
    query = '''
    SELECT wkf.storage_directory || wkf.filename
    FROM behavior_sessions bs JOIN donors d ON d.id=bs.donor_id
    LEFT JOIN ophys_sessions os ON os.id=bs.ophys_session_id AND os.stimulus_name NOT IN ('OPHYS_7_receptive_field_mapping')
    LEFT JOIN equipment e ON e.id=bs.equipment_id
    LEFT JOIN jobs j ON j.job_queue_id = 1071407587 AND j.archived = 'f' AND j.enqueued_object_id=bs.id AND j.job_state_id = 3
    JOIN well_known_files wkf ON wkf.attachable_id=bs.id
    JOIN well_known_file_types wkft ON wkft.id=wkf.well_known_file_type_id AND wkft.name = 'BehaviorNwb'
    WHERE donor_id IN
    (
      SELECT sp.donor_id
      FROM ophys_experiments_visual_behavior_experiment_containers oevbec
      JOIN ophys_experiments oe ON oe.id=oevbec.ophys_experiment_id
      JOIN visual_behavior_experiment_containers vbec ON vbec.id=oevbec.visual_behavior_experiment_container_id
      JOIN specimens sp ON sp.id=vbec.specimen_id
      WHERE vbec.workflow_state = 'published' AND oe.workflow_state = 'passed'
    )
    -- minus sessions from 1171 behavior ophys experiments
    AND
    (
      bs.ophys_session_id IS NULL OR (
        bs.ophys_session_id NOT IN (
          SELECT oe.ophys_session_id
          FROM ophys_experiments_visual_behavior_experiment_containers oevbec
          JOIN ophys_experiments oe ON oe.id=oevbec.ophys_experiment_id
          JOIN visual_behavior_experiment_containers vbec ON vbec.id=oevbec.visual_behavior_experiment_container_id
          JOIN specimens sp ON sp.id=vbec.specimen_id
          WHERE vbec.workflow_state = 'published' AND oe.workflow_state = 'passed'
        )
        AND bs.ophys_session_id NOT IN (
          SELECT os.id
          FROM ophys_sessions os
          WHERE os.stimulus_name IN ('OPHYS_7_receptive_field_mapping')
        )
      )
    )
    ORDER BY 1;
    Collapse
    '''
    file_paths = mixin.select(query)
    return file_paths


def get_timeseries_ini_wkf_info(ophys_session_id):
    """use SQL and the LIMS well known file system to get the
    timeseries_XYT.ini file for a given ophys session
    *from a Scientifica rig*

    Arguments:
        ophys_session_id {int} -- 9 digit ophys session id

    Returns:

    """
    warn_str = gen_depr_str.format('from_lims',
                                   'get_timeseries_ini_filepath')
    warnings.warn(warn_str)

    QUERY = '''
    SELECT wkf.storage_directory || wkf.filename
    FROM well_known_files wkf
    JOIN well_known_file_types wkft ON wkft.id=wkf.well_known_file_type_id
    JOIN specimens sp ON sp.id=wkf.attachable_id
    JOIN ophys_sessions os ON os.specimen_id=sp.id
    WHERE wkft.name = 'SciVivoMetadata'
    AND wkf.storage_directory LIKE '%ophys_session_{0}%'
    AND os.id = {0}

    '''.format(ophys_session_id)

    lims_cursor = db.get_psql_dict_cursor()
    lims_cursor.execute(QUERY)

    timeseries_ini_wkf_info = (lims_cursor.fetchall())
    return timeseries_ini_wkf_info


def get_timeseries_ini_location(ophys_session_id):
    """use SQL and the LIMS well known file system to
        get info for the timeseries_XYT.ini file for a
        given ophys session, and then parses that information
        to get the filepath

    Arguments:
        ophys_session_id {int} -- 9 digit ophys session id

    Returns:
        filepath -- [description]
    """
    warn_str = gen_depr_str.format('from_lims',
                                   'get_timeseries_ini_filepath')
    warnings.warn(warn_str)

    timeseries_ini_wkf_info = get_timeseries_ini_wkf_info(ophys_session_id)
    timeseries_ini_path = timeseries_ini_wkf_info[0]['?column?']  # idk why it's ?column? but it is :(
    timeseries_ini_path = timeseries_ini_path.replace('/allen', '//allen')  # works with windows and linux filepaths
    return timeseries_ini_path


def pmt_gain_from_timeseries_ini(timeseries_ini_path):
    """parses the timeseries ini file (scientifica experiments only)
        and extracts the pmt gain setting

    Arguments:
        timeseries_ini_path {[type]} -- [description]

    Returns:
        int -- int of the pmt gain
    """
    config = configp.ConfigParser()

    config.read(timeseries_ini_path)
    pmt_gain = int(float(config['_']['PMT.2']))
    return pmt_gain


def get_pmt_gain_for_session(ophys_session_id):
    """finds the timeseries ini file for a given ophys session
        on a Scientifica rig, parses the file and returns the
        pmt gain setting for that session

    Arguments:
        ophys_session_id {int} -- [description]

    Returns:
        int -- pmt gain setting
    """
    try:
        timeseries_ini_path = get_timeseries_ini_location(ophys_session_id)
        pmt_gain = pmt_gain_from_timeseries_ini(timeseries_ini_path)
    except IndexError:
        ophys_experiment_id = get_ophys_experiment_id_for_ophys_session_id(ophys_session_id)
        print("lims query did not return timeseries_XYT.ini location for session_id: " + str(
            ophys_session_id) + ", experiment_id: " + str(ophys_experiment_id))
        pmt_gain = np.nan
    return pmt_gain


def get_pmt_gain_for_experiment(ophys_experiment_id):
    """finds the timeseries ini file for  the ophys_session_id
        associated with an ophys_experiment_id  from a Scientifica
        rig, parses the file and returns the
        pmt gain setting for that session

    Arguments:
        ophys_experiment_id {[type]} -- [description]

    Returns:
        int -- pmt gain setting
    """
    ophys_session_id = get_ophys_session_id_for_ophys_experiment_id(ophys_experiment_id)
    pmt_gain = get_pmt_gain_for_session(ophys_session_id)
    return pmt_gain


def get_wkf_dff_h5_location(ophys_experiment_id):
    """uses well known file system to query lims
        and get the directory and filename for the
        dff traces h5 for a given ophys experiment

    Arguments:
        ophys_experiment_id {int} -- 9 digit unique identifier for
                                    an ophys experiment

    Returns:
        string -- filepath (directory and filename) for the dff.h5 file
                    for the given ophys_experiment_id
    """
    warn_str = gen_depr_str.format('from_lims',
                                   'get_dff_traces_filepath')
    warnings.warn(warn_str)

    QUERY = '''
    SELECT storage_directory || filename
    FROM well_known_files
    WHERE well_known_file_type_id = 514173073 AND
    attachable_id = {0}

    '''.format(ophys_experiment_id)

    lims_cursor = db.get_psql_dict_cursor()
    lims_cursor.execute(QUERY)

    dff_h5_location_info = (lims_cursor.fetchall())

    dff_h5_path = dff_h5_location_info[0]['?column?']  # idk why it's ?column? but it is :(
    dff_h5_path = dff_h5_path.replace('/allen', '//allen')  # works with windows and linux filepaths
    return dff_h5_path


def get_wkf_roi_trace_h5_location(ophys_experiment_id):
    """uses well known file system to query lims
        and get the directory and filename for the
        roi_traces.h5 for a given ophys experiment

    Arguments:
        ophys_experiment_id {int} -- 9 digit unique identifier for
                                    an ophys experiment

    Returns:
        string -- filepath (directory and filename) for the roi_traces.h5 file
                    for the given ophys_experiment_id
    """
    warn_str = gen_depr_str.format('from_lims',
                                   'get_roi_traces_filepath')
    warnings.warn(warn_str)

    QUERY = '''
    SELECT storage_directory || filename
    FROM well_known_files
    WHERE well_known_file_type_id = 514173076 AND
    attachable_id = {0}

    '''.format(ophys_experiment_id)

    lims_cursor = db.get_psql_dict_cursor()
    lims_cursor.execute(QUERY)

    trace_h5_location_info = (lims_cursor.fetchall())

    trace_h5_path = trace_h5_location_info[0]['?column?']  # idk why it's ?column? but it is :(
    trace_h5_path = trace_h5_path.replace('/allen', '//allen')  # works with windows and linux filepaths
    return trace_h5_path


def get_roi_traces_array(ophys_experiment_id):
    """use SQL and the LIMS well known file system to find and load
            "roi_traces.h5" then return the traces as an array

        Arguments:
            ophys_experiment_id {int} -- 9 digit ophys experiment ID

        Returns:
            raw_traces_array -- mxn array where m = rois and n = time
        """
    filepath = get_wkf_roi_trace_h5_location(ophys_experiment_id)
    f = h5py.File(filepath, 'r')
    roi_traces_array = np.asarray(f['data'])
    f.close()
    return roi_traces_array


def get_wkf_neuropil_trace_h5_location(ophys_experiment_id):
    """uses well known file system to query lims
        and get the directory and filename for the
        neuropil_traces.h5 for a given ophys experiment

    Arguments:
        ophys_experiment_id {int} -- 9 digit unique identifier for
                                    an ophys experiment

    Returns:
        string -- filepath (directory and filename) for the neuropil_traces.h5 file
                    for the given ophys_experiment_id
    """
    warn_str = gen_depr_str.format('from_lims',
                                   'get_neuropil_traces_filepath')
    warnings.warn(warn_str)

    QUERY = '''
    SELECT storage_directory || filename
    FROM well_known_files
    WHERE well_known_file_type_id = 514173078 AND
    attachable_id = {0}

    '''.format(ophys_experiment_id)

    lims_cursor = db.get_psql_dict_cursor()
    lims_cursor.execute(QUERY)

    trace_h5_location_info = (lims_cursor.fetchall())

    trace_h5_path = trace_h5_location_info[0]['?column?']  # idk why it's ?column? but it is :(
    trace_h5_path = trace_h5_path.replace('/allen', '//allen')  # works with windows and linux filepaths
    return trace_h5_path


def get_neuropil_traces_array(ophys_experiment_id):
    """use SQL and the LIMS well known file system to find and load
            "neuropil_traces.h5" then return the traces as an array

        Arguments:
            ophys_experiment_id {int} -- 9 digit ophys experiment ID

        Returns:
            neuropil_traces_array -- mxn array where m = rois and n = time
        """
    warn_str = gen_depr_str.format('from_lims',
                                   'load_neuropil_traces_array')
    warnings.warn(warn_str)

    filepath = get_wkf_neuropil_trace_h5_location(ophys_experiment_id)
    f = h5py.File(filepath, 'r')
    neuropil_traces_array = np.asarray(f['data'])
    f.close()
    return neuropil_traces_array


def get_wkf_extracted_trace_h5_location(ophys_experiment_id):
    """uses well known file system to query lims
        and get the directory and filename for the
        neuropil_traces.h5 for a given ophys experiment

    Arguments:
        ophys_experiment_id {int} -- 9 digit unique identifier for
                                    an ophys experiment

    Returns:
        string -- filepath (directory and filename) for the neuropil_traces.h5 file
                    for the given ophys_experiment_id
    """
    warn_str = gen_depr_str.format('from_lims',
                                   'get_extracted_traces_input_filepath')
    warnings.warn(warn_str)

    QUERY = '''
    SELECT storage_directory || filename
    FROM well_known_files
    WHERE well_known_file_type_id = 486797213 AND
    attachable_id = {0}

    '''.format(ophys_experiment_id)

    lims_cursor = db.get_psql_dict_cursor()
    lims_cursor.execute(QUERY)

    trace_h5_location_info = (lims_cursor.fetchall())

    trace_h5_path = trace_h5_location_info[0]['?column?']  # idk why it's ?column? but it is :(
    trace_h5_path = trace_h5_path.replace('/allen', '//allen')  # works with windows and linux filepaths
    return trace_h5_path


def get_wkf_demixed_traces_h5_location(ophys_experiment_id):
    """uses well known file system to query lims
        and get the directory and filename for the
        roi_traces.h5 for a given ophys experiment

    Arguments:
        ophys_experiment_id {int} -- 9 digit unique identifier for
                                    an ophys experiment

    Returns:
        string -- filepath (directory and filename) for the roi_traces.h5 file
                    for the given ophys_experiment_id
    """
    warn_str = gen_depr_str.format('from_lims',
                                   'get_demixed_traces_filepath')
    warnings.warn(warn_str)

    QUERY = '''
    SELECT storage_directory || filename
    FROM well_known_files
    WHERE well_known_file_type_id = 820011707 AND
    attachable_id = {0}

    '''.format(ophys_experiment_id)

    lims_cursor = db.get_psql_dict_cursor()
    lims_cursor.execute(QUERY)

    trace_h5_location_info = (lims_cursor.fetchall())

    trace_h5_path = trace_h5_location_info[0]['?column?']  # idk why it's ?column? but it is :(
    trace_h5_path = trace_h5_path.replace('/allen', '//allen')  # works with windows and linux filepaths
    return trace_h5_path


def get_demixed_traces_array(ophys_experiment_id):
    """use SQL and the LIMS well known file system to find and load
            "demixed_traces.h5" then return the traces as an array

        Arguments:
            ophys_experiment_id {int} -- 9 digit ophys experiment ID

        Returns:
            demixed_traces_array -- mxn array where m = rois and n = time
    """
    warn_str = gen_depr_str.format('from_lims',
                                   'load_demixed_traces_array')
    warnings.warn(warn_str)

    filepath = get_wkf_demixed_traces_h5_location(ophys_experiment_id)
    f = h5py.File(filepath, 'r')
    demixed_traces_array = np.asarray(f['data'])
    f.close()
    return demixed_traces_array


def get_motion_corrected_movie_h5_wkf_info(ophys_experiment_id):
    """use SQL and the LIMS well known file system to get the
        "motion_corrected_movie.h5" information for a given
        ophys_experiment_id

    Arguments:
        ophys_experiment_id {int} -- 9 digit ophys experiment ID

    Returns:
        [type] -- [description]
    """
    warn_str = gen_depr_str.format('from_lims',
                                   'get_motion_corrected_movie_filepath')
    warnings.warn(warn_str)

    QUERY = '''
     SELECT storage_directory || filename
     FROM well_known_files
     WHERE well_known_file_type_id = 886523092 AND
     attachable_id = {0}

    '''.format(ophys_experiment_id)

    lims_cursor = db.get_psql_dict_cursor()
    lims_cursor.execute(QUERY)

    motion_corrected_movie_h5_wkf_info = (lims_cursor.fetchall())
    return motion_corrected_movie_h5_wkf_info


def get_motion_corrected_movie_h5_location(ophys_experiment_id):
    """use SQL and the LIMS well known file system to get info for the
        "motion_corrected_movie.h5" file for a ophys_experiment_id,
        and then parses that information to get the filepath

    Arguments:
        ophys_experiment_id {int} -- 9 digit ophys experiment ID

    Returns:
        filepath -- [description]
    """
    warn_str = gen_depr_str.format('from_lims',
                                   'get_motion_xy_offset_filepath')
    warnings.warn(warn_str)

    motion_corrected_movie_h5_wkf_info = get_motion_corrected_movie_h5_wkf_info(ophys_experiment_id)
    motion_corrected_movie_h5_path = motion_corrected_movie_h5_wkf_info[0][
        '?column?']  # idk why it's ?column? but it is :(
    motion_corrected_movie_h5_path = motion_corrected_movie_h5_path.replace('/allen',
                                                                            '//allen')  # works with windows and linux filepaths
    return motion_corrected_movie_h5_path


def load_motion_corrected_movie(ophys_experiment_id):
    """uses well known file system to get motion_corrected_movie.h5
        filepath and then loads the h5 file with h5py function.
        Gets the motion corrected movie array in the h5 from the only
        datastream/key 'data' and returns it.

    Arguments:
        ophys_experiment_id {int} -- 9 digit ophys experiment ID

    Returns:
        HDF5 dataset -- 3d array-like  (z, y, x) dimensions
                        z: timeseries/frame number
                        y: single frame y axis
                        x: single frame x axis
    """
    warn_str = gen_depr_str.format('from_lims',
                                   'load_motion_corrected_movie')
    warnings.warn(warn_str)

    motion_corrected_movie_h5_path = get_motion_corrected_movie_h5_location(ophys_experiment_id)
    motion_corrected_movie_h5 = h5py.File(motion_corrected_movie_h5_path, 'r')
    motion_corrected_movie = motion_corrected_movie_h5['data']

    return motion_corrected_movie


def get_rigid_motion_transform_csv_wkf_info(ophys_experiment_id):
    """use SQL and the LIMS well known file system to get the
        "rigid_motion_transform.csv" information for a given
        ophys_experiment_id

    Arguments:
        ophys_experiment_id {int} -- 9 digit ophys experiment ID

    Returns:
        [type] -- [description]
    """
    warn_str = gen_depr_str.format('from_lims',
                                   'load_rigid_motion_transform')
    warnings.warn(warn_str)

    QUERY = '''
     SELECT storage_directory || filename
     FROM well_known_files
     WHERE well_known_file_type_id = 514167000 AND
     attachable_id = {0}

    '''.format(ophys_experiment_id)

    lims_cursor = db.get_psql_dict_cursor()
    lims_cursor.execute(QUERY)

    rigid_motion_transform_csv_wkf_info = (lims_cursor.fetchall())
    return rigid_motion_transform_csv_wkf_info


def get_rigid_motion_transform_csv_location(ophys_experiment_id):
    """use SQL and the LIMS well known file system to get info for the
        rigid_motion_transform.csv" file for a ophys_experiment_id,
        and then parses that information to get the filepath

    Arguments:
        ophys_experiment_id {int} -- 9 digit ophys experiment ID

    Returns:
        filepath -- [description]
    """
    warn_str = gen_depr_str.format('from_lims',
                                   'get_motion_xy_offset_filepath')
    warnings.warn(warn_str)

    rigid_motion_transform_csv_wkf_info = get_rigid_motion_transform_csv_wkf_info(ophys_experiment_id)
    rigid_motion_transform_csv_path = rigid_motion_transform_csv_wkf_info[0][
        '?column?']  # idk why it's ?column? but it is :(
    rigid_motion_transform_csv_path = rigid_motion_transform_csv_path.replace('/allen',
                                                                              '//allen')  # works with windows and linux filepaths
    return rigid_motion_transform_csv_path


def load_rigid_motion_transform_csv(ophys_experiment_id):
    """use SQL and the LIMS well known file system to locate
        and load the rigid_motion_transform.csv file for
        a given ophys_experiment_id

    Arguments:
        ophys_experiment_id {int} -- 9 digit ophys experiment ID

    Returns:
        dataframe -- dataframe with the following columns:
                        "framenumber":
                                  "x":
                                  "y":
                        "correlation":
                           "kalman_x":
                           "kalman_y":
    """
    warn_str = gen_depr_str.format('from_lims',
                                   'load_rigid_motion_transform')
    warnings.warn(warn_str)
    rigid_motion_transform_csv_path = get_rigid_motion_transform_csv_location(ophys_experiment_id)
    rigid_motion_transform_df = pd.read_csv(rigid_motion_transform_csv_path)
    return rigid_motion_transform_df


# CONTAINER LEVEL INFO

def get_unique_cell_specimen_ids_for_container(container_id):
    """
    Retrieves and concatenates the cell_specimen_table for all experiments within a container,
    then returns a list of unique cell_specimen_ids for the container.
    :param container_id: container ID
    :return: list of cell_specimen_ids for a given container
    """
    experiments_table = get_filtered_ophys_experiment_table()
    container_expts = experiments_table[experiments_table.ophys_container_id == container_id]
    experiment_ids = np.sort(container_expts.index.values)
    cell_specimen_table = pd.DataFrame()
    for experiment_id in experiment_ids:
        dataset = get_ophys_dataset(experiment_id)
        ct = dataset.cell_specimen_table.copy()
        cell_specimen_table = pd.concat([cell_specimen_table, ct])
    cell_specimen_ids = cell_specimen_table.index.unique()
    return cell_specimen_ids


# FROM MTRAIN DATABASE


def get_mtrain_stage_name(dataframe):
    foraging_ids = dataframe['foraging_id'][~pd.isnull(dataframe['foraging_id'])]
    query = """
            SELECT
            stages.name as stage_name,
            bs.id as foraging_id
            FROM behavior_sessions bs
            LEFT JOIN states ON states.id = bs.state_id
            LEFT JOIN stages ON stages.id = states.stage_id
            WHERE bs.id IN ({})
        """.format(",".join(["'{}'".format(x) for x in foraging_ids]))
    mtrain_response = pd.read_sql(query, mtrain_engine.get_connection())
    dataframe = dataframe.merge(mtrain_response, on='foraging_id', how='left')
    dataframe = dataframe.rename(columns={"stage_name": "stage_name_mtrain"})
    return dataframe


def build_container_df(experiment_table):
    '''
    build dataframe with one row per container
    '''
    table = experiment_table.copy()
    # table = get_filtered_ophys_experiment_table().sort_values(by='date_of_acquisition', ascending=False).reset_index()
    ophys_container_ids = table['ophys_container_id'].unique()
    list_of_dicts = []
    for ophys_container_id in ophys_container_ids:
        subset = table.query('ophys_container_id == @ophys_container_id').sort_values(
            by='date_of_acquisition',
            ascending=True
        ).drop_duplicates('ophys_session_id').reset_index()
        temp_dict = {
            'ophys_container_id': ophys_container_id,
            # 'container_workflow_state': table.query('ophys_container_id == @ophys_container_id')['container_workflow_state'].unique()[0],
            'project_code': subset['project_code'].unique()[0],
            'mouse_id': subset['mouse_id'].unique()[0],
            'sex': subset['sex'].unique()[0],
            'age_in_days': subset['age_in_days'].min(),
            'full_genotype': subset['full_genotype'][0],
            'cre_line': subset['cre_line'][0],
            'targeted_structure': subset['targeted_structure'].unique()[0],
            'imaging_depth': subset['imaging_depth'].unique()[0],
            'first_acquisition_date': subset['date_of_acquisition'].min().split(' ')[0],
            'equipment_name': subset['equipment_name'].unique(),
        }
        for idx, row in subset.iterrows():
            temp_dict.update(
                {'session_{}'.format(idx): '{} experiment_id:{}'.format(row['session_type'], row['ophys_experiment_id'])})

        list_of_dicts.append(temp_dict)
    container_df = pd.DataFrame(list_of_dicts).sort_values(by='ophys_container_id', ascending=False)
    container_df = container_df.set_index(['ophys_container_id'])
    return container_df


# def build_mouse_df(experiment_table):
#     '''
#     build dataframe with one row per mouse
#     '''
#     table = experiment_table.copy()
#     mouse_ids = table['mouse_id'].unique()
#     list_of_dicts = []
#     for mouse_id in mouse_ids:
#         subset = table.query('mouse_id == @mouse_id').sort_values(by='date_of_acquisition',
#                                             ascending=True).drop_duplicates('container_id').reset_index()
#         temp_dict = {
#             'mouse_id': mouse_id,
#             'project_code': subset['project_code'].unique()[0],
#             # 'container_id': subset['container_id'].unique()[0],
#             'full_genotype': subset['full_genotype'][0],
#             'cre_line': subset['cre_line'][0],
#             'targeted_structure': subset['targeted_structure'].unique()[0],
#             'imaging_depth': subset['imaging_depth'].unique()[0],
#             'sex': subset['sex'].unique()[0],
#             'age_in_days': subset['age_in_days'].min(),
#             'first_acquisition_date': subset['date_of_acquisition'].min().split(' ')[0],
#             'equipment_name': subset['equipment_name'].unique(),
#         }
#         for idx, row in subset.iterrows():
#             temp_dict.update(
#                 {'session_{}'.format(idx): '{} container_id:{}'.format(row['session_type'], row['ophys_session_id'])})
#
#         list_of_dicts.append(temp_dict)
#     mouse_df = pd.DataFrame(list_of_dicts).sort_values(by=['project_code', 'mouse_id', 'container_id'], ascending=False)
#     mouse_df = mouse_df.set_index(['mouse_id', 'container_id'])
#     return mouse_df

# multi session summary data #########

def get_annotated_experiments_table():
    project_codes = ['VisualBehavior', 'VisualBehaviorTask1B',
                     'VisualBehaviorMultiscope', 'VisualBehaviorMultiscope4areasx2d']

    experiments_table = get_filtered_ophys_experiment_table()
    experiments_table = experiments_table[experiments_table.project_code.isin(project_codes)]
    # add columns
    experiments_table['depth'] = ['superficial' if experiments_table.loc[expt].imaging_depth <= 250 else 'deep' for expt
                                  in experiments_table.index]  # 355
    experiments_table['location'] = [experiments_table.loc[expt].cre_line.split('-')[0] + '_' +
                                     experiments_table.loc[expt].depth for expt in experiments_table.index]

    experiments_table['layer'] = None
    indices = experiments_table[(experiments_table.imaging_depth < 100)].index.values
    experiments_table.loc[indices, 'layer'] = 'L1'
    indices = experiments_table[(experiments_table.imaging_depth < 270) &
                                (experiments_table.imaging_depth >= 100)].index.values
    experiments_table.loc[indices, 'layer'] = 'L2/3'

    indices = experiments_table[
        (experiments_table.imaging_depth >= 270) & (experiments_table.imaging_depth < 350)].index.values
    experiments_table.loc[indices, 'layer'] = 'L4'
    indices = experiments_table[
        (experiments_table.imaging_depth >= 350) & (experiments_table.imaging_depth < 550)].index.values
    experiments_table.loc[indices, 'layer'] = 'L5'

    experiments_table['location_layer'] = [experiments_table.loc[expt].cre_line.split('-')[0] + '_' +
                                           experiments_table.loc[expt].targeted_structure + '_' +
                                           experiments_table.loc[expt].layer for expt in experiments_table.index]

    indices = experiments_table[experiments_table.location == 'Slc17a7_superficial'].index.values
    experiments_table.loc[indices, 'location'] = 'Excitatory superficial'
    indices = experiments_table[experiments_table.location == 'Slc17a7_deep'].index.values
    experiments_table.loc[indices, 'location'] = 'Excitatory deep'
    indices = experiments_table[experiments_table.location == 'Vip_superficial'].index.values
    experiments_table.loc[indices, 'location'] = 'Vip'
    indices = experiments_table[experiments_table.location == 'Sst_superficial'].index.values
    experiments_table.loc[indices, 'location'] = 'Sst'
    indices = experiments_table[experiments_table.location == 'Vip_deep'].index.values
    experiments_table.loc[indices, 'location'] = 'Vip'
    indices = experiments_table[experiments_table.location == 'Sst_deep'].index.values
    experiments_table.loc[indices, 'location'] = 'Sst'

    experiments_table['session_number'] = [int(session_type[6]) for session_type in
                                           experiments_table.session_type.values]
    experiments_table['cre'] = [cre.split('-')[0] for cre in experiments_table.cre_line.values]

    return experiments_table


def add_superficial_deep_to_experiments_table(experiments_table):
    experiments_table['depth'] = ['superficial' if experiments_table.loc[expt].imaging_depth <= 250 else 'deep' for expt
                                  in experiments_table.index]  # 355
    experiments_table['location'] = [experiments_table.loc[expt].cre_line.split('-')[0] + '_' +
                                     experiments_table.loc[expt].depth for expt in experiments_table.index]
    indices = experiments_table[experiments_table.location == 'Slc17a7_superficial'].index.values
    experiments_table.loc[indices, 'location'] = 'Excitatory superficial'
    indices = experiments_table[experiments_table.location == 'Slc17a7_deep'].index.values
    experiments_table.loc[indices, 'location'] = 'Excitatory deep'
    indices = experiments_table[experiments_table.location == 'Vip_superficial'].index.values
    experiments_table.loc[indices, 'location'] = 'Vip'
    indices = experiments_table[experiments_table.location == 'Sst_superficial'].index.values
    experiments_table.loc[indices, 'location'] = 'Sst'
    indices = experiments_table[experiments_table.location == 'Vip_deep'].index.values
    experiments_table.loc[indices, 'location'] = 'Vip'
    indices = experiments_table[experiments_table.location == 'Sst_deep'].index.values
    experiments_table.loc[indices, 'location'] = 'Sst'

    return experiments_table


def get_file_name_for_multi_session_df_no_session_type(df_name, project_code, conditions, use_events, filter_events):
    if use_events:
        if filter_events:
            suffix = '_filtered_events'
        else:
            suffix = '_events'
    else:
        suffix = ''

    if len(conditions) == 5:
        filename = 'mean_' + df_name + '_' + project_code + '_' + conditions[1] + '_' + conditions[
            2] + '_' + conditions[3] + '_' + conditions[4] + suffix + '.h5'
    elif len(conditions) == 4:
        filename = 'mean_' + df_name + '_' + project_code + '_' + conditions[1] + '_' + conditions[
            2] + '_' + conditions[3] + suffix + '.h5'
    elif len(conditions) == 3:
        filename = 'mean_' + df_name + '_' + project_code + '_' + conditions[1] + '_' + conditions[
            2] + suffix + '.h5'
    elif len(conditions) == 2:
        filename = 'mean_' + df_name + '_' + project_code + '_' + conditions[1] + suffix + '.h5'
    elif len(conditions) == 1:
        filename = 'mean_' + df_name + '_' + project_code + '_' + conditions[0] + suffix + '.h5'

    return filename


# def get_file_name_for_multi_session_df(df_name, project_code, session_type, conditions, use_events, filter_events):
#     if use_events:
#         if filter_events:
#             suffix = '_filtered_events'
#         else:
#             suffix = '_events'
#     else:
#         suffix = ''
#
#     if len(conditions) == 6:
#         filename = 'mean_' + df_name + '_' + project_code + '_' + session_type + '_' + conditions[1] + '_' + conditions[2] + '_' + conditions[3] + '_' + conditions[4] + '_' + conditions[5] + suffix + '.h5'
#     elif len(conditions) == 5:
#         filename = 'mean_' + df_name + '_' + project_code + '_' + session_type + '_' + conditions[1] + '_' + conditions[2] + '_' + conditions[3] + '_' + conditions[4] + suffix + '.h5'
#     elif len(conditions) == 4:
#         filename = 'mean_' + df_name + '_' + project_code + '_' + session_type + '_' + conditions[1] + '_' + conditions[2] + '_' + conditions[
#             3] + suffix + '.h5'
#     elif len(conditions) == 3:
#         filename = 'mean_' + df_name + '_' + project_code + '_' + session_type + '_' + conditions[1] + '_' + conditions[2] + suffix + '.h5'
#     elif len(conditions) == 2:
#         filename = 'mean_' + df_name + '_' + project_code + '_' + session_type + '_' + conditions[1] + suffix + '.h5'
#     elif len(conditions) == 1:
#         filename = 'mean_' + df_name + '_' + project_code + '_' + session_type + '_' + conditions[0] + suffix + '.h5'
#
#     return filename


def get_file_name_for_multi_session_df(data_type, event_type, project_code, session_type, conditions):

    if len(conditions) == 6:
        filename = 'mean_response_df_' + data_type + '_' + event_type + '_' + project_code + '_' + session_type + '_' + conditions[1] + '_' + conditions[2] + '_' + conditions[3] + '_' + conditions[4] + '_' + conditions[5] + '.h5'
    elif len(conditions) == 5:
        filename = 'mean_response_df_' + data_type + '_' + event_type + '_' + project_code + '_' + session_type + '_' + conditions[1] + '_' + conditions[2] + '_' + conditions[3] + '_' + conditions[4] + '.h5'
    elif len(conditions) == 4:
        filename = 'mean_response_df_' + data_type + '_' + event_type + '_' + project_code + '_' + session_type + '_' + conditions[1] + '_' + conditions[2] + '_' + conditions[3] + '.h5'
    elif len(conditions) == 3:
        filename = 'mean_response_df_' + data_type + '_' + event_type + '_' + project_code + '_' + session_type + '_' + conditions[1] + '_' + conditions[2] + '.h5'
    elif len(conditions) == 2:
        filename = 'mean_response_df_' + data_type + '_' + event_type + '_' + project_code + '_' + session_type + '_' + conditions[1] + '.h5'
    elif len(conditions) == 1:
        filename = 'mean_response_df_' + data_type + '_' + event_type + '_' + project_code + '_' + session_type + '_' + conditions[0] + '.h5'

    return filename


def load_multi_session_df(cache_dir, df_name, conditions, experiments_table, remove_outliers=False, use_session_type=True,
                          use_events=True, filter_events=False):
    """
    Loops through all experiments in the provided experiments_table, creates a response dataframe indicated by df_name,
    creates a mean response dataframe for a given set of conditions, and concatenates across all experiments to create
    one large multi session dataframe with trial averaged responses and other relevant metrics. Saves multi_session_df
    to the cache dir as a separate .h5 file per project_code and session_type combination present in the provided
    experiments_table.
    :param cache_dir: to level directory directory to save resulting dataframes, must contain folder called 'multi_session_summary_dfs'
    :param df_name: the name of the response dataframe to be created using the ResponseAnalysis class, such as 'stimulus_response_df'
    :param conditions: the set of conditions over which to group and average cell responses using the get_mean_df()
                        function in response_analysis.utilities, such as ['cell_specimen_id', 'engagement_state', 'image_name']
    :param experiments_table: full or subset of experiments_table from loading.get_filtered_ophys_experiments_table()
    :param remove_outliers: Boolean, whether to remove cells with a max average dF/F > 5 (not a principled way of doing this)
    :param use_session_type: Boolean for whether or not to save resulting dataframes by session type or to aggregate across session types.
                        Grouping and saving by session type is typically necessary given the large size of these dataframes.
    :param use_events: Boolean, whether to use events instead of dF/F when creating response dataframes
    :return: multi_session_df for conditions specified above
    """

    cache_dir = get_platform_analysis_cache_dir()
    cache = bpc.from_s3_cache(cache_dir=cache_dir)
    experiments_table = cache.get_ophys_experiment_table()

    project_codes = experiments_table.project_code.unique()
    multi_session_df = pd.DataFrame()
    for project_code in project_codes:
        experiments = experiments_table[(experiments_table.project_code == project_code)]
        if project_code == 'VisualBehaviorMultiscope':
            experiments = experiments[experiments.session_type != 'OPHYS_2_images_B_passive']
        # expts = experiments_table.reset_index()
        if use_session_type:
            for session_type in np.sort(experiments.session_type.unique()):
                try:
                    filename = get_file_name_for_multi_session_df(df_name, project_code, session_type, conditions,
                                                                  use_events, filter_events)
                    filepath = os.path.join(get_platform_analysis_cache_dir(), 'multi_session_summary_dfs', filename)
                    # print('reading file at', filepath)
                    df = pd.read_hdf(filepath, key='df')
                    # df = df.merge(expts, on='ophys_experiment_id')
                    if remove_outliers:
                        outlier_cells = df[df.mean_response > 5].cell_specimen_id.unique()
                        df = df[df.cell_specimen_id.isin(outlier_cells) == False]
                    multi_session_df = pd.concat([multi_session_df, df])
                except BaseException:
                    print('no multi_session_df for', project_code, session_type)
        else:
            filename = get_file_name_for_multi_session_df_no_session_type(df_name, project_code, conditions, use_events, filter_events)
            filepath = os.path.join(cache_dir, 'multi_session_summary_dfs', filename)
            df = pd.read_hdf(filepath, key='df')
            # df = df.merge(expts[['ophys_experiment_id', 'cre_line', 'location', 'location_layer',
            #                      'layer', 'ophys_session_id', 'project_code', 'session_type',
            #                      'specimen_id', 'depth', 'exposure_number', 'ophys_container_id']], on='ophys_experiment_id')
            if remove_outliers:
                outlier_cells = df[df.mean_response > 5].cell_specimen_id.unique()
            df = df[df.cell_specimen_id.isin(outlier_cells) == False]
    return multi_session_df


def remove_outlier_traces_from_multi_session_df(multi_session_df):
    indices = [row for row in multi_session_df.index.values if (multi_session_df.mean_trace.values[row].max() > 5)]

    multi_session_df = multi_session_df[multi_session_df.index.isin(indices) == False]
    multi_session_df = multi_session_df.reset_index()
    multi_session_df = multi_session_df.drop(columns=['index'])
    return multi_session_df


def remove_first_novel_session_retakes_from_df(df):
    indices = df[(df.session_number == 4) & (df.prior_exposures_to_image_set != 0)].index
    df = df.drop(index=indices)
    df = df.reset_index(drop=True)
    return df


def remove_problematic_data_from_multi_session_df(multi_session_df):
    # notes on containers, experiments & mice #####
    # experiments to exclude
    # another Slc mouse with large familiar passive 840542948, also ramping in some familiar sessions, generally noisy
    # all mouse 920877188 remove & investiate later - weirdness
    # slc superficial very high novel 2 container 1018027834
    # Slc container 1018027644 has inconsistent familiar sessions
    # another Slc mouse with large passive familiar responses 837581585, only one container
    # slc deep with abnormally large familiar passive 1018028067, 1018028055

    # havent exlcuded these yet
    # 904363938, 905955240 #weird
    # 903485718, 986518889, 903485718 # omission in Slc?!
    # 971813765 # potentially bad

    # experiments of interest
    # Sst mouse with omission ramping 813702151, containers 1018028135, 1019028153
    # sst large novelty response N1 mouse 850862430 container 1018028339, 1018028351
    # SST omission ramping for familiar 904922342

    # slc containers with omission responses for familiar sessions, superficial - 1018028046, 1018028061 same mouse
    # another Slc with omission ramping 840390377 but only container 1018027878
    # Slc with omission ramping in all session types 843122504 container 1018027663 VISl superficial at 275 but not at 75 in VIsl or at 275 in V1

    # Vip mouse with really large familiar 1 activity 810573072, noisy familiar 2
    # VIp mouse with abnormally high familiar 3 791871803
    # Vip with abormally high novel 1 837628436
    # Vip mouse with novelty like responses to familiar 3 807248992, container 1018028367 especially but also others

    bad_specimen_ids = [810573072]  # 840542948, 920877188, 837581585
    bad_experiment_ids = [989610992, 904352692, 905955211, 974362760, 904363938, 905955240,
                          957759566, 957759574, 916220452, 934550023,
                          986518876, 986518891, 989610991, 989213058, 989213062, 990400778, 990681008, 991852002,
                          # specimen 920877188
                          915229064, 934550019, 919419011,  # specimen 840542948
                          847267618, 847267620, 848039121, 848039125, 848039123, 848760990,  # specimen 810573072
                          886585138, 886565136,  # specimen 837581585
                          ]
    multi_session_df = multi_session_df[multi_session_df.specimen_id.isin(bad_specimen_ids) == False]
    multi_session_df = multi_session_df[multi_session_df.ophys_experiment_id.isin(bad_experiment_ids) == False]

    return multi_session_df


def annotate_and_clean_multi_session_df(multi_session_df):
    def get_session_labels():
        return ['F1', 'F2', 'F3', 'N1', 'N2', 'N3']

    multi_session_df.session_number = [int(number) for number in multi_session_df.session_number.values]
    multi_session_df['session_name'] = [get_session_labels()[session_number - 1] for session_number in
                                        multi_session_df.session_number.values]

    multi_session_df = remove_first_novel_session_retakes_from_df(multi_session_df)
    multi_session_df = remove_outlier_traces_from_multi_session_df(multi_session_df)
    # multi_session_df = remove_problematic_data_from_multi_session_df(multi_session_df)

    return multi_session_df


def get_concatenated_stimulus_presentations(project_codes=None, session_numbers=None):
    """
    loads stimulus presentation table data for multiple sessions from cached files.
    dataframe contains stimulus information including image_name, change, licked, omitted, etc for every stimulus presentation.
    can be merged with stimulus_response_dfs to get cell responses for all stimulus presentations.

    if desired project codes and session numbers are not specified, all data will be loaded.
    project_codes and session_numbers should be provided as lists,
    ex: project_codes = ['VisualBehaviorTask1B', 'VisualBehaviorMultiscope], session_numbers = [3, 4]
    """

    save_dir = os.path.join(get_decoding_analysis_dir(), 'data')
    experiments_table = get_filtered_ophys_experiment_table()
    if project_codes is None:
        project_codes = experiments_table.project_code.unique()
    if session_numbers is None:
        session_numbers = experiments_table.session_number.unique()

    concatenated_stimulus_presentations = pd.DataFrame()
    for project_code in project_codes:
        for session_number in session_numbers:
            try:
                df = pd.read_hdf(os.path.join(save_dir, 'stimulus_presentations_' + project_code + '_session_' + str(
                    session_number) + '.h5'), key='df')
                concatenated_stimulus_presentations = pd.concat([concatenated_stimulus_presentations, df])
            except Exception as e:
                print('problem for', project_code, session_number)
                print(e)
    return concatenated_stimulus_presentations


def get_concatenated_stimulus_response_dfs(project_codes=None, session_numbers=None):
    """
    loads stimulus response dataframes for multiple sessions from cached files.
    dataframe contains the response of each cell for each stimulus presentation across sessions,
    including the trace in a [0, 0.75] second window, the mean response in that window, etc.
    can be merged with stimulus presentations to get the stimulus conditions for each cell response.

    if desired project codes and session numbers are not specified, all data will be loaded.
    project_codes and session_numbers should be provided as lists,
    ex: project_codes = ['VisualBehaviorTask1B', 'VisualBehaviorMultiscope], session_numbers = [3, 4]
    """

    save_dir = os.path.join(get_decoding_analysis_dir(), 'data')
    experiments_table = get_filtered_ophys_experiment_table()
    if project_codes is None:
        project_codes = experiments_table.project_code.unique()
    if session_numbers is None:
        session_numbers = experiments_table.session_number.unique()
    cre_lines = experiments_table.cre_line.unique()

    concatenated_stimulus_response_dfs = pd.DataFrame()
    for project_code in project_codes:
        for session_number in session_numbers:
            for cre_line in cre_lines:
                try:
                    df = pd.read_hdf(os.path.join(save_dir,
                                                  'stimulus_response_dfs_' + project_code + '_' + cre_line + '_session_' + str(
                                                      session_number) + '.h5'), key='df')
                    concatenated_stimulus_response_dfs = pd.concat([concatenated_stimulus_response_dfs, df])
                except Exception as e:
                    print('problem for', project_code, cre_line, session_number)
                    print(e)
    return concatenated_stimulus_response_dfs


def get_stimulus_response_data_across_sessions(project_codes=None, session_numbers=None):
    """
    loads and merges stimulus_response_dfs, stimulus_presentations data, and experiments_table metadata
    across sessions for a given set of project_codes and session_numbers.
    returns all cell responses for all image flashes for that set of sessions.

    if desired project codes and session numbers are not specified, all data will be loaded (slow).
    project_codes and session_numbers should be provided as lists,
    ex: project_codes = ['VisualBehaviorTask1B', 'VisualBehaviorMultiscope], session_numbers = [3, 4]
    """

    experiments_table = get_filtered_ophys_experiment_table()
    if project_codes is None:
        project_codes = experiments_table.project_code.unique()
    if session_numbers is None:
        session_numbers = experiments_table.session_number.unique()

    stim_presentations = get_concatenated_stimulus_presentations(project_codes, session_numbers)
    stim_response_dfs = get_concatenated_stimulus_response_dfs(project_codes, session_numbers)
    stimulus_response_data = stim_response_dfs.merge(stim_presentations, on='ophys_session_id')
    stimulus_response_data = stimulus_response_data.merge(experiments_table,
                                                          on=['ophys_experiment_id', 'ophys_session_id'])
    return stimulus_response_data


def get_cell_info(cell_specimen_ids=None, ophys_experiment_ids=None):
    '''
    returns a table of info about each unique cell ROI
    input:
        list, array, or series of cell_specimen_ids or a single cell_specimen_id
        list, array, or series of ophys_experiment_ids or a single ophys_experiment_id
    returns:
        a dataframe with columns:
            cell_roid: unique ID in LIMS
            cell_specimen_id: unique ID for each matched cell across planes
            cell_specimen_id_created_at: date cell_specimen_id created
            cell_specimen_id_updated_at: date cell_specimen_id updated
            ophys_session_id
            ophys_experiment_id
            experiment_container_id
            supercontainer_id

    examples:
    >> cell_info = get_cell_info(cell_specimen_ids = [1018032458, 1018032468, 1063351131])

    >> cell_info = get_cell_info(ophys_experiment_ids = [850517344, 953659749])
    '''
    if isinstance(cell_specimen_ids, int):
        search_vals = "({})".format(cell_specimen_ids)
        search_key = 'cell_specimen_id'
    elif isinstance(cell_specimen_ids, (list, np.ndarray, pd.Series)):
        search_vals = tuple(cell_specimen_ids)
        search_key = 'cell_specimen_id'
    elif isinstance(ophys_experiment_ids, int):
        search_vals = "({})".format(ophys_experiment_ids)
        search_key = 'oe.id'
    elif isinstance(ophys_experiment_ids, (list, np.ndarray, pd.Series)):
        search_vals = tuple(ophys_experiment_ids)
        search_key = 'oe.id'

    query = '''
    select
        cell_rois.id as cell_roi_id,
        cell_rois.cell_specimen_id,
        specimens.created_at as cell_specimen_id_created_at,
        specimens.updated_at as cell_specimen_id_updated_at,
        oe.ophys_session_id,
        oe.id as ophys_experiment_id,
        vbec.id as experiment_container_id,
        visual_behavior_supercontainers.id as supercontainer_id
    from cell_rois
    left join ophys_experiments as oe on cell_rois.ophys_experiment_id = oe.id
    left join specimens on cell_rois.cell_specimen_id = specimens.id
    left join ophys_experiments_visual_behavior_experiment_containers as oevbec on oe.id = oevbec.ophys_experiment_id
    left join visual_behavior_experiment_containers as vbec on vbec.id = oevbec.visual_behavior_experiment_container_id
    left join visual_behavior_supercontainers on visual_behavior_supercontainers.specimen_id = vbec.specimen_id
    where {} in {}
    '''
    return db.lims_query(query.format(search_key, search_vals))


def get_container_response_df(ophys_container_id, df_name='omission_response_df', use_events=False):
    """
    get concatenated dataframe of response_df type specificied by df_name, across all experiments from a container,
    using the ResponseAnalysis class to build event locked response dataframes
    """
    from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis
    experiments_table = get_filtered_ophys_experiment_table()
    container_expts = experiments_table[experiments_table.ophys_container_id == ophys_container_id]
    container_df = pd.DataFrame()
    for ophys_experiment_id in container_expts.index.values:
        dataset = get_ophys_dataset(ophys_experiment_id)
        analysis = ResponseAnalysis(dataset, use_events)
        odf = analysis.get_response_df(df_name=df_name)
        odf['ophys_experiment_id'] = ophys_experiment_id
        odf['session_number'] = experiments_table.loc[ophys_experiment_id].session_number
        container_df = pd.concat([container_df, odf])
    return container_df


def get_cell_summary(search_dict={}):
    '''
    gets summary stats for all cells
    relies on cache of summary stats in internal mongo database
    merges in filtered_ophys_experiment_table for convenience
    input:
        search_dict -- dictionary of key/value pairs to constrain search (empty dict returns all cells)
    returns:
        pandas dataframe with one row per cell
        see database.get_cell_dff_data for description of columns
    '''
    cell_table = db.get_cell_dff_data(search_dict=search_dict)
    experiment_table = get_filtered_ophys_experiment_table().reset_index()
    cell_table = cell_table.merge(
        experiment_table,
        left_on='ophys_experiment_id',
        right_on='ophys_experiment_id'
    )
    return cell_table


def get_remaining_crosstalk_amount_dict(experiment_id):
    import allensdk.core.json_utilities as ju
    import visual_behavior.data_access.utilities as utilities

    session_dir = utilities.get_ophys_session_dir(utilities.get_lims_data(experiment_id))
    candidate_folders = [folder for folder in os.listdir(os.path.join(session_dir, 'crosstalk')) if 'roi' in folder]
    folder = [folder for folder in candidate_folders if str(experiment_id) in folder]
    json_path = os.path.join(session_dir, 'crosstalk', folder[0], str(experiment_id) + '_crosstalk.json')
    crosstalk_dict = ju.read(json_path)

    remaining_crosstalk_dict = {}
    for key in list(crosstalk_dict.keys()):
        remaining_crosstalk_dict[int(key)] = crosstalk_dict[key][1]

    return remaining_crosstalk_dict


def get_cell_table_from_lims(ophys_experiment_ids=None, columns_to_return='*', valid_rois_only=False, platform_paper_only=False):
    '''
    retrieves the full cell_specimen table from LIMS for the specified ophys_experiment_ids
    if no ophys_experiment_ids are passed, all experiments from the `VisualBehaviorOphysProjectCache` will be retrieved

    Parameters
    ----------
    ophys_experiment_ids : list
        A list of ophys_experiment_ids for which to retrieve the associated cells.
        If None, all experiments from the `VisualBehaviorOphysProjectCache` will be retrieved.
        Default = None
    columns_to_return
        A list of which colums to return.
        If "*" is passed, all columns will be returned.
        Queries will be faster if fewer columns are returned.
        Possible columns that can be returned:
            cell_roi_id
            cell_specimen_id
            ophys_experiment_id
            x
            y
            width
            height
            valid_roi
            mask_matrix
            max_correction_up
            max_correction_down
            max_correction_right
            max_correction_left
            mask_image_plane
            ophys_cell_segmentation_run_id
        default = '*'
    valid_rois_only: bool
        If False (default), all ROIs will be returned
        If True, only valid ROIs will be returned
    platform_paper_only: bool
        Only has an effect is ophys_experiment_ids==None
        If False (default), all ROIs will be returned
        If True, ROIs from Multiscope 4areasx2depths and Ai94 data will be excluded

    Returns
    -------
    pandas.DataFrame
        A dataframe with one row per cell and each of the requested columns


    Examples:
    -------
    This will return all columns for all released experiments
    This takes about 5 seconds
    >> cell_table = get_cell_table()

    This will return only the columns ['ophys_experiment_id','cell_specimen_id'] for all released experiments
    This takes about 1.5 seconds
    >> get_cell_table(columns_to_return = ['ophys_experiment_id','cell_specimen_id'])

    This will return only the columns ['ophys_experiment_id','cell_specimen_id'] for the specified experiments
    This takes about 20 ms
    >> oeids = [792813858, 888876943, 986518885, 942596355, 908381680]
    >> cell_table = get_cell_table(ophys_session_ids = oeids, columns_to_return = ['ophys_experiment_id','cell_specimen_id'])

    This will return all columns for the specified experiments
    This takes about 50 ms
    >> oeids = [792813858, 888876943, 986518885, 942596355, 908381680]
    >> cell_table = get_cell_table(ophys_session_ids = oeids)


    '''
    # get ophys_experiment_ids from lims if none were passed
    # this includes failed experiments
    if ophys_experiment_ids is None:
        cache = bpc.from_lims()
        experiment_table = cache.get_ophys_experiment_table()

        # Exclude 4x2 and GCaMP6s mice
        if platform_paper_only:
            cache = bpc.from_lims(data_release_date=['03-25-2021', '08-12-2021'])
            experiment_table = experiment_table[(experiment_table.project_code != "VisualBehaviorMultiscope4areasx2d") & (experiment_table.reporter_line != "Ai94(TITL-GCaMP6s)")]

        ophys_experiment_ids = experiment_table.index.unique()

    if columns_to_return != '*':
        columns_to_return = ', '.join(columns_to_return).replace('cell_roi_id', 'id')

    if valid_rois_only:
        query = '''
            select {}
            from cell_rois
            where ophys_experiment_id in {} and cell_specimen_id is not null and valid_roi = True
        '''
    else:
        query = '''
            select {}
            from cell_rois
            where ophys_experiment_id in {} and cell_specimen_id is not null
        '''

    # Since we are querying from the 'cell_rois' table, the 'id' column is actually 'cell_roi_id'. Rename.
    lims_rois = db.lims_query(
        query.format(columns_to_return, tuple(ophys_experiment_ids))
    ).rename(columns={'id': 'cell_roi_id'})

    return lims_rois


def get_cell_table(platform_paper_only=True, add_extra_columns=True):
    """
    loads ophys_cells_table from the SDK using platform paper analysis cache and merges with experiment_table to get metadata
    if 'platform_paper_only' is True, will filter out Ai94 and VisuaBehaviorMultiscope4areasx2d and add extra columns
    :return:
    """
    cache_dir = get_platform_analysis_cache_dir()
    cache = bpc.from_s3_cache(cache_dir=cache_dir)
    # load cell table
    cell_table = cache.get_ophys_cells_table()
    # optionally filter to limit to platform paper datasets
    if platform_paper_only == True:
        # load experiments table and merge
        experiment_table = get_platform_paper_experiment_table(add_extra_columns=add_extra_columns)
        cell_table = cell_table.reset_index().merge(experiment_table, on='ophys_experiment_id')
        cell_table = cell_table[(cell_table.reporter_line != 'Ai94(TITL-GCaMP6s)') & (cell_table.project_code != 'VisualBehaviorMultiscope4areasx2d')]
        cell_table = cell_table.set_index('cell_roi_id')
    else:
        # load platform experiments table and merge
        experiment_table = cache.get_ophys_experiment_table()
        cell_table = cell_table.reset_index().merge(experiment_table, on='ophys_experiment_id')
        cell_table = cell_table.set_index('cell_roi_id')
    return cell_table


def get_data_dict(ophys_experiment_ids, data_types=None, save_dir=None):
    """
    create dictionary of stimulus_response_dfs for all data types for a set of ophys_experiment_ids
    data types include [filtered_events, events, dff, running_speed, pupil_diameter, lick_rate]
    If stimulus_response_df files have been pre-computed, load from file, otherwise generate new response df
    """
    if data_types is None:
        data_types = ['filtered_events', 'running_speed', 'pupil_diameter', 'lick_rate']
    # get cache
    from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache
    cache_dir = get_platform_analysis_cache_dir()
    cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir)
    # define params
    time_window = [-3, 3.1]
    interpolate = True
    output_sampling_rate = 30
    # set up dict to collect data in
    data_dict = {}
    for ophys_experiment_id in ophys_experiment_ids:
        data_dict[ophys_experiment_id] = {}
        data_dict[ophys_experiment_id]['dataset'] = {}
        for data_type in data_types:
            data_dict[ophys_experiment_id][data_type] = {}

    # aggregate data
    for ophys_experiment_id in ophys_experiment_ids:

        # dataset = get_ophys_dataset(ophys_experiment_id)
        dataset = cache.get_behavior_ophys_experiment(ophys_experiment_id)
        data_dict[ophys_experiment_id]['dataset']['dataset'] = dataset

        for data_type in data_types:
            try:
                sdf = get_stimulus_response_df(dataset, time_window=time_window, interpolate=interpolate, output_sampling_rate=output_sampling_rate,
                                               data_type=data_type, load_from_file=True)
                data_dict[ophys_experiment_id][data_type]['changes'] = sdf[sdf.is_change]
                data_dict[ophys_experiment_id][data_type]['omissions'] = sdf[sdf.omitted]
            except BaseException:
                print('could not get response df for', ophys_experiment_id, data_type)

    return data_dict
