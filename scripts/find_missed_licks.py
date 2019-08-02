import sys
import os
import glob
import time
import copy
import fnmatch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import medfilt
from scipy.interpolate import interp1d
import seaborn as sns
from pathlib import Path

from allensdk.brain_observatory.extract_running_speed.__main__ import extract_running_speeds
from allensdk.internal.api import behavior_ophys_api as boa
from allensdk.brain_observatory.behavior import behavior_ophys_session as bos
from allensdk.internal.api import PostgresQueryMixin
from visual_behavior.utilities import Movie
import sync
from sync import Dataset


"""
07/26/2019
mattv@alleninstitute.org
"""


def get_session(df):
    # input is dataframe created through SQL query to the allensdk.internal api
        
    for ind_iter, (ind_row, row) in enumerate(df.iterrows()):
        #api and session for row experiment ID
        exp_id = row['ophys_experiment_id']
        api = boa.BehaviorOphysLimsApi(exp_id)
        session = bos.BehaviorOphysSession(api)

    return session


def process_running(session):

	running_speed = session.running_speed.values
	running_speed = medfilt(running_speed, kernel_size=5)
	running_speed_times = session.running_speed.timestamps

	return running_speed, running_speed_times


def make_trial_df(session, running_speed, running_speed_times, win=(-0.5, 1.5)):

	# get change times
	trial_df = copy.deepcopy(session.trials)
	trial_df = trial_df[trial_df['auto_rewarded']==False]
	ct = trial_df.change_time

	# extract running speed in window around each event

	all_segs = []
	all_times = []
	for c,change in enumerate(ct):
	    if ~np.isnan(change):
	        rs_idx = [t for t,time in enumerate(running_speed_times) if (time>(change+win[0])) and (time<(change+win[1]))]
	        run_seg = running_speed[rs_idx]
	        run_times = running_speed_times[rs_idx]-change
	        if len(run_seg) != len(run_times):
	            print("ERROR")
	            all_segs.append(np.nan)
	            all_times.append(np.nan)
	            continue
	        all_times.append(run_times)
	        all_segs.append(run_seg)
	    else:
	        all_segs.append(np.nan)
	        all_times.append(np.nan)

	trial_df['win_running_speed'] = all_segs
	trial_df['win_running_times'] = all_times

	return trial_df


def plot_running_by_response(hit_df, miss_df):

	fig,ax=plt.subplots(1,2, figsize=(10,4))

	# plot hit running
	for r,run in enumerate(hit_df['win_running_speed'].values):
	    ax[0].plot(hit_df['win_running_times'].iloc[r], run, 'k', alpha=.2)
	    
	# plot miss running
	for r,run in enumerate(miss_df['win_running_speed'].values):
	    ax[1].plot(miss_df['win_running_times'].iloc[r], run, 'r', alpha=.2)


def compare_running(trial_df, hit_df, miss_df, win=(-0.5, 1.5), resamp_win=120):

	rssamp_ts = np.linspace(win[0],win[1],resamp_win)

	hit_mat = []
	for r,run in enumerate(hit_df['win_running_speed'].values):
	    run_resamp = np.interp(rssamp_ts, hit_df['win_running_times'].iloc[r], run)
	    hit_mat.append(run_resamp) 	
	template = np.mean(hit_mat, axis=0)
	hit_mse = [((template - run)**2).mean(axis=None) for run in hit_mat]
	hit_mat = np.array(hit_mat)

	mse = []
	miss_mat = []
	for r,run in enumerate(miss_df['win_running_speed'].values):
	    run_resamp = np.interp(rssamp_ts, miss_df['win_running_times'].iloc[r], run)
	    miss_mat.append(run_resamp)
	miss_mat = np.array(miss_mat)

	# calculate mean-squared-error between the tempalte and the miss running speed
	miss_norm = normalize_rows(miss_mat)
	temp_norm = normalize_rows(template)
	mse = ((miss_norm - temp_norm) ** 2).mean(axis=1)

	miss_mse_order = np.argsort(mse)
	miss_sort = miss_mat[miss_mse_order]
	miss_df = miss_df.iloc[miss_mse_order] # sort so that first rows have best match to template

	hit_mse_order = np.argsort(hit_mse)
	hit_df = hit_df.iloc[hit_mse_order] # sort so that first rows have best match to template

	return mse, hit_mat, miss_mat, template, rssamp_ts


def normalize_rows(dat):
	# scale to zero mean, unit variance
	dat = (dat - dat.mean(axis=0)) / dat.std(axis=0)
	return dat


def get_matching_running(mse, miss_df, thresh):

	low_mse_idx = np.where(np.array(mse)<thresh)[0]
	high_mse_idx = np.where(np.array(mse)>thresh)[0]

	mse_order = np.argsort(mse)

	miss_tocheck = miss_df['change_time'].iloc[low_mse_idx]

	return miss_tocheck, low_mse_idx, high_mse_idx


def plot_template_matches(trial_df, hit_df, miss_df, mse, hit_mat, miss_mat, template, rssamp_ts, thresh):

	miss_tocheck, low_mse_idx, high_mse_idx = get_matching_running(mse, miss_df, thresh)

	# plot running around hits and misses
	fig,ax=plt.subplots(1,2, figsize=(10,4))

	# plot hit running
	for i in range(hit_mat.shape[0]):
	    ax[0].plot(rssamp_ts, hit_mat[i,:], 'k', alpha=.2)
	    ax[0].plot(rssamp_ts, template, 'b')
	    ax[0].set_ylabel('running speed (cm/s)')
	    ax[0].set_xlabel('time from change (sec)')
	    
	print(miss_mat.shape, len(low_mse_idx))
	# plot miss running
	for i in low_mse_idx:
		ax[1].plot(rssamp_ts, miss_mat[i,:], 'r', alpha=.5)
		ax[1].plot(rssamp_ts, template, 'b')
	for i in high_mse_idx:
		ax[1].plot(rssamp_ts, miss_mat[i,:], 'k', alpha=.05)
		ax[1].set_ylabel('running speed (cm/s)')
		ax[1].set_xlabel('time from change (sec)')


def get_video(experiment_df):
	# get video files, sync files .. etc

	session_path = experiment_df['storage_directory'].values[0]
	session_path = Path('/' + session_path)
	print(session_path)
	matches = [file for file in os.listdir(session_path) if fnmatch.fnmatch(file, '*sync.h5')]
	sync_file = [os.path.join(session_path, match) for match in matches][0]
	sync_data = Dataset(sync_file)

	sample_freq = sync_data.meta_data['ni_daq']['counter_output_freq']

	video_paths = experiment_df.apply(lambda row: glob.glob(r'{}/*.avi'.format(session_path)), axis=1)

	camera_times = {}
	for camera in ['cam1','cam2']:
	    camera_times[camera] = sync_data.get_rising_edges('{}_exposure'.format(camera))/sample_freq

	# also get the lick times from sync for use below
	lick_times = sync_data.get_rising_edges('lick_sensor')/sample_freq

	movie = {}
	for movie_path in video_paths.values[0]:
		cam_index = int(movie_path.split('.avi')[0][-1]) + 1 # add 1: camera names are 1-indexed, movie names are 0-indexed
		camera_name = 'cam{}'.format(cam_index)
		movie[camera_name] = Movie(movie_path, sync_timestamps=camera_times[camera_name])
		#print('movie keys: {}'.format(movie.keys()))

	return experiment_df, movie


def plot_stop_frames(movie, miss_time, labtracks_id, ophys_experiment_id):
	time_lags = list(np.arange(6/30.,24/30.,1/30.))

	num_cols = int(np.ceil(len(time_lags)/4.0))
	fig,ax=plt.subplots(4,num_cols,figsize=(num_cols*3,12))

	mov = movie['cam1'] # cam1 is body cam, cam2 is eye cam

	for i, ax in enumerate(fig.axes):
		if i < len(time_lags):
			ax.set_title('{:0.3f} s\nafter change time'.format(time_lags[i]),fontsize=10)
			frame = mov.get_frame(time=miss_time+time_lags[i])
			ax.imshow(frame)
			ax.axis('off')
			fig.suptitle('labtracks_ID: '+labtracks_id+', ophys_experiment_id:'+ ophys_experiment_id+', change at {}'.format(miss_time))
		else:
			ax.axis('off')


def get_master_df():

	api = PostgresQueryMixin()
	query = '''
			SELECT

			oec.visual_behavior_experiment_container_id as container_id,
			oec.ophys_experiment_id,
			oe.workflow_state,
			d.full_genotype as full_genotype,
			d.id as donor_id,
			id.depth as imaging_depth,
			st.acronym as targeted_structure,
			os.name as session_name,
			os.storage_directory,
			equipment.name as equipment_name

			FROM ophys_experiments_visual_behavior_experiment_containers oec
			LEFT JOIN ophys_experiments oe ON oe.id = oec.ophys_experiment_id
			LEFT JOIN ophys_sessions os ON oe.ophys_session_id = os.id
			LEFT JOIN specimens sp ON sp.id=os.specimen_id
			LEFT JOIN donors d ON d.id=sp.donor_id
			LEFT JOIN imaging_depths id ON id.id=os.imaging_depth_id
			LEFT JOIN structures st ON st.id=oe.targeted_structure_id
			LEFT JOIN equipment ON equipment.id=os.equipment_id
			LEFT JOIN projects p ON p.id=os.project_id

			WHERE p.code = 'VisualBehavior'
	        '''

	## Additionally limit by qc state, and drop mesoscope sessions because we 
	## can't load the data with the SDK
	experiment_df = pd.read_sql(query, api.get_connection())
	states_to_use = ['container_qc', 'passed']

	conditions = [
	    "workflow_state in @states_to_use",
	    "equipment_name != 'MESO.1'",
	]

	query_string = ' and '.join(conditions)
	experiment_df = experiment_df.query(query_string)

	return experiment_df


def annotate_lick(ind_frame, annotated_lick_inds, annotated_no_lick_inds):
    print("Is there a real lick? [y/n]")
    response = input()
    if response=='y':
        annotated_lick_inds.append(ind_frame)
    elif response=='n':
        annotated_no_lick_inds.append(ind_frame)
    else:
        print("Please enter y or n")
        annotate_lick(ind_frame, annotated_lick_inds, annotated_no_lick_inds)


def audit_running_template():
    print("Does the running template look good? [y/n]")
    response = input()
    if response=='y':
        return True
    elif response=='n':
        return False
    else:
        print("Please enter y or n")
        audit_running_template(ind_frame, annotated_lick_inds, annotated_no_lick_inds)


def get_first_licks(df):
	first_licks = []
	for r,row in df.iterrows():
		licks = row.lick_times
		change = row.change_time
		lick = next(iter(licks), np.nan)
		first_licks.append(lick - change)
	return first_licks


def get_nearest_sample(ts, time):
    return np.abs(ts - time).argmin()


def make_output_dir(basepath):

	output_folder = time.strftime("%Y%m%d-%H%M%S")
	output_base_folder = os.path.join(basepath, output_folder)

	try:
		os.mkdir(output_base_folder)
		print("Directory " , output_base_folder ,  " Created ")
	except FileExistsError:
		print("Directory " , output_base_folder ,  " already exists")

	return output_base_folder


def find_miss_licks():
	# execute to sort miss trials by their matched running-speed to hit trials and display facecam video of licking 

	output_base_folder = make_output_dir(r'\\allen\programs\braintv\workgroups\nc-ophys\Matt\VB_QC\miss_licks')

	master_df = get_master_df()

	sample = np.random.choice(list(np.arange(0,len(master_df))), 100, replace=False)

	# pull 10 sessions, don't count sesions that can't load
	donors = []
	containers = []
	experiments = []
	nolick_indices = []
	lick_indices = []
	all_trial_dfs = []
	i=0

	for s,d in enumerate(sample):

		while i<10:

			temp_df = master_df.iloc[d:d+1].copy()
			try:
				session = get_session(temp_df)
				df, movie = get_video(temp_df)
				running_speed, running_speed_times = process_running(session)
			except:
				print('Cannot load data from ophys_experiment_id ' + str(temp_df['ophys_experiment_id'].values))
				continue

			win=(-0.5, 1.5)
			thresh = 200 # take this number of hte best fits NOT A THRESHOLD

			trial_df = make_trial_df(session, running_speed, running_speed_times, win=win)
			hit_df = trial_df[trial_df.hit==True]
			miss_df = trial_df[trial_df.miss==True]

			mse, hit_mat, miss_mat, template, rssamp_ts = compare_running(trial_df, hit_df, miss_df, win=win, resamp_win=120)

			if not (len(hit_df)>50) and (len(miss_df)>10):
				print('Not enough hits or misses: ' + str(len(hit_df)) + ' hits, ' + str(len(miss_df)) + ' misses')
				continue

			miss_tocheck, low_mse_idx, high_mse_idx = get_matching_running(mse, miss_df, thresh=thresh)

			plot_template_matches(trial_df, hit_df, miss_df, mse, hit_mat, miss_mat, template, rssamp_ts, thresh=thresh)
			plt.show()
			go_on = audit_running_template()

			if not go_on:
				continue

			if len(miss_tocheck)>5:
				tocheck = np.random.choice(list(miss_tocheck.values), 5, replace=False)
			else:
				tocheck = list(miss_tocheck.values)

			annotated_lick_inds = []
			annotated_no_lick_inds = []
			for f in tocheck:
				plot_stop_frames(movie, f,[],[])
				plt.show()
				annotate_lick(f, annotated_lick_inds, annotated_no_lick_inds)

			donors.append(temp_df['donor_id'])
			experiments.append(temp_df['ophys_experiment_id'])
			containers.append(temp_df['container_id'])
			lick_indices.append(annotated_lick_inds)
			nolick_indices.append(annotated_lick_inds)
			all_trial_dfs.append(trial_df)

			i+=1

	summary_df = pd.DataFrame()
	summary_df['donor_id'] = donors
	summary_df['experiment_id'] = experiments
	summary_df['container_id'] = containers
	summary_df['lick_indx'] = lick_indices
	summary_df['nolick_indx'] = nolick_indices
	summary_df['trial_df'] = all_trial_dfs

	output_filename = 'summary_df.pkl'
	output_filepath = os.path.join(output_base_folder, output_filename)
	summary_df.to_pickle(output_filepath)


def plot_specific_sessions(session_list):

	master_df = get_master_df()

	basepath = r'\\allen\programs\braintv\workgroups\nc-ophys\Matt\VB_QC\miss_licks'
	output_base_folder = make_output_dir(basepath)

	for s,session_path in enumerate(session_list):
		unix_path = '/allen/programs/braintv/production/visualbehavior/prod0'
		full_path = unix_path + '/' + session_path
		temp_df = master_df[master_df['storage_directory']==full_path].copy()

		try:
			session = get_session(temp_df)
			df, movie = get_video(temp_df)
			running_speed, running_speed_times = process_running(session)
		except:
			print('Cannot load data from ophys_experiment_id ' + str(temp_df['ophys_experiment_id'].values))
			continue

		win=(-0.5, 1.5)
		thresh = 1.0 # take this number of the best fits

		trial_df = make_trial_df(session, running_speed, running_speed_times, win=win)
		hit_df = trial_df[trial_df.hit==True]
		miss_df = trial_df[trial_df.miss==True]

		mse, hit_mat, miss_mat, template, rssamp_ts = compare_running(trial_df, hit_df, miss_df, win=win, resamp_win=120)

		if not (len(hit_df)>50) and (len(miss_df)>10):
			print('Not enough hits or misses: ' + str(len(hit_df)) + ' hits, ' + str(len(miss_df)) + ' misses')
			continue

		miss_tocheck, low_mse_idx, high_mse_idx = get_matching_running(mse, miss_df, thresh=thresh)

		plot_template_matches(trial_df, hit_df, miss_df, mse, hit_mat, miss_mat, template, rssamp_ts, thresh=thresh)

		labtracks_id = str(session.metadata['LabTracks_ID'])
		ophys_experiment_id = str(temp_df['ophys_experiment_id'].values[0])
		fig_savestring = labtracks_id + '_' + ophys_experiment_id
		output_filepath = os.path.join(output_base_folder,  fig_savestring+'.png')
		plt.gca()
		plt.savefig(output_filepath)

		plot_stop_frames(movie, miss_tocheck.values[0], labtracks_id, ophys_experiment_id)
		output_filepath_mov = os.path.join(output_base_folder,  fig_savestring+'_mov.png')
		plt.gca()
		plt.savefig(output_filepath_mov)


def plot_just_hits(session_list):

	master_df = get_master_df()

	basepath = r'\\allen\programs\braintv\workgroups\nc-ophys\Matt\VB_QC\miss_licks'
	output_base_folder = make_output_dir(basepath)

	for s,session_path in enumerate(session_list):
		unix_path = '/allen/programs/braintv/production/visualbehavior/prod0'
		full_path = unix_path + '/' + session_path
		temp_df = master_df[master_df['storage_directory']==full_path].copy()

		try:
			session = get_session(temp_df)
			df, movie = get_video(temp_df)
			running_speed, running_speed_times = process_running(session)
		except:
			print('Cannot load data from ophys_experiment_id ' + str(temp_df['ophys_experiment_id'].values))
			continue

		trial_df = make_trial_df(session, running_speed, running_speed_times, win=(-0.5, 1.5))
		hit_df = trial_df[trial_df.hit==True]

		labtracks_id = str(session.metadata['LabTracks_ID'])
		ophys_experiment_id = str(temp_df['ophys_experiment_id'].values[0])
		fig_savestring = labtracks_id + '_' + ophys_experiment_id
		output_filepath_mov = os.path.join(output_base_folder,  fig_savestring+'_mov.png')

		plot_stop_frames(movie, hit_df['change_time'].iloc[10], labtracks_id, ophys_experiment_id)
		plt.gca()
		plt.savefig(output_filepath_mov)


def compare_missed_licks_dprime(session_list, thresh=1.0):

	master_df = get_master_df()

	basepath = r'\\allen\programs\braintv\workgroups\nc-ophys\Matt\VB_QC\miss_licks'
	output_base_folder = make_output_dir(basepath)

	for s,session_path in enumerate(session_list):
		unix_path = '/allen/programs/braintv/production/visualbehavior/prod0'
		full_path = unix_path + '/' + session_path
		temp_df = master_df[master_df['storage_directory']==full_path].copy()

		try:
			session = get_session(temp_df)
			df, movie = get_video(temp_df)
			running_speed, running_speed_times = process_running(session)
		except:
			print('Cannot load data from ophys_experiment_id ' + str(temp_df['ophys_experiment_id'].values))
			continue

		win=(-0.5, 1.5)

		trial_df = make_trial_df(session, running_speed, running_speed_times, win=win)
		hit_df = trial_df[trial_df.hit==True]
		miss_df = trial_df[trial_df.miss==True]

		mse, hit_mat, miss_mat, template, rssamp_ts = compare_running(trial_df, hit_df, miss_df, win=win, resamp_win=120)

		if not (len(hit_df)>50) and (len(miss_df)>10):
			print('Not enough hits or misses: ' + str(len(hit_df)) + ' hits, ' + str(len(miss_df)) + ' misses')
			continue

		miss_tocheck, low_mse_idx, high_mse_idx = get_matching_running(mse, miss_df, thresh=thresh)

		return len(low_mse_idx), session




exp_list = ['specimen_823826986/ophys_session_852794147/',
    'specimen_814111935/ophys_session_846605051/',
    'specimen_823826986/ophys_session_858863712/',
    'specimen_814111935/ophys_session_842752650/',
    'specimen_813703544/ophys_session_845219209/',
    'specimen_834823477/ophys_session_878918807/',
    'specimen_784057626/ophys_session_833812106/',
    'specimen_847076524/ophys_session_894204946/',
    'specimen_803258386/ophys_session_848253761/']

plot_specific_sessions(exp_list)
#plot_just_hits(exp_list)