import pandas as pd
from allensdk.internal.api import PostgresQueryMixin
from allensdk.brain_observatory.behavior import behavior_ophys_session as bos
from allensdk.internal.api import behavior_ophys_api as boa
from multiprocessing import Pool
from functools import partial
import numpy as np

"""
Nick Ponvert, Allen Institute for Brain Science
"""

def make_lims_df():
	api = PostgresQueryMixin()
	query = '''
		SELECT
		vbc.id as container_id,
		vbc.workflow_state as container_workflow_state,
		d.id as donor_id,
		d.full_genotype,
		genders.name as sex,
		bs.created_at,
		bs.id as behavior_session_id,
		e.name as equipment_name,
		os.name as ophys_session_name,
		oe.id as ophys_experiment_id,
		os.storage_directory,
		oe.workflow_state as ophys_workflow_state,
		imaging_depths.depth as imaging_depth,
		st.acronym as targeted_structure
		FROM visual_behavior_experiment_containers vbc
		JOIN specimens sp ON sp.id=vbc.specimen_id
		JOIN donors d ON d.id=sp.donor_id
		JOIN genders ON genders.id = d.gender_id
		JOIN behavior_sessions bs ON bs.donor_id = d.id
		JOIN equipment e on e.id = bs.equipment_id
		LEFT JOIN ophys_sessions os on os.foraging_id = bs.foraging_id
		LEFT JOIN projects p ON p.id=os.project_id
		LEFT JOIN ophys_experiments oe on oe.ophys_session_id = os.id
		LEFT JOIN structures st ON st.id=oe.targeted_structure_id
		LEFT JOIN imaging_depths ON imaging_depths.id=oe.imaging_depth_id
		WHERE e.name in ('CAM2P.3', 'CAM2P.4', 'CAM2P.5')
		AND vbc.workflow_state in ('completed', 'postprocessing', 'container_qc')
		AND os.name is not NULL
		'''

	lims_df = pd.read_sql(query, api.get_connection())
	return lims_df

def parallelize(data, func, num_of_processes):
	data_split = np.array_split(data, num_of_processes)
	pool = Pool(num_of_processes)
	data = pd.concat(pool.map(func, data_split))
	pool.close()
	pool.join()
	return data

def run_on_subset(func, data_subset):
	return data_subset.apply(func, axis=1)

def parallelize_on_rows(data, func, num_of_processes=8):
	return parallelize(data, partial(run_on_subset, func), num_of_processes)

def get_stage_name(row):
	api = boa.BehaviorOphysLimsApi(int(row['ophys_experiment_id']))
	session = bos.BehaviorOphysSession(api)
	try:
		stage_name = api.get_task_parameters()['stage']
	except Exception as e:
		print("Load error")
		print(e)
		return "Load error"
	else:
		print(stage_name)
		return stage_name

def run():
	lims_df = make_lims_df()
	lims_df['stage_name'] = parallelize_on_rows(lims_df, get_stage_name)
	# Load errors happen for RF mapping sessions
	behavior_sessions = lims_df.query("stage_name != 'Load error'").copy()
	# Need to figure out what the 'retake number' is for each type of session
	mouse_gb = behavior_sessions.groupby('donor_id')
	unique_mice = behavior_sessions['donor_id'].unique()
	for mouse_id in unique_mice:
		mouse_df = mouse_gb.get_group(mouse_id)
		stage_gb = mouse_df.groupby('stage_name')
		unique_stages = mouse_df['stage_name'].unique()
		for stage_name in unique_stages:
			# Get the sessions for this stage and sort by the date
			sessions_this_stage = stage_gb.get_group(stage_name).sort_values('created_at')
			# Iterate through the sorted sessions and save the index to the row
			for ind_enum, (ind_row, row) in enumerate(sessions_this_stage.iterrows()):
				behavior_sessions.at[ind_row, 'retake_number'] = ind_enum
	sessions_to_use = behavior_sessions.query("ophys_workflow_state == 'passed'").sort_values(['container_id', 'stage_name'])
	return sessions_to_use