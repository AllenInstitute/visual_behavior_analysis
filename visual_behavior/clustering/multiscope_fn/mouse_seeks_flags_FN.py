from ophysextractor.utils.util import mongo, get_psql_dict_cursor
import pandas as pd
import numpy as np


# %% LIMS query to get the list of failed experiments from a particular project

def fetch_records(project=None):
    # pulls information for visual coding or visual behavior experiments and returns as a database
    if project == 'vb':
        # All visual behavior experiments (if you want the failed ones add AND oe.workflow_state = ‘failed’)
        all_fail_query = '''
        SELECT os.id, os.date_of_acquisition , users.login AS operator
        FROM ophys_sessions os
        JOIN ophys_experiments oe ON oe.ophys_session_id = os.id
        JOIN projects p ON os.project_id = p.id
        JOIN users ON users.id = os.operator_id
        WHERE p.code = 'VisualBehavior'
        '''
    if project == 'vc':
        # All failed c600 experiments
        all_fail_query = '''
        SELECT os.id, os.date_of_acquisition , users.login AS operator, tag.name AS fail_tag
        FROM ophys_sessions os
        JOIN ophys_experiments oe ON oe.ophys_session_id = os.id
        JOIN projects p ON os.project_id = p.id
        JOIN users ON users.id = os.operator_id
        JOIN ophys_experiment_tags_ophys_experiments tid ON tid.ophys_experiment_id = oe.id
        JOIN ophys_experiment_tags tag ON tid.ophys_experiment_tag_id = tag.id
        WHERE p.code = 'c600'
        '''
    if project == 'meso':
        all_fail_query = '''
        SELECT os.id AS session_id, oe.id AS experiment_id, os.date_of_acquisition , users.login AS operator
        FROM ophys_sessions os
        JOIN ophys_experiments oe ON oe.ophys_session_id = os.id
        JOIN projects p ON os.project_id = p.id
        JOIN users ON users.id = os.operator_id
        WHERE p.code = 'VisualBehaviorMultiscope'
            AND oe.workflow_state = 'failed'
        '''
    lims_cursor = get_psql_dict_cursor()
    lims_cursor.execute(all_fail_query)
    records = lims_cursor.fetchall()
    return(records)


# %% Go to mongoDb and find information about each failed experiment

def all_fails(session_ids=None):  # records = None):

    cols = np.array(['sessions_id', 'experiments_id', 'experiments_qc', 'rig', 'operator', 'date', 'genotype', 'mouse', 'zdrift',
                     'intensity_drift', 'nb_saturated_pixels', 'flags_notes', 'flags_metric', 'overrides_notes', 'overrides_metric', 'failure_tag'])
    failedQC = pd.DataFrame([], columns=cols)

    # session_id = np.unique(session_ids)[0]
    for session_id in np.unique(session_ids):
        print(session_id)
        # ms_r = mongo.db.ophys_session_log.find({u'id': record[u'session_id']}).next()
        ms_r = mongo.db.ophys_session_log.find({u'id': session_id}).next()
#            print(ms_r.keys())

        # %% Loop over each experiment of a given session

        this_sess = pd.DataFrame([], columns=cols)
        for iexp in range(8):  # len(ms_r[u'ophys_experiments'])): # this is the correct way; the reason I am going with 8 is that 3 sessions are assigned 16 experiments (8 of which empty), so we cant go with len(ms_r) # iexp = 0

            exp_id = ms_r[u'ophys_experiments'][iexp]['id']
            print('\t%d %d' % (iexp, exp_id))
#                if ms_r[u'ophys_experiments'][iexp][u'workflow_state'] == 'failed': # we don't need this! already LIMS query has found these experiments as "failed".
            try:
                failure_sessions = ms_r[u'id']
                failure_experiments = exp_id
                failure_experiments_qc = ms_r[u'ophys_experiments'][iexp][u'workflow_state']
                operator = ms_r[u'operator']
                rig = ms_r[u'rig']
                date = ms_r[u'date_of_acquisition']
                genotype = ms_r[u'genotype']
                mouse = ms_r[u'external_specimen_name']

                this_sess.at[iexp, ['sessions_id', 'experiments_id', 'experiments_qc', 'rig', 'operator', 'date', 'genotype',
                                    'mouse']] = failure_sessions, failure_experiments, failure_experiments_qc, operator, rig, date, genotype, mouse

            except:
                pass

            met = mongo.qc.metrics.find({'lims_id': exp_id}).next()
            try:
                zdrift = met['local_z_stack']['z_drift_corr_um_diff']
            except:
                zdrift = np.nan
                pass

            try:
                intensity_drift = met['motion_corr_physio']['percent_change_intensity']
                nb_saturated_pixels = met['motion_corr_physio']['nb_saturated_pixels']
            except:
                intensity_drift = np.nan
                nb_saturated_pixels = np.nan
                pass

            this_sess.at[iexp, ['zdrift', 'intensity_drift', 'nb_saturated_pixels']
                         ] = zdrift, intensity_drift, nb_saturated_pixels

            ######### check overrides #########
#                if len(failure_tags_temp) == 0:
            otot_metric = []
            otot_notes = []
            try:
                # get experiment_ids from the override field. we do this, because we are not sure if index 0 in overrides actually means experiment index 0.
                ordies_ids = []
                for ioride in range(len(ms_r[u'overrides'])):
                    ordies_ids.append(
                        int(ms_r[u'overrides'][ioride]['lims_id']))

                # is there an override for the session?
                this_sessA = np.argwhere(
                    np.array(ordies_ids) == ms_r[u'id']).flatten()
                if len(this_sessA) > 0:
                    for this_sessi in this_sessA:
                        this_sessi = int(this_sessi)
                        otot_metric.append(
                            ms_r[u'overrides'][this_sessi][u'metric'])
                        otot_notes.append(
                            ms_r[u'overrides'][this_sessi][u'notes'])

                # is there an override for the experiment
                this_expA = np.argwhere(
                    np.array(ordies_ids) == ms_r[u'ophys_experiments'][iexp]['id']).flatten()
                if len(this_expA) > 0:
                    for this_exp in this_expA:
                        this_exp = int(this_exp)
#                            if len(ms_r[u'overrides'][this_exp][u'metric']) > 0:
                        otot_metric.append(
                            ms_r[u'overrides'][this_exp][u'metric'])
                        otot_notes.append(
                            ms_r[u'overrides'][this_exp][u'notes'])
#                overrides_metric.append(otot_metric) # add overrides for both the experiment and the session
#                overrides_notes.append(otot_notes)

            except:
                pass

            this_sess.at[iexp, ['overrides_notes',
                                'overrides_metric']] = otot_notes, otot_metric

            ######### check flags #########
#                if len(failure_tags_temp) == 0:
            otot_metric = []
            otot_notes = []
            try:
                #                    if ms_r[u'failure_tags'] == 0:
                #                        failure_tags_temp.append([ms_r[u'flags'][0][u'metric']])
                # get experiment_ids from the override field. we do this, because we are not sure if index 0 in overrides actually means experiment index 0.
                ordies_ids = []
                for ioride in range(len(ms_r[u'flags'])):
                    ordies_ids.append(int(ms_r[u'flags'][ioride]['lims_id']))

                this_sessA = np.argwhere(
                    np.array(ordies_ids) == ms_r[u'id']).flatten()
                if len(this_sessA) > 0:
                    for this_sessi in this_sessA:
                        this_sessi = int(this_sessi)
                        otot_metric.append(
                            ms_r[u'flags'][this_sessi][u'metric'])
                        otot_notes.append(ms_r[u'flags'][this_sessi][u'notes'])

                # is there an override for the experiment
                this_expA = np.argwhere(
                    np.array(ordies_ids) == ms_r[u'ophys_experiments'][iexp]['id']).flatten()
                if len(this_expA) > 0:
                    for this_exp in this_expA:
                        this_exp = int(this_exp)
#                            if len(ms_r[u'flags'][this_exp][u'metric']) > 0:
                        otot_metric.append(ms_r[u'flags'][this_exp][u'metric'])
                        otot_notes.append(ms_r[u'flags'][this_exp][u'notes'])
#                flags_metric.append(otot_metric) # add flags for both the experiment and the session
#                flags_notes.append(otot_notes)

            except:
                pass

            this_sess.at[iexp, ['flags_notes', 'flags_metric']
                         ] = otot_notes, otot_metric


#            NOTE: THE FAILURE TAGS PART NEEDS FIX ONCE IT IS FIXED ON MONGODB
#            (it is not clear right now which experiment the tags belong to!)
            ######### check failure tags #########
            # currently (as of 07/03/19) the problem with failure_tags is that experiment_id information is missing, so it is not easy to figure out what experiment the failure_tag belongs to.
            failure_tags = []
            try:
                if len(ms_r[u'failure_tags']) > 0:
                    failure_tags = ms_r[u'failure_tags']
            except:
                pass

            this_sess.at[iexp, ['failure_tag']] = [failure_tags]

        ############ Done with all experiments of a session ############
        try:
            failedQC = failedQC.append(this_sess)

        except Exception as e:
            print(session_id)
            raise(e)

    # %%
    return(failedQC)


# %%
"""    
records = fetch_records(project = 'meso') # record of all failed visBehMes experiments

session_ids = np.array([dict(records[i])['session_id'] for i in range(len(records))])    
failedQC = all_fails(session_ids)
"""
