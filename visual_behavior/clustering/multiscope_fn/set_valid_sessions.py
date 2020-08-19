#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
After running run_convert_level_1_to_level_2_mesoscope_fn.py, run the code below, to set validity log for the sessions.

This code will save a file named "valid_sessions_xxxxx.pkl" in directory "/allen/programs/braintv/workgroups/nc-ophys/Farzaneh/ValidSessions".
The file will have vars related to the list of sessions and valid sessions.

Valid sessions are set by removing some old (bad) sessions, test/ophys7 sessions; and ignoring some failed QC measures (if desired).

Once you get the list of sessions, you can load it in your analysis codes; e.g. see "omissions_traces_peaks_init.py"


Created on Thu Aug  1 15:07:27 2019
@author: farzaneh
"""

from mouse_seeks_flags_FN import * # it has the key function all_fails
from def_funs_general import *
from def_funs import *
import pickle
import os
import socket
# if 1, some vars related to list of sessions will be saved to a pickle file on the server
save_valid_vars = 1

remove_noBehSess = 1  # remove sessions that don't have behavioral data
# ["VisualBehaviorMultiscope" , "MesoscopeDevelopment"]
projects = ["VisualBehaviorMultiscope"]


# %%
#from sklearn.metrics import roc_curve, auc

if socket.gethostname() == 'OSXLT1JHD5.local':  # allen mac
    dirAna = "/Users/farzaneh.najafi/Documents/analysis_codes/"
    dir0 = '/Users/farzaneh.najafi/OneDrive - Allen Institute/Analysis'

elif socket.gethostname() == 'ibs-farzaneh-ux2':  # allen pc
    dirAna = "/home/farzaneh/Documents/analysis_codes/"
    dir0 = '/home/farzaneh/OneDrive/Analysis'

os.chdir(dirAna)

#import imaging_decisionMaking_exc_inh.utils.lassoClassifier.crossValidateModel as crossValidateModel

dirMs = os.path.join(dirAna, 'multiscope_fn')

#exec(open(os.path.join(dirMs, "def_funs.py")).read())
#exec(open(os.path.join(dirMs, "def_funs_general.py")).read())
#exec(open(os.path.join(dirMs, "mouse_seeks_flags_FN.py")).read())
# execfile("def_funs.py")

# Instuctions here:
#    https://stackoverflow.com/questions/37006114/anaconda-permanently-include-external-packages-like-in-pythonpath
# in terminal: conda-develop /home/farzaneh/Documents/analysis_codes/multiscope_fn/


dir_server_me = '/allen/programs/braintv/workgroups/nc-ophys/Farzaneh'

# make valid_sess dir
dir_valid_sess = os.path.join(dir_server_me, 'ValidSessions')
if not os.path.exists(dir_valid_sess):
    os.makedirs(dir_valid_sess)

now = (datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")


# %%
################################################################################################################
################################################################################################################
############## Set the list of all sessions (excluding ophys7 and test sessions) until 7 days ago ##############
################################################################################################################
################################################################################################################

# %% Get list of mesoscope sessions

list_all_sessions, list_sessions_date, list_sessions_experiments = get_all_multiscope_exp(
    projects)

list_all_sessions0 = np.array(list_all_sessions)
list_sessions_date0 = np.array(list_sessions_date)
list_sessions_experiments0 = np.array(list_sessions_experiments)

print(len(list_all_sessions0))


# check the number of experiments per session; at this point it's ok if we get some sessions with other than 8 experiments; because we are going to remove some sessions below; but we will run this check again!
a = np.array([len(list_sessions_experiments0[i, 0])
              for i in range(len(list_sessions_experiments0))])
print(np.unique(a))
for i in range(len(a)):
    if a[i] != 8:
        print(
            f'Something is wrong with session {i}, id: {list_all_sessions0[i]}. We cant have {a[i]} experiments!\n')


# %%
################################################################################################################
############## Start removing bad sessions from list_all_sessions ##############
################################################################################################################

# %%
list_all_sessions = list_all_sessions0
list_sessions_date = list_sessions_date0
list_sessions_experiments = list_sessions_experiments0

print(len(list_all_sessions))


# %% Sort sessions based on their id

s_i = np.argsort(list_all_sessions)

list_all_sessions = list_all_sessions[s_i]
list_sessions_date = list_sessions_date[s_i]
list_sessions_experiments = list_sessions_experiments[s_i]

# print(list_all_sessions.shape)

# list_all_sessions = list_all_sessions0


# %% Analyze experiments recorded prior to 7 days ago (to give them enough time for preprocessing)

daysBefore = 7
latest_day_toAn = (datetime.datetime.now() -
                   datetime.timedelta(days=daysBefore)).strftime("%Y%m%d")
days_inds_toAn = [datetime.datetime.strptime(list_sessions_date[i], "%Y-%m-%d").strftime(
    "%Y%m%d") < latest_day_toAn for i in range(len(list_sessions_date))]

list_all_sessions = np.array(list_all_sessions)[days_inds_toAn]
list_sessions_date = np.array(list_sessions_date)[days_inds_toAn]
list_sessions_experiments = np.squeeze(
    np.array(list_sessions_experiments)[days_inds_toAn])

print(list_all_sessions.shape)


# %%
################################################################################################################
############## REMOVE BAD SESSIONS ##############
################################################################################################################

# %% Remove very old experiments with different version of the stimulus

old_id = 775366121
list_sessions_date = list_sessions_date[list_all_sessions > old_id]
list_sessions_experiments = list_sessions_experiments[list_all_sessions > old_id]
list_all_sessions = list_all_sessions[list_all_sessions > old_id]

print(list_all_sessions.shape)


# %% Remove sessions that don't have behavioral data (based on the spreadsheet that Sam provided.)

if remove_noBehSess:
    no_behavior_sessions = [787091911, 811072245, 814281477, 816129916, 825313264,
                            825891463, 826699824, 827312519, 829037311, 834404531, 836986885, 838105133, 838938867]

    list_sessions_date = list_sessions_date[~np.in1d(
        list_all_sessions, no_behavior_sessions)]
    list_sessions_experiments = list_sessions_experiments[~np.in1d(
        list_all_sessions, no_behavior_sessions)]
    list_all_sessions = list_all_sessions[~np.in1d(
        list_all_sessions, no_behavior_sessions)]

    print(list_all_sessions.shape)


# %% Remove the following sessions, while waiting for a solution from LIMS

# 857040020 : dff_trace shape and max_downsamp image missing	"** experiments 0 and 2 : error about dff_traces shape... no valid ROIs, should fail QC.
# 869456991 : workflow state not in qc
# 807249534 : workflow state not in qc
# 882756028 : It was marked as 'Failed' by Sam last month. In his words, "This definitely looks like the offsets were not set correctly in scanimage when the session was taken, so I think we can just mark this session as a failure. "
# 873720614: frame duration is 178ms!! (instead of 93ms)
err_notSureWhatToDo = [807249534, 857040020,
                       869456991, 873720614, 881094781, 882060185, 882756028]

list_sessions_date = list_sessions_date[~np.in1d(
    list_all_sessions, err_notSureWhatToDo)]
list_sessions_experiments = list_sessions_experiments[~np.in1d(
    list_all_sessions, err_notSureWhatToDo)]
list_all_sessions = list_all_sessions[~np.in1d(
    list_all_sessions, err_notSureWhatToDo)]

print(list_all_sessions.shape)


# %% Set the "stage" of all sessions

stage_mongo_all = np.array([get_stage_mongo(int(session_id))
                            for session_id in list_all_sessions])

su = np.unique(stage_mongo_all)
for i in range(len(su)):
    print('%s' % su[i])

#import sys
# if sys.version_info[0]>2:
#    print(*np.unique(stage_mongo_all), sep='\n')

# some session stages don't come out correctly (because they have weird names, and your get_stage_mongo code doesn't catch them)
weirdNameSess = np.in1d(
    list_all_sessions, [946294865, 958590885, 978224723, 929423904, 940448261])
# call them testOr7
stage_mongo_all[weirdNameSess] = 'testOr7'

# Find sessions with stages to be removed (ophys7 and testing)
# ['Ophys7', 'ophys7', '7', '7RF', 'test', 'testing9']
stages_to_remove = ['7', 'test', 'testOr7']
noAnalysStages = np.array([any([(stg.find(rmv) != -1)
                                for rmv in stages_to_remove]) for stg in stage_mongo_all])
print('%d sessions are ophys7 or test' % sum(noAnalysStages))

# noAnalysStages = np.in1d(stage_mongo_all, stages_to_remove) # this looks for the exact string as opposed to a pattern...
#noAnalysStages = validity_log_all[np.in1d(validity_log_all.stage_mongo.values, ['Ophys7', '7', '7RF'])].session_id
#print('%d sessions are ophys7' %len(np.unique(noAnalysStages)))


# %% Remove Ophys7 and test sessions

list_sessions_date = list_sessions_date[~noAnalysStages]
list_sessions_experiments = list_sessions_experiments[~noAnalysStages]
list_all_sessions = list_all_sessions[~noAnalysStages]

print(list_all_sessions.shape)

# also ophys7, but stage mongo is weirdly named, so it doesnt get caught in get_stage_mongo
# 929423904

# %% Remove sessions with platform_info error
# Marina: these sessions are all RF mapping sessions (Ophys7), and their stimulus files are different. So that's why platform info is missing.

'''
err_platform = [845444695, 846652517, 851740017, 856201876, 875508749] # 846652517: sam: this will fail QC bc of damage to the eye.
err_platform_inds = [i for i in range(len(list_all_sessions)) if any(np.in1d(err_platform, list_all_sessions[i]))]

list_all_sessions = np.delete(list_all_sessions, err_platform_inds)
print(list_all_sessions.shape)
'''
list_sessions_experiments_copy = copy.deepcopy(list_sessions_experiments)


# %% Now again check the number of experiments per session; it cant e different than 8!

a = np.array([len(list_sessions_experiments_copy[i])
              for i in range(len(list_sessions_experiments_copy))])
print(np.unique(a))
for i in range(len(a)):
    if a[i] != 8:
        print(
            f'Something is wrong with session {i}, id: {list_all_sessions[i]}. We cant have {a[i]} experiments!\n')
        print(list_sessions_experiments_copy[i])
        if np.in1d([886130638, 914161594, 914224851], list_all_sessions[i]).any():
            print('it is one of the expected sessions... passing!\n')
        else:
            sys.exit('Why does this session have 16 experiments?!!')

# Sessions [886130638, 914161594, 914224851] have 16 experiments; we don't know why, but remove the last 8 experiments because those are empty! (Wayne contacted about it: 03/30/2020; corona19 era!)
sess16 = list_all_sessions[a>8]
for sess16i in sess16:
    isess = np.argwhere(list_all_sessions == sess16i).squeeze()
    list_sessions_experiments[isess] = list_sessions_experiments_copy[isess][:8]

a = np.array([len(list_sessions_experiments[i])
              for i in range(len(list_sessions_experiments))])
if any(a!=8):
    print(np.unique(a))
    sys.exit('There are still sessions with other than 8 experiments!') 

    
# %%
################################################################################################################
################################################################################################################
############## Set validity_log_all: a dataframe that can be used to find analyzeable sessions ##############
############## (sessions with a proper analysis folder, including df/f etc) ##############
################################################################################################################
################################################################################################################

# %% Set validity_log_all; this part takes some time!

#session_id = 901149889; Session_obj = LimsOphysSession(lims_id=session_id); list_mesoscope_exp = Session_obj.data_pointer['ophys_experiment_ids']; print(list_mesoscope_exp)

validity_log_all = pd.DataFrame([], columns=[
                                'session_id', 'lims_id', 'date', 'stage_metadata', 'stage_mongo', 'valid', 'log'])
exp_date_analyze = []  # if [], run the "is_session_valid" funciton on the session regardless of its experiment date; #20190605 # run the function is_session_valid only if the session was recorded after exp_date_analyze

# np.arange(92,len(list_all_sessions)): #np.array(list_all_sessions)[[0,1,2]]: #list_all_sessions:
for i in np.arange(0,len(list_all_sessions)): # 170
    print(f"Setting validity log for session {i+1}/{len(list_all_sessions)}")
    
    session_id = int(list_all_sessions[i])
    print(session_id)
    list_mesoscope_exp = list_sessions_experiments[i]
#    if session_id > 889944877:
    validity_log = is_session_valid(
        session_id, list_mesoscope_exp, exp_date_analyze)

    validity_log_all = validity_log_all.append(validity_log, ignore_index=True)

validity_log_all = validity_log_all.reindex(validity_log.columns, axis=1)
# print 'Analyzing %s to %s' %(validity_log_all.date.values[0], validity_log_all.date.values[-1])
# validity_log_all.iloc[np.arange(0,len(validity_log_all),8)]

validity_log_all0 = copy.deepcopy(validity_log_all)

#validity_log_all = copy.deepcopy(validity_log_all0)


# %% Investigate the invalid sessions

invalid_session_dates = np.unique(
    validity_log_all[validity_log_all['valid'].values == False].date.values)
invalid_session_ids = np.unique(
    validity_log_all[validity_log_all['valid'].values == False].session_id.values)

print('invalid_session_dates:\n%s\n' % invalid_session_dates)
print('invalid_session_ids:\n%s\n' % invalid_session_ids)
print('Total of %d invalid experiments (from %d sessions), out of %d total experiments' % (
    sum(validity_log_all['valid'].values == False), len(invalid_session_ids), len(validity_log_all)))
#s = validity_log_all[validity_log_all['valid']==False].stage_mongo
# print(list(s.values))

#valid_sessions = np.unique(validity_log_all[validity_log_all['valid']==True].session_id.values)

# print the log for why experiments are invalid
a = validity_log_all[validity_log_all['valid'] == False].log.values
#print('Invalid experiments log:\n%s\n' %np.unique(a))


# how many experiments and sessions are set to failed qc
failedQC_experiments_vlog = validity_log_all[np.logical_and(
    validity_log_all['valid'] == False, validity_log_all.workflow_state == 'failed')]
failed_sessions = np.unique(failedQC_experiments_vlog['session_id'].values)
print('%d experiments (from %d invalid sessions) have failed qc' %
      (len(failedQC_experiments_vlog), len(failed_sessions)))

# get all the ones that are invalid despite not having a failed qc
nonFailedQC_experiments_vlog = validity_log_all[np.logical_and(
    validity_log_all['valid'] == False, validity_log_all.workflow_state != 'failed')]
nf_sessions = np.unique(nonFailedQC_experiments_vlog['session_id'].values)
invalid_session_dates_qcNonFailed = np.unique(
    [validity_log_all[validity_log_all['session_id'] == nf_sessions[i]].date.values for i in range(len(nf_sessions))])
print('%d experiments (from %d invalid sessions) have non-failed qc' %
      (len(nonFailedQC_experiments_vlog), len(nf_sessions)))

if len(nf_sessions) > 0:
    print('\ninvalid_session_dates with non-failed qc experiments:\n%s\n' %
          invalid_session_dates_qcNonFailed)
    # invalid_session_ids_qcNonFailed = invalid_session_ids[~np.in1d(invalid_session_ids, failed_sessions)] # invalid_session_ids[np.in1d(invalid_session_ids, nf_sessions)]
    print('invalid_session_ids with non-failed qc experiments:\n%s\n' %
          nf_sessions)  # invalid_session_ids_qcNonFailed)

print('%d invalid sessions with non-failed qc experiments out of %d total sessions\n' %
      (len(nf_sessions), len(list_all_sessions)))

if len(nf_sessions) > 0:
    print('(NOTE: the issue with these sessions should be resolved!\n\n')
else:
    print('Good!\n\n')


# %% Save to a text file the failure reason for each experiment that was failed due to a reason other than failed-QC... only show the reason for the 1st experiment of the session.

# for i in failed_sessions:

# save the text file name
fn = r"Invalid_experiments_log_%s.txt" % now
fn = os.path.join(dir_valid_sess, fn)

file1 = open(fn, "a")

str01 = 'Total of %d invalid experiments (from %d sessions), out of %d total experiments (ie %d sessions)\n\n' % (sum(
    validity_log_all['valid'].values == False), len(invalid_session_ids), len(validity_log_all), len(validity_log_all)/8.)
str02 = '%d experiments (from %d invalid sessions) have failed qc\n' % (
    len(failedQC_experiments_vlog), len(failed_sessions))
str03 = '%d experiments (from %d invalid sessions) have non-failed qc\n\n' % (
    len(nonFailedQC_experiments_vlog), len(nf_sessions))
str00 = '%d invalid_session_ids with non-failed qc experiments:\n%s\n\n' % (
    len(nf_sessions), nf_sessions)

file1.write(str01+str02+str03+str00)

for i in nf_sessions:  # i = 2

    str0 = '------------------------------------------------------------------------------------------------------------------------------\n'

    file1.write(str0)

    print(str0)
    thiss = validity_log_all[validity_log_all['session_id'] == i]
#    print(thiss)

    elist = thiss['lims_id'].values
    vlist = thiss['valid'].values
    fexp = elist[vlist == False]

    str1 = 'Session_id: %d, date: %s\n' % (i, thiss['date'].iloc[0])
    print(str1)

    for iff in range(len(fexp)):  # iff = 0
        str2 = 'Invalid experiment: %d\n' % fexp[iff]

        ie = thiss['lims_id'].values == fexp[iff]

        ws = thiss['workflow_state'].iloc[ie].values[0]
        err = thiss['log'].iloc[ie].values[0]

        str3 = 'workflow_state: %s\n' % ws
        str4 = 'errors:\n%s\n%s\n%s\n%s\n%s\n\n\n' % (
            err[0], err[1], err[2], err[3], err[4])

        print(str2)
        print(str3)
        print(str4)

        stra = str1 + str2 + str3 + str4
        file1.write(stra)

#    print([list(validity_log_all[validity_log_all['session_id']==i].lims_id.values)])
#    print([list(validity_log_all[validity_log_all['session_id']==i]['valid'].values)])
#    print(thiss['log'].iloc[0])
#    print(thiss['log'].iloc[0])
#    print(thiss.iloc[0])

#    input()

file1.close()


# use this if you want to manually reset valid values for an experiment:
# validity_log_all.loc[validity_log_all['session_id']==i, 'valid'].iloc[4] = True


# %%
################################################################################################################
############## Set failedQC: a dataframe that shows failure tags for experiments that failed QC ##############
############## (we will later use it to decide which failedQC measures to ignore or not) ##############
################################################################################################################


# %% Get the failure tags for experiments with failed QC. So we can decide which ones to keep for analysis, and which ones to exclude.
# Get failedQC dataframe (that shows a bunch of information (including failure tags) for all experiments of all sessions that have at least one failed experiment)

failedQC = all_fails(failed_sessions)  # a function in mouse_seeks_flags_FN
failedQC_allExperiments = failedQC[failedQC.experiments_qc == 'failed']

#failedQC_allExperiments.to_hdf('~/Desktop/failedQC_allExperiments.h5', key='df', format='fixed')


# %% Print some information about why each session has failed the QC

# Show all the failed QC experiments and why they failed
'''
display([[failedQC_allExperiments.sessions_id.iloc[i], failedQC_allExperiments.experiments_id.iloc[i], \
          failedQC_allExperiments.zdrift.iloc[i], failedQC_allExperiments.intensity_drift.iloc[i], \
          failedQC_allExperiments.nb_saturated_pixels.iloc[i], failedQC_allExperiments.overrides_metric.iloc[i], \
          failedQC_allExperiments.overrides_notes.iloc[i], failedQC_allExperiments['failure_tag'].iloc[i]] for i in range(len(failedQC_allExperiments))])


# Look at failed experiments one by one, to see why they failed
for i in np.arange(0,len(failedQC_allExperiments)): # range(len(failedQC_allExperiments)): #
    print('_______ %d _______' %i)
    display([failedQC_allExperiments.sessions_id.iloc[i], failedQC_allExperiments.experiments_id.iloc[i], \
             failedQC_allExperiments.zdrift.iloc[i], failedQC_allExperiments.intensity_drift.iloc[i], \
             failedQC_allExperiments.nb_saturated_pixels.iloc[i], failedQC_allExperiments.overrides_metric.iloc[i], \
             failedQC_allExperiments.overrides_notes.iloc[i], failedQC_allExperiments['failure_tag'].iloc[i]])
    input()
'''

# Show the failed QC sessions and why they failed
# the index of the first experiment of the variable failedQC
aa = [np.argwhere(failedQC.sessions_id.values - np.unique(failedQC.sessions_id.values)
                  [i] == 0).flatten()[0] for i in range(len(np.unique(failedQC.sessions_id.values)))]
# columns=['experiment_id', 'failure_tag'],
failedQC_failureTag = pd.DataFrame(
    {'sessions_id': failedQC.iloc[aa].sessions_id.values, 'failure_tag': failedQC.iloc[aa]['failure_tag'].values}, index=aa)
# 'num_failed_experiments': # add sum of failed experiments for each session
# 'overrides_metric':failedQC.iloc[aa].sessions_id.values
# 'overrides_notes':failedQC.iloc[aa].sessions_id.values
display([[failedQC_failureTag.sessions_id.iloc[i], failedQC_failureTag['failure_tag'].iloc[i]]
         for i in range(len(failedQC_failureTag))])


# Print of list of failure tags
"""
print('\nReasons for failure:')
display(list(np.unique(np.concatenate((failedQC_failureTag['failure_tag'].values)))))
"""

# Reasons for failure:
all_failure_reasons = ['z_drift_corr_um_diff',
                       'parent_averaged_depth_image',
                       'parent_averaged_surface_image',
                       'epilepsy_probability',
                       'target_match',
                       'num_contingent_trials',
                       'd_prime_peak',
                       'percent_change_intensity',
                       'nb_saturated_pixels',
                       'nb_dropped_eye_tracking_frames',
                       'eye_tracking_global_mean']  # image of the eye, e.g. stress foam makes it invalid


#%% ############ Print a summary of how many sessions failed for what qc measure ############

all_measures = 'zdrift', 'parent_depth_match', 'parent_surface_match', 'epilepsy', 'target_match', 'num_contingent_trials', 'd_prime_peak', 'intensity_change', 'saturated_pixels', 'dropped_eye_tracking_frames', 'stress_foam'

# The following sessions have truly failed QC (I know this by having gone through these sessions manually)
# 881328283  # 882386411 : questionable
# These sessions : sync data is missing : [898881223, 899181497, 900209262]
err_truly_failedQC = [881328283, 882386411]
#err_sync_data_missing = [898881223, 899181497, 900209262, 901149889]
# np.concatenate((err_truly_failedQC, err_sync_data_missing))
err_all = err_truly_failedQC

file1 = open(fn, "a")
file1.write(
    '\n------------------- % Failed experiments for each QC measure -------------------\n')

for ignore_inds in range(len(all_measures)):
    all_measures = 'zdrift', 'parent_depth_match', 'parent_surface_match', 'epilepsy', 'target_match', 'num_contingent_trials', 'd_prime_peak', 'intensity_change', 'saturated_pixels', 'dropped_eye_tracking_frames', 'stress_foam'
#    print('____________________________')
    mname = all_measures[ignore_inds]
#    print(mname)

    # For each experiment loop through all failure tags, to see if the failure tag belongs to the ignore_measures
    ignore_experiments = []
    for iexp in range(len(failedQC_allExperiments)):
        ign = 0
        if np.in1d(failedQC_allExperiments.iloc[iexp].sessions_id, err_all) == False:
            for ireason in range(len(failedQC_allExperiments['failure_tag'].iloc[iexp])):

                # did the experiment fail for any of the following reasons?
                zdrift = failedQC_allExperiments['failure_tag'].iloc[iexp][ireason].find(
                    'z_drift')
                parent_depth_match = failedQC_allExperiments['failure_tag'].iloc[iexp][ireason].find(
                    'parent_averaged_depth')
                parent_surface_match = failedQC_allExperiments['failure_tag'].iloc[iexp][ireason].find(
                    'parent_averaged_surface')

                epilepsy = failedQC_allExperiments['failure_tag'].iloc[iexp][ireason].find(
                    'epilepsy')
                target_match = failedQC_allExperiments['failure_tag'].iloc[iexp][ireason].find(
                    'target_match')

                num_contingent_trials = failedQC_allExperiments['failure_tag'].iloc[iexp][ireason].find(
                    'num_contingent_trials')
                d_prime_peak = failedQC_allExperiments['failure_tag'].iloc[iexp][ireason].find(
                    'd_prime_peak')

                intensity_change = failedQC_allExperiments['failure_tag'].iloc[iexp][ireason].find(
                    'change_intensity')
                saturated_pixels = failedQC_allExperiments['failure_tag'].iloc[iexp][ireason].find(
                    'saturated_pixels')
                dropped_eye_tracking_frames = failedQC_allExperiments['failure_tag'].iloc[iexp][ireason].find(
                    'dropped_eye')
                stress_foam = failedQC_allExperiments['failure_tag'].iloc[iexp][ireason].find(
                    'eye_tracking_global_mean')

                all_measures = np.array([zdrift, parent_depth_match, parent_surface_match, epilepsy, target_match, num_contingent_trials,
                                         d_prime_peak, intensity_change, saturated_pixels, dropped_eye_tracking_frames, stress_foam])

                # which measures to ignore
                ############ NOTE: you may need to change the following ############
    #            ignore_measures = zdrift, parent_depth_match, parent_surface_match, epilepsy, target_match
        #        ignore_measures = zdrift
        #        ignore_measures = parent_depth_match, parent_surface_match
    #            ignore_measures_beh = num_contingent_trials, d_prime_peak

                #######
                ignore_all = all_measures[ignore_inds]
    #            ignore_all = np.concatenate((ignore_measures, ignore_measures_beh))
        #        ignore_all = ignore_measures
        #        ignore_all = ignore_measures_beh

                if any(np.in1d(ignore_all, 0)):
                    ign = 1

            if ign == 1:
                ignore_experiments.append(
                    failedQC_allExperiments.iloc[iexp]['experiments_id'])

    strr = '\n%d %% %s (%d / %d QC-failed experiments)' % (100*len(ignore_experiments)/len(
        failedQC_allExperiments), mname, len(ignore_experiments), len(failedQC_allExperiments))
    print(strr)

    file1.write(strr)

file1.close()


# %% Now decide what QC measure you want to ignore, so you can later reset validity log.

############ What measures to ignore ############
######################## (NOTE: you may change this later especially for zdrift and parent_match once they improve) ########################
all_measures = 'zdrift', 'parent_depth_match', 'parent_surface_match', 'epilepsy', 'target_match', 'num_contingent_trials', 'd_prime_peak', 'intensity_change', 'saturated_pixels', 'dropped_eye_tracking_frames', 'stress_foam'
ignore_inds = np.array([5, 6]) # behavioral performance : 'num_contingent_trials', 'd_prime_peak'
ignore_inds = np.array([1, 2, 4]) # parent and target match
ignore_inds = np.array([1, 2, 4, 5, 6]) # parent and target match + behavioral performance
# ignore_inds = np.array([0, 1, 2, 4, 5, 6]) # zdrift + parent and target match + behavioral performance
# ignore_inds = np.array([0, 1, 2, 3, 4, 5, 6])  # from the list above # zdrift + epilepsy + parent and target match + behavioral performance
print('The following measure will be ignored:')
print(list(np.array(all_measures)[ignore_inds]))
# Remember: some experiments have failed qc their sync data is missing. The failure tags for them is empty ... so they will remain failed because they will not be ignored below.


# The following sessions have truly failed QC (I know this by having gone through these sessions manually)
# 881328283  # 882386411 : questionable
# These sessions : sync data is missing : [898881223, 899181497, 900209262]
#err_truly_failedQC = [881328283 , 882386411]
#err_sync_data_missing = [898881223, 899181497, 900209262, 901149889]
# err_all = err_truly_failedQC # np.concatenate((err_truly_failedQC, err_sync_data_missing))

# For each experiment loop through all failure tags, to see if the failure tag belongs to the ignore_measures
ignore_experiments = []
for iexp in range(len(failedQC_allExperiments)):
    # get the log for why the experiment was invalid; the first 4 logs are unrelated to workflow_state so if an experiment failed for those reasons, we should leave it invalid.
    eid = failedQC_allExperiments.iloc[iexp]['experiments_id']
    a = validity_log_all0[validity_log_all0['lims_id'].values==eid]['log'].values[0]
    leave_invalid = any([a[il]!='' for il in range(len(a)-1)])

    ign = 0
    if np.logical_and(leave_invalid==False, np.in1d(failedQC_allExperiments.iloc[iexp].sessions_id, err_all) == False):
        for ireason in range(len(failedQC_allExperiments['failure_tag'].iloc[iexp])):

            # did the experiment fail for any of the following reasons?
            zdrift = failedQC_allExperiments['failure_tag'].iloc[iexp][ireason].find(
                'z_drift')
            parent_depth_match = failedQC_allExperiments['failure_tag'].iloc[iexp][ireason].find(
                'parent_averaged_depth')
            parent_surface_match = failedQC_allExperiments['failure_tag'].iloc[iexp][ireason].find(
                'parent_averaged_surface')

            epilepsy = failedQC_allExperiments['failure_tag'].iloc[iexp][ireason].find(
                'epilepsy')
            target_match = failedQC_allExperiments['failure_tag'].iloc[iexp][ireason].find(
                'target_match')

            num_contingent_trials = failedQC_allExperiments['failure_tag'].iloc[iexp][ireason].find(
                'num_contingent_trials')
            d_prime_peak = failedQC_allExperiments['failure_tag'].iloc[iexp][ireason].find(
                'd_prime_peak')

            intensity_change = failedQC_allExperiments['failure_tag'].iloc[iexp][ireason].find(
                'change_intensity')
            saturated_pixels = failedQC_allExperiments['failure_tag'].iloc[iexp][ireason].find(
                'saturated_pixels')
            dropped_eye_tracking_frames = failedQC_allExperiments['failure_tag'].iloc[iexp][ireason].find(
                'dropped_eye')
            stress_foam = failedQC_allExperiments['failure_tag'].iloc[iexp][ireason].find(
                'eye_tracking_global_mean')

            all_measures = np.array([zdrift, parent_depth_match, parent_surface_match, epilepsy, target_match, num_contingent_trials,
                                     d_prime_peak, intensity_change, saturated_pixels, dropped_eye_tracking_frames, stress_foam])

            # which measures to ignore
            ############ NOTE: you may need to change the following ############
#            ignore_measures = zdrift, parent_depth_match, parent_surface_match, epilepsy, target_match
    #        ignore_measures = zdrift
    #        ignore_measures = parent_depth_match, parent_surface_match
#            ignore_measures_beh = num_contingent_trials, d_prime_peak

            #######
            ignore_all = all_measures[ignore_inds]
#            ignore_all = np.concatenate((ignore_measures, ignore_measures_beh))
    #        ignore_all = ignore_measures
    #        ignore_all = ignore_measures_beh

            if any(np.in1d(ignore_all, 0)):
                ign = 1

        if ign == 1:
            ignore_experiments.append(
                failedQC_allExperiments.experiments_id.iloc[iexp])


print('\n%d out of %d failed-QC experiments will be ignored.' %
      (len(ignore_experiments), len(failedQC_allExperiments)))


# %%
#################################################################################################################
############################# Reset validitiy_log_all according to ignore_sessions #############################
#################################################################################################################

# %% Set to valid the validity_log_all of ignore_experiments, so we can analyse them.

validity_log_all.loc[np.in1d(
    validity_log_all['lims_id'].values, ignore_experiments), 'valid'] = True
# sum(validity_log_all['valid']==False), sum(validity_log_all0['valid']==False)
print('\nFinal number of experiments to be excluded from analysis: %d' %
      sum(validity_log_all['valid'].values == False))

# original number:
#print('\nFinal number of experiments to be excluded from analysis: %d' %sum(validity_log_all0['valid'].values==False))


# %% Set the final list of valid sessions to be used for analysis

# the dataframe only for valid experiments
validity_log_all_onlyValid = validity_log_all[validity_log_all['valid'].values == True]
list_all_sessions_valid = np.unique(
    validity_log_all_onlyValid['session_id'].values)
list_all_experiments_valid = [validity_log_all_onlyValid[validity_log_all_onlyValid['session_id']
                                                         == list_all_sessions_valid[i]].lims_id.values for i in range(len(list_all_sessions_valid))]


print('Final number of experiments for analysis: %d' %
      len(validity_log_all_onlyValid))
print('Final number of sessions for analysis: %d' %
      len(list_all_sessions_valid))

# sanity check:
if np.equal(sum(validity_log_all['valid'].values == True) + len(failedQC_allExperiments) - len(ignore_experiments) + len(nonFailedQC_experiments_vlog),
            len(list_all_sessions)*8) == False:
    sys.exit('Numbers do not add up! check what is wrong!')


# %% Set list_all_experiments (for every valid session, each shows all experiment ids (valid or not))
# for sessions that dont have some experiment ids (because those experiments are not valid), add to list_all_experiment_ids the
# experiment id of the missing experiments. This is important to do otherwise, we wont be able to easily find the same plane across sessions.
# note: the experiments ids in list_all_experiments_valid are sorted (coming from LimsOphysSession), so we can be sure that index i
# in list_all_experiments_valid is always the same plane.

inds_not8planes = np.argwhere(
    np.array([len(i) for i in list_all_experiments_valid]) != 8).squeeze()
all_exp_sess_not8planes = [validity_log_all.iloc[validity_log_all['session_id'].values == list_all_sessions_valid[inds_not8planes[i]]].
                           lims_id.values for i in range(len(inds_not8planes))]
list_all_experiments = copy.deepcopy(list_all_experiments_valid)
for i in range(len(inds_not8planes)):
    list_all_experiments[inds_not8planes[i]] = all_exp_sess_not8planes[i]

# check
np.argwhere(np.array([len(i) for i in list_all_experiments_valid]) != 8).squeeze(),\
    np.argwhere(np.array([len(i)
                          for i in list_all_experiments]) != 8).squeeze()


# %% Save list of sessions and validity log

if save_valid_vars:
    dict_valid = {'list_all_sessions0': list_all_sessions0, 'list_sessions_date0': list_sessions_date0, 'list_sessions_experiments0': list_sessions_experiments0,
                  'validity_log_all0': validity_log_all0, 'validity_log_all': validity_log_all, 'list_all_sessions_valid': list_all_sessions_valid, 'list_all_experiments_valid': list_all_experiments_valid,
                  'list_all_experiments': list_all_experiments}

    # make valid_sess dir
#    dir_valid_sess = os.path.join(dir_server_me, 'ValidSessions')
#    if not os.path.exists(dir_valid_sess):
#        os.makedirs(dir_valid_sess)

    # save the vars
    fn = "valid_sessions_%s.pkl" % now
    # 'valid_sessions' + '.pkl')
    validSessName = os.path.join(dir_valid_sess, fn)
    print(validSessName)

    f = open(validSessName, 'wb')
    pickle.dump(dict_valid, f)
    f.close()


# Read the pickle file saved in the script: set_valid_sessions.py
#pkl = open(validSessName, 'rb')
#dictNow = pickle.load(pkl)
