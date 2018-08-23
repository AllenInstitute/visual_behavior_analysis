"""
Created on Saturday July 14 2018

@author: marinag
"""

import numpy as np
from scipy import stats
import pandas as pd


def get_nearest_frame(timepoint, timestamps):
    return int(np.nanargmin(abs(timestamps - timepoint)))


def get_trace_around_timepoint(timepoint, trace, timestamps, window, frame_rate):
    frame_for_timepoint = get_nearest_frame(timepoint, timestamps)
    lower_frame = frame_for_timepoint + (window[0] * frame_rate)
    upper_frame = frame_for_timepoint + (window[1] * frame_rate)
    trace = trace[int(lower_frame):int(upper_frame)]
    timepoints = timestamps[int(lower_frame):int(upper_frame)]
    return trace, timepoints


def get_mean_in_window(trace, window, frame_rate):
    return np.nanmean(trace[int(window[0] * frame_rate): int(window[1] * frame_rate)])


def get_sd_in_window(trace, window, frame_rate):
    return np.std(trace[int(window[0] * frame_rate): int(window[1] * frame_rate)])


def get_sd_over_baseline(trace, response_window, baseline_window, frame_rate):
    baseline_std = get_sd_in_window(trace, baseline_window, frame_rate)
    response_mean = get_mean_in_window(trace, response_window, frame_rate)
    return response_mean / (baseline_std)


def get_p_val(trace, response_window, frame_rate):
    response_window_duration = response_window[1] - response_window[0]
    baseline_end = int(response_window[0] * frame_rate)
    baseline_start = int((response_window[0] - response_window_duration) * frame_rate)
    stim_start = int(response_window[0] * frame_rate)
    stim_end = int((response_window[0] + response_window_duration) * frame_rate)
    (_, p) = stats.f_oneway(trace[baseline_start:baseline_end], trace[stim_start:stim_end])
    return p


def ptest(x, num_conditions):
    ptest = len(np.where(x < (0.05 / num_conditions))[0])
    return ptest


def get_mean_sem(group):
    mean_response = np.mean(group['mean_response'])
    sem_response = np.std(group['mean_response'].values) / np.sqrt(len(group['mean_response'].values))
    mean_trace = np.mean(group['trace'])
    sem_trace = np.std(group['trace'].values) / np.sqrt(len(group['trace'].values))
    return pd.Series({'mean_response': mean_response, 'sem_response': sem_response,
                      'mean_trace': mean_trace, 'sem_trace': sem_trace})

def annotate_trial_response_df_with_pref_stim(rdf):
    rdf['pref_stim'] = False
    mean_response = rdf.groupby(['cell', 'change_image_name']).apply(get_mean_sem)
    m = mean_response.unstack()
    for cell in m.index:
        image_index = np.where(m.loc[cell]['mean_response'].values==np.max(m.loc[cell]['mean_response'].values))[0][0]
        pref_image = m.loc[cell]['mean_response'].index[image_index]
        trials = rdf[(rdf.cell==cell)&(rdf.change_image_name==pref_image)].index
        for trial in trials:
            rdf.loc[trial,'pref_stim'] = True
    return rdf

def annotate_mean_df_with_pref_stim(mean_df):
    mdf = mean_df.reset_index()
    mdf['pref_stim'] = False

    for cell in mdf.cell.unique():
        mc =  mdf[(mdf.cell==cell)]
        pref_image = mc[(mc.mean_response==np.max(mc.mean_response.values))].change_image_name.values[0]
        row = mdf[(mdf.cell==cell)&(mdf.change_image_name==pref_image)].index
        mdf.loc[row,'pref_stim'] = True
    return mdf

def get_fraction_significant_trials(group):
    fraction_significant_trials = len(group[group.p_value<0.005])/float(len(group))
    return pd.Series({'fraction_significant_trials': fraction_significant_trials})

def get_fraction_responsive_trials(group):
    fraction_responsive_trials = len(group[group.mean_response>0.1])/float(len(group))
    return pd.Series({'fraction_responsive_trials': fraction_responsive_trials})


def get_mean_df(trial_response_df, conditions=['cell', 'change_image_name']):
    rdf = trial_response_df.copy()

    mdf = rdf.groupby(conditions).apply(get_mean_sem)
    mdf = mdf[['mean_response', 'sem_response', 'mean_trace', 'sem_trace']]
    mdf = mdf.reset_index()
    mdf = annotate_mean_df_with_pref_stim(mdf)

    fraction_significant_trials = rdf.groupby(conditions).apply(get_fraction_significant_trials)
    fraction_significant_trials = fraction_significant_trials.reset_index()
    mdf['fraction_significant_trials'] = fraction_significant_trials.fraction_significant_trials

    fraction_responsive_trials = rdf.groupby(conditions).apply(get_fraction_responsive_trials)
    fraction_responsive_trials = fraction_responsive_trials.reset_index()
    mdf['fraction_responsive_trials'] = fraction_responsive_trials.fraction_responsive_trials

    return mdf

