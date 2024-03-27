from visual_behavior.ophys.dataset import visual_behavior_ophys_dataset as vbod
from visual_behavior.ophys.response_analysis import response_processing as rp
from sklearn.preprocessing import minmax_scale
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import sys
import os

cache_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_pilot_analysis/visual_behavior_pilot_manuscript_resubmission'
output_dir = r'/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/figures/ophys_pilot_manuscript_smoothing_comparison_figures'

oeid = int(sys.argv[1])

dataset = vbod.VisualBehaviorOphysDataset(experiment_id=oeid, cache_dir=cache_dir)
esp = dataset.extended_stimulus_presentations
events = esp.query('change')['start_time'].values
event_ids = esp.query('change').index.values
ophys_timestamps = dataset.ophys_timestamps
trace_arr = dataset.events_array
trace_ids = dataset.cell_specimen_ids
response_analysis_params = rp.get_default_stimulus_response_params()

response_xr_no_filter = rp.get_response_xr(dataset, 
                                           trace_arr, 
                                           ophys_timestamps, 
                                           events, 
                                           event_ids, 
                                           trace_ids, 
                                           response_analysis_params)

def response_xr_filtered(scale):
    filt = stats.halfnorm(loc=0, scale=scale).pdf(np.arange(20))
    
    filtered_arr = np.empty(trace_arr.shape)
    for ind_cell in range(trace_arr.shape[0]):
        this_trace = trace_arr[ind_cell, :]
        this_trace_filtered = np.convolve(this_trace, filt)[:len(this_trace)]
        filtered_arr[ind_cell, :] = this_trace_filtered

    response_xr_filtered = rp.get_response_xr(dataset, 
                                               filtered_arr, 
                                               ophys_timestamps, 
                                               events, 
                                               event_ids, 
                                               trace_ids, 
                                               response_analysis_params)
    return response_xr_filtered

filtered_6 = response_xr_filtered(6)
filtered_3 = response_xr_filtered(3)
filtered_2 = response_xr_filtered(2)

response_xr_dff = rp.get_response_xr(dataset, 
                                     dataset.dff_traces_array, 
                                     ophys_timestamps, 
                                     events, 
                                     event_ids, 
                                     trace_ids, 
                                     response_analysis_params)

sigma_colors = ['#ad7fa8' , '#f57900' , '#fce94f']
timebase = filtered_6.coords['eventlocked_timestamps'].values

for csid in trace_ids:
    plt.clf()
    plt.plot(
        timebase,
        minmax_scale(filtered_6.sel({'trace_id':csid})['eventlocked_traces'].mean(dim=['trial_id'])),
        label='filtered, $\sigma$=6',
        lw=2,
        color=sigma_colors[0],
        alpha=0.9
    )

    plt.plot(
        timebase,
        minmax_scale(filtered_3.sel({'trace_id':csid})['eventlocked_traces'].mean(dim=['trial_id'])),
        label='filtered, $\sigma$=3',
        lw=2,
        color=sigma_colors[1],
        alpha=0.8
    )

    plt.plot(
        timebase,
        minmax_scale(filtered_2.sel({'trace_id':csid})['eventlocked_traces'].mean(dim=['trial_id'])),
        label='filtered, $\sigma$=2',
        lw=2,
        color=sigma_colors[2],
        alpha=0.8
    )


    plt.plot(
        timebase,
        minmax_scale(response_xr_no_filter.sel({'trace_id':csid})['eventlocked_traces'].mean(dim=['trial_id'])),
        label='raw events',
        lw=2,
        color='0.5',
        alpha=0.6
        
    )

    plt.plot(
        timebase,
        minmax_scale(response_xr_dff.sel({'trace_id':csid})['eventlocked_traces'].mean(dim=['trial_id'])),
        label='dff',
        lw=2,
        color='g',
        alpha=0.6
    )
    plt.legend(loc=2)
    plt.ylabel('Scaled amplitude')
    plt.xlabel('Time from change image onset (s)')
    title_str = f'{dataset.analysis_folder}\ncsid: {csid}'
    fig_name_str = f'{dataset.analysis_folder}_csid: {csid}'
    plt.title(title_str)
    plt.savefig(os.path.join(output_dir, '{}.png'.format(fig_name_str)))
