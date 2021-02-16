#!/usr/bin/env python
# coding: utf-8

# In[31]:


import os
import numpy as np
import pandas as pd

import visual_behavior.data_access.loading as loading
import visual_behavior.data_access.processing as processing


# experiment_id = experiments_to_work_with[7]
# experiment_id
#
# # ### load average depth image
#
# # This loads an image of the FOV, averaged over 16 frames
# im = loading.get_average_depth_image(experiment_id)
#
# dataset = loading.get_ophys_dataset(experiment_id)
# average_image = dataset.average_projection.copy()
#
# compute_basic_snr_for_frame(average_image)
#
# movie = loading.load_motion_corrected_movie(experiment_id)
# frame = movie[500, :, :]
#
# compute_basic_snr_for_frame(frame)
#
# compute_basic_snr_for_frame(frame)
#
# frame_rate = dataset.metadata['ophys_frame_rate']
# n_frames = int(1 * 60 * frame_rate)
#
# segment = movie[-n_frames:, :, :]
# fingerprint_avg_image = np.nanmean(segment, axis=0)
# fingerprint_max_image = np.nanmax(segment, axis=0)
#
# frame_rate = dataset.metadata['ophys_frame_rate']
# n_frames = int(1 * 60 * frame_rate)
#
# segment = movie[-n_frames:, :, :]
# fingerprint_avg_image = np.nanmean(segment, axis=0)
# fingerprint_max_image = np.nanmax(segment, axis=0)
#
# basic_snr_fingerprint_avg_image = compute_basic_snr_for_frame(fingerprint_avg_image)
# basic_snr_fingerprint_max_image = compute_basic_snr_for_frame(fingerprint_avg_image)
#
# flat_image = np.ndarray.flatten(fingerprint_avg_image)
#
# dataset = loading.get_ophys_dataset(experiment_id)
# traces = dataset.dff_traces
#
# traces = processing.compute_robust_snr_on_dataframe(traces)
#


def compute_basic_snr_for_frame(frame):
    basic_snr = np.std(frame) / np.mean(frame)
    return basic_snr


# ### Doug's metric
def assign_snr(dataset):
    for idx, row in dataset.dff_traces.iterrows():
        dff = row['dff'][
            1000:-1000]  # avoid first/last 100 datapoints to avoid spurious dff data at start/finish of recording
        dataset.dff_traces.at[idx, 'snr_99p_over_std'] = np.percentile(dff, 99) / np.std(dff)
        dataset.dff_traces.at[idx, 'peak_over_std'] = np.max(dff) / np.std(dff)
        dataset.dff_traces.at[idx, 'snr_mu_over_std'] = np.mean(dff) / np.std(dff)
    return dataset


def get_best_snr(dataset):
    assign_snr(dataset)
    return dataset.dff_traces['peak_over_std'].max()

# ### Sam / mouse-seeks metric

# http://stash.corp.alleninstitute.org/projects/SSCI/repos/ophysextractor/browse/ophysextractor/datasets/motion_corr_physio.py
# get_snr_metrics()


def get_snr_metrics_df_for_experiments(experiment_ids):
    metrics_list = []
    problems_list = []
    for experiment_id in experiment_ids:
        print(experiment_id)
        try:
            dataset = loading.get_ophys_dataset(experiment_id)

            # movie = loading.load_motion_corrected_movie(experiment_id)
            # movie_frame = movie[20000, :, :]
            depth_image = loading.get_average_depth_image(experiment_id)
            average_image = dataset.average_projection.data.copy()
            max_projection_image = dataset.max_projection.data.copy()
            traces = dataset.dff_traces.copy()
            # fingerprint avg and max
            # frame_rate = dataset.metadata['ophys_frame_rate']
            # n_frames = int(3 * 60 * frame_rate)
            # segment = movie[-n_frames:, :, :]
            # fingerprint_avg_image = np.nanmean(segment, axis=0)
            # fingerprint_max_image = np.nanmax(segment, axis=0)

            # basic SNR for movie frame
            # basic_snr_movie_frame = compute_basic_snr_for_frame(movie_frame)

            # basic SNR for depth image
            basic_snr_depth_image = compute_basic_snr_for_frame(depth_image)

            # basic SNR for average image
            basic_snr_average_image = compute_basic_snr_for_frame(average_image)

            # basic SNR for max image
            basic_snr_max_image = compute_basic_snr_for_frame(max_projection_image)

            # STD of avg image aka contrast
            std_avg_image = np.std(average_image)
            mean_avg_image = np.mean(average_image)

            # STD of  max image aka contrast
            std_max_image = np.std(max_projection_image)
            mean_max_image = np.mean(max_projection_image)

            # # basic SNR for fingerprint avg image
            # basic_snr_fingerprint_avg_image = compute_basic_snr_for_frame(fingerprint_avg_image)
            #
            # # basic SNR for fingerprint max image
            # basic_snr_fingerprint_max_image = compute_basic_snr_for_frame(fingerprint_max_image)
            #
            # # STD of fingerprint avg image aka contrast
            # std_fingerprint_avg_image = np.std(fingerprint_avg_image)
            #
            # # STD of fingerprint max image aka contrast
            # std_fingerprint_max_image = np.std(fingerprint_max_image)

            # robust SNR on traces
            traces = processing.compute_robust_snr_on_dataframe(traces)
            median_robust_snr_traces = traces.robust_snr.median()
            max_robust_snr_traces = traces.robust_snr.max()

            # Doug SNR on traces
            peak_over_std_max = get_best_snr(dataset)

            metrics_list.append([experiment_id, basic_snr_depth_image, basic_snr_average_image, basic_snr_max_image,
                                 std_avg_image, std_max_image, mean_avg_image, mean_max_image,
                                 # basic_snr_fingerprint_avg_image, basic_snr_fingerprint_max_image,
                                 # std_fingerprint_avg_image, std_fingerprint_max_image, basic_snr_movie_frame,
                                 median_robust_snr_traces, max_robust_snr_traces, peak_over_std_max])
        except BaseException:
            problems_list.append(experiment_id)
            print('problem for', experiment_id)

    columns = ['experiment_id', 'basic_snr_depth_image', 'basic_snr_average_image', 'basic_snr_max_image',
               'std_avg_image', 'std_max_image', 'mean_avg_image', 'mean_max_image',
               # 'basic_snr_fingerprint_avg_image', 'basic_snr_fingerprint_max_image',
               # 'contrast_std_fingerprint_avg_image', 'contrast_std_fingerprint_max_image', 'basic_snr_movie_frame',
               'median_robust_snr_traces', 'max_robust_snr_traces', 'peak_over_std_max_traces', ]

    metrics_df = pd.DataFrame(metrics_list, columns=columns)

    return metrics_df, problems_list


if __name__ == '__main__':

    try:
        experiments_table = loading.get_filtered_ophys_experiment_table(include_failed_data=True)
        experiments = experiments_table[(experiments_table.cre_line == 'Slc17a7-IRES2-Cre') &
                                        (experiments_table.imaging_depth < 250) &
                                        (experiments_table.imaging_depth > 150)]
        experiment_ids = experiments.index.values
        metrics_df, problems_list = get_snr_metrics_df_for_experiments(experiment_ids)
        save_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/qc_plots/snr_metrics'
        # save_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\qc_plots\snr_metrics'
        metrics_df.to_csv(os.path.join(save_dir, 'snr_metrics_Slc.csv'))
        problems_list = pd.DataFrame(problems_list)
        problems_list.to_csv(os.path.join(save_dir, 'problems_list_Slc.csv'))
    except Exception as e:
        print(e)
