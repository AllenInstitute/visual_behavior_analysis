import argparse

# import visual_behavior.data_access.loading as loading
# import visual_behavior.visualization.ophys.platform_paper_figures as ppf

import matplotlib.pyplot as plt
import visual_behavior.visualization.utils as ut
import visual_behavior.visualization.qc.experiment_plots as ep


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ophys_experiment_id", type=int,
                        help="Experiment ID to process")
    args = parser.parse_args()
    ophys_experiment_id = args.ophys_experiment_id


    save_dir = r'/allen/aibs/informatics/danielsf/mfish_learning/segmentation_220216'

    try:
        figsize=(5,5)
        fig, ax = plt.subplots(figsize=figsize)
        ax = ep.plot_valid_segmentation_mask_outlines_for_experiment(ophys_experiment_id, ax=ax)
        ut.save_figure(fig, figsize, save_dir, 'plots', str(ophys_experiment_id) + '_legacy_segmentation')
    except:
        figsize = (5, 5)
        fig, ax = plt.subplots(figsize=figsize)
        ax = ep.plot_motion_correction_max_image_for_experiment(ophys_experiment_id, ax=ax)
        ut.save_figure(fig, figsize, save_dir, 'plots', str(ophys_experiment_id) + '_legacy_segmentation_failed')



    # dataset = loading.get_ophys_dataset(ophys_experiment_id)

    # trials = dataset.trials.copy()
    # start_times = trials[trials.stimulus_change].start_time.values
    #
    # save_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/platform_paper_plots'
    #
    # for start_time in start_times[:20]:
    #     ppf.plot_behavior_timeseries_stacked(dataset, start_time, duration_seconds=20, save_dir=save_dir, ax=None)
