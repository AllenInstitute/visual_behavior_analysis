import argparse

import visual_behavior.data_access.loading as loading
import visual_behavior.visualization.ophys.platform_paper_figures as ppf


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ophys_experiment_id", type=int,
                        help="Experiment ID to process")
    args = parser.parse_args()
    ophys_experiment_id = args.ophys_experiment_id

    dataset = loading.get_ophys_dataset(ophys_experiment_id)

    trials = dataset.trials.copy()
    start_times = trials[trials.stimulus_change].start_time.values

    for start_time in start_times[::4][:20]:
        plot_behavior_timeseries(dataset, start_time, duration_seconds=20, save_dir=save_dir, ax=None)
