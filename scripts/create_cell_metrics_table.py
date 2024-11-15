import argparse

from visual_behavior.ophys.response_analysis import cell_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ophys_experiment_id", type=int,
                        help="Experiment ID to process")
    args = parser.parse_args()
    ophys_experiment_id = args.ophys_experiment_id
    print('generating cell metrics for', ophys_experiment_id)

    interpolate = True
    time_window = [-2, 2.1]
    response_window_duration = 0.5

    for data_type in ['events', 'filtered_events', 'dff']:
        cell_metrics.generate_and_save_all_metrics_tables_for_experiment(ophys_experiment_id, data_type=data_type, interpolate=interpolate,
                                                        time_window=time_window, response_window_duration=response_window_duration,
                                                        overwrite=False)
