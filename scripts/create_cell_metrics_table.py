import argparse

from visual_behavior.ophys.response_analysis import cell_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ophys_experiment_id", type=int,
                        help="Experiment ID to process")
    args = parser.parse_args()
    ophys_experiment_id = args.ophys_experiment_id

    cell_metrics.generate_and_save_all_metrics_tables_for_experiment(ophys_experiment_id)