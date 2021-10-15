from visual_behavior.ophys.response_analysis import cell_metrics


if __name__ == '__main__':

    ophys_experiment_table = loading.get_filtered_ophys_experiment_table(release_data_only=True)

    cell_metrics.generate_and_save_all_metrics_tables_for_all_experiments(ophys_experiment_table)