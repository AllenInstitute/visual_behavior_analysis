from visual_behavior.ophys.response_analysis import cell_metrics
import visual_behavior.data_access.loading as loading

if __name__ == '__main__':
    # from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache as bpc
    # cache_dir = loading.get_platform_analysis_cache_dir()
    # cache = bpc.from_s3_cache(cache_dir=cache_dir)
    # experiments_table = cache.get_ophys_experiment_table()
    # print(cache_dir)

    experiments_table = loading.get_platform_paper_experiment_table()

    interpolate = True
    output_sampling_rate = 30

    for data_type in ['events', 'filtered_events', 'dff']:
        cell_metrics.load_and_save_all_metrics_tables_for_all_experiments(experiments_table, data_type=data_type,
                                                                      interpolate=interpolate, output_sampling_rate=output_sampling_rate)
