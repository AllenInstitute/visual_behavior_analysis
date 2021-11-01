import visual_behavior.utilities as vbu
import argparse

# def cache_behavior_performance(behavior_session_id, method):
#     vbu.cache_behavior_stats(behavior_session_id, method)

if __name__ == "__main__":
    # define args
    parser = argparse.ArgumentParser()
    parser.add_argument('--behavior-session-id', type=int)
    parser.add_argument('--method', type=str)

    args = parser.parse_args()
    # behavior_session_id = args.behavior_session_id
    # method = args.method
    # print(behavior_session_id, method)
    # vbu.cache_behavior_stats(behavior_session_id=behavior_session_id, method=method)
    # cache_behavior_performance(args.behavior_session_id, args.method)

    # import pandas as pd
    # import visual_behavior.data_access.loading as loading
    #
    # df = pd.read_csv(os.path.join(loading.get_platform_analysis_cache_dir(), 'behavior_only_sessions_without_nwbs.csv'))
    # behavior_session_ids = df.behavior_session_id.values

    # behavior only data
    import visual_behavior.data_access.loading as loading
    behavior_sessions = loading.get_platform_paper_behavior_session_table()
    # remove all ophys sessions
    behavior_sessions = behavior_sessions[behavior_sessions.session_type.str.contains('OPHYS') == False]
    behavior_session_ids = behavior_sessions.index.values

    for behavior_session_id in behavior_session_ids:
        vbu.cache_behavior_stats(behavior_session_id, 'sdk')