import visual_behavior.utilities as vbu
import argparse

def cache_behavior_performance(behavior_session_id):
    vbu.cache_behavior_stats(behavior_session_id)

if __name__ == "__main__":
    # define args
    parser = argparse.ArgumentParser()
    parser.add_argument('--behavior-session-id', type=int)
    args = parser.parse_args()

    cache_behavior_performance(args.behavior_session_id)