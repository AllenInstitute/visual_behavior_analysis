import visual_behavior.utilities as vbu
import argparse


if __name__ == "__main__":
    # define args
    parser = argparse.ArgumentParser()
    parser.add_argument('--behavior-session-id', type=int)
    parser.add_argument('--method', type=str)

    args = parser.parse_args()
    behavior_session_id = args.behavior_session_id
    method = args.method

    # response probability
    if method == 'response_probability':

        # save response probability matrices across image transitions
        print('generating response probability matrix for', behavior_session_id, method, 'engaged_only: ', False)
        vbu.cache_response_probability(behavior_session_id, engaged_only=False)

        print('generating response probability matrix for', behavior_session_id, method, 'engaged_only: ', True)
        vbu.cache_response_probability(behavior_session_id, engaged_only=True)

    elif method == 'sdk':

        # save out performance metrics directly from SDK dataset object (doesnt work with per_image or engaged_only)
        print('generating behavior metrics for', behavior_session_id, method)
        vbu.cache_behavior_stats(behavior_session_id=behavior_session_id, method=method, per_image=False, engaged_only=False)

    else:

        ## behavior metrics full session
        print('generating behavior metrics for', behavior_session_id, method, 'per_image: ', False, 'engaged_only: ', False)
        vbu.cache_behavior_stats(behavior_session_id=behavior_session_id, method=method, per_image=False, engaged_only=False)

        print('generating behavior metrics for', behavior_session_id, method, 'per_image: ', True, 'engaged_only: ', False)
        vbu.cache_behavior_stats(behavior_session_id=behavior_session_id, method=method, per_image=True, engaged_only=False)

        ## behavior_metrics engaged only

        print('generating behavior metrics for', behavior_session_id, method, 'per_image:', False,'engaged_only:', True)
        vbu.cache_behavior_stats(behavior_session_id=behavior_session_id, method=method, per_image=False, engaged_only=True)

        print('generating behavior metrics for', behavior_session_id, method, 'per_image:', True,'engaged_only:', True)
        vbu.cache_behavior_stats(behavior_session_id=behavior_session_id, method=method, per_image=True, engaged_only=True)


