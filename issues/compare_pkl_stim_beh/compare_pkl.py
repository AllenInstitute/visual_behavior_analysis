import pandas as pd
import datetime
import collections

f1 = '/allen/programs/braintv/production/visualbehavior/prod0/specimen_756577249/ophys_session_789220000/789220000_stim.pkl'
f2 = '/allen/programs/braintv/production/visualbehavior/prod0/specimen_756577249/behavior_session_789295700/789220000.pkl'

p1 = pd.read_pickle(f1)
p2 = pd.read_pickle(f2)

missing_key_set = set()
key_list = set(p1.keys()).union(p2.keys())
for key in key_list:
    v1 = p1[key] if key in p1 else None
    v2 = p2[key] if key in p2 else None
    
    if v1 is None or v2 is None:
        if v1 == v2:
            pass # Probably will never happen
        else:

            # Mismatch; record as such
            if v1 is None:
                file_missing_key = f1
            else:
                file_missing_key = f2
            missing_key_set.add((key, file_missing_key))
    else:

        # Both fields exist, test them against eachother:
        if isinstance(v1, str):
            assert v1 == v2
        elif isinstance(v1, datetime):
            print v1, v2
        else:
            print type(v1)