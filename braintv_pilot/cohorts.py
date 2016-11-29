import os
import pandas as pd
from . import basepath

def get_cohort_assignment(cohorts):
    for c,m_list in cohorts.iteritems():
        for m in m_list:
            yield dict(
                cohort=c,
                mouse_id=m,
                day_zero=day_zero[c],
            )


def mouse_info(mouse):
    network_dir = os.path.join(basepath,mouse)
    
    info = dict(mouse_id=mouse)
    with open(os.path.join(network_dir,'info.txt'),'r') as f:
        for line in f:
            items = line.split(':')
            key, val = items[0], ':'.join(items[1:]).strip()
            info[key] = val
    return info

_spreadsheet_path = os.path.join(basepath,"VisualBehaviorDevelopment_CohortIDs.xlsx")
cohort_assignment = pd.read_excel(_spreadsheet_path)