import os
import pandas as pd

COHORTS = {
    'Cohort 1': ['M246782', 'M249123', 'M258174', 'M250783', 'M250786','M258196'],
    'Cohort 2': ['M258173', 'M258089', 'M258276', 'M261584', 'M258274','M258275'],
    'Cohort 3': ['M272465','M265158','M271966','M265154','M258194','M265166'],
    'Cohort 4': ['M272464','M271733','M271728','M276951','M276950','M276600',]
}

def get_cohort_assignment(cohorts):
    for c,m_list in cohorts.iteritems():
        for m in m_list:
            yield dict(
                cohort=c,
                mouse_id=m,
            )


def mouse_info(mouse):
    network_dir = r"\\aibsdata\neuralcoding\Behavior\Data\{}".format(mouse)
    
    info = dict(mouse_id=mouse)
    with open(os.path.join(network_dir,'info.txt'),'r') as f:
        for line in f:
            items = line.split(':')
            key, val = items[0], ':'.join(items[1:]).strip()
            info[key] = val
    return info

cohort_assignment = pd.DataFrame(get_cohort_assignment(COHORTS))