import os
import pandas as pd
from braintv_behav import basepath
from mouse_info import Mouse

def mouse_info(mouse):    
    '''
    Returns the info found in mouse's info.txt file
    
    Parameters
    ------
    mouse : str
        mouse identifier
    
    Returns
    ------
    info_txt : dictionary containing key:value pairs in mouse's info.txt file
    '''
def mouse_info(mouse):
    info_txt = Mouse(mouse).info_txt
    info_txt.update(mouse_id=mouse)
    return info_txt


def load_cohort_assignment():
    _spreadsheet_path = os.path.join(basepath,"VisualBehaviorDevelopment_CohortIDs.xlsx")
    return pd.read_excel(_spreadsheet_path)
