import os
import pandas as pd
from braintv_behav import basepath
from mouse_info import Mouse

def mouse_info(mouse):
    return Mouse(mouse).info_txt

_spreadsheet_path = os.path.join(basepath,"VisualBehaviorDevelopment_CohortIDs.xlsx")
cohort_assignment = pd.read_excel(_spreadsheet_path)