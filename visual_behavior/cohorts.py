import os
import shutil
import pandas as pd
from visual_behavior import basepath, project_dir
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
    info_txt = Mouse(mouse).info_txt
    info_txt.update(mouse_id=mouse)
    return info_txt


def load_cohort_assignment():
    """ loads data about cohort assignment for visual behavior development mice

    Returns
    -------
    cohort_assignment : pandas DataFrame where each row is a cohort mouse

    """

    _spreadsheet_path = os.path.join(basepath,"VisualBehaviorDevelopment_CohortIDs.xlsx")
    return pd.read_excel(_spreadsheet_path)

def copy_latest(destination=None):
    """ copies all of the latest pkl files for cohort mice to a local directory

    Parameters
    ----------
    destination : str, optional
        destination to copy all of the cohort data file to

    """

    cohort_assignment = load_cohort_assignment()
    network_dir = basepath

    if destination is None:
        destination = os.path.join(project_dir,'data','raw')

    if os.path.exists(destination)==False:
        os.mkdir(destination)
    for rr,row in cohort_assignment.iterrows():
        mouse = row['mouse']
        src = os.path.join(network_dir,mouse,'output')
        for fn in os.listdir(src):
            if os.path.exists(os.path.join(destination,fn)) == False:
                # logging.info("copying {}".format(fn))
                print "copying {}".format(fn)
                shutil.copy(os.path.join(src,fn),os.path.join(destination,fn))
