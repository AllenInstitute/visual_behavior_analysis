# -*- coding: utf-8 -*-

import os
import shutil
import logging
from .cohorts import cohort_assignment


network_dir = r'\\aibsdata\neuralcoding\Behavior\Data'
project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
local_dir = os.path.join(project_dir,'data','raw')

def copy_latest():
    if os.path.exists(local_dir)==False:
        os.mkdir(local_dir)
    for rr,row in cohort_assignment.iterrows():
        mouse = row['mouse_id']
        src = os.path.join(network_dir,mouse,'output')
        for fn in os.listdir(src):
            if os.path.exists(os.path.join(local_dir,fn)) == False:
                logging.info("copying {}".format(fn))
                shutil.copy(os.path.join(src,fn),os.path.join(local_dir,fn))

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('copying latest files from cohorts')
    copy_latest()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
