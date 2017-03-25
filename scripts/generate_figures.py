#!/usr/bin/env python

import matplotlib as mpl
import platform
if platform.system() == 'Linux': 
    mpl.use('Agg')
import os
import numpy as np 
import time
import sys
import shutil
import tables as tb
import getpass
import imp

import gc
if platform.system() == 'Linux': 
    sys.path.append(r'//data/nc-ophys/Doug/imaging_behavior_master')
    dro_path = '//data/nc-ophys/BehaviorCode/dro'
elif platform.system() != 'Linux' and getpass.getuser() == 'dougo':
    dro_path = '/Users/dougo/Dropbox/PythonCode/dro'
else:
    dro_path = '//aibsdata2/nc-ophys/BehaviorCode/dro'

imp.load_source('dro_plots',os.path.join(dro_path,'plotting_functions.py'))

imp.load_source('dro',os.path.join(dro_path,'utilities.py'))
import dro


import dro_plots

import pandas as pd

def copy_figures():
    if platform.system() == 'Linux':
        basepath = "/data/neuralcoding/Behavior/Data"
    else:
        basepath = "//aibsdata/neuralcoding/Behavior/Data"
    # Open a spreadsheet that contains the mouse/cohort IDs
    spreadsheet_path = os.path.join(basepath,"VisualBehaviorDevelopment_CohortIDs.xlsx")
    df = pd.read_excel(spreadsheet_path)

    # For each entry in the spreadsheet, copy figures to a central folder
    for idx,row in df.iterrows():
        dst_figurepath = os.path.join(basepath,row.experiment,'cohort'+str(row.cohort),row.mouse,'figures')
        src_figurepath = os.path.join(basepath,row.mouse,'figures')
        if os.path.exists(dst_figurepath) == False:
            os.makedirs(dst_figurepath)
        
        for fn in os.listdir(src_figurepath):
            if os.path.exists(os.path.join(dst_figurepath,fn)) == False and fn[0] != '.':
                shutil.copy(os.path.join(src_figurepath,fn),os.path.join(dst_figurepath,fn))


def run(mouse_id,figures_to_generate='most_recent',load_existing_dataframe=True,keep_figures_open=False):

    if platform.system() == 'Linux':
        data_location = os.path.join('/data/neuralcoding/Behavior/Data',mouse_id,'output')
        save_location = os.path.join('/data/neuralcoding/Behavior/Data',mouse_id,'figures')
    elif platform.system() == 'Windows':
        data_location = os.path.join('//aibsdata/neuralcoding/Behavior/Data',mouse_id,'output')
        save_location = os.path.join('//aibsdata/neuralcoding/Behavior/Data',mouse_id,'figures')
    elif platform.system() == 'Darwin':
        warnings.warn('paths not set up for Mac')
        pass

    if os.path.exists(save_location)==False:
        os.mkdir(save_location)

    df = dro.load_from_folder(data_location,load_existing_dataframe=load_existing_dataframe)

    print "dataframe loaded"

    dates = np.sort(df.startdatetime.unique())
    if figures_to_generate == 'most_recent':
        dates_to_plot = [dates[-1]]
    elif figures_to_generate == 'all':
        dates_to_plot = dates

    print "about to generate figures for: ",dates_to_plot

    for date in dates_to_plot:
        dft = df[df.startdatetime==date]
        fname = dft.iloc[0].filename.split(".pkl")[0]
        fig = dro_plots.make_daily_figure(dft)
        dro.save_figure(fig, os.path.join(save_location,fname))
        print "saving figure:",os.path.join(save_location,fname)
        if keep_figures_open == False:
            fig.clf()

    summary_figure = dro_plots.make_summary_figure(df,mouse_id)
    dro.save_figure(summary_figure, os.path.join(save_location,'summary_figure'))
    if keep_figures_open == False:
        summary_figure.clf()

    try:
        copy_figures()
    except:
        pass



if __name__ == "__main__":
    if len(sys.argv)>=2:
        mouse_id = sys.argv[1]
    else:
        mouse_id = 'M272464'

    if len(sys.argv)>=3:
        figures_to_generate = sys.argv[2]
    else:
        figures_to_generate = 'most_recent'

    if len(sys.argv)>=4:
        load_existing_dataframe = sys.argv[3]
    else:
        load_existing_dataframe = True

    run(mouse_id,figures_to_generate,load_existing_dataframe)