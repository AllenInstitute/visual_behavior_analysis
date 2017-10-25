import Tkinter, tkFileDialog
import pandas as pd
import numpy as np
import glob
import os

root = Tkinter.Tk()
root.withdraw()

def open_IC_CSV(filepath):
    traces = pd.read_csv(filepath)
    IC_number = [col.split(" ")[2] for col in traces.columns if 'time' not in col.lower()]

    return IC_number

def rename_with_leading_zeros(first_tif_filename):
    directory = os.path.split(first_tif_filename)[0]
    filename = os.path.split(first_tif_filename)[1]
    files = glob.glob(os.path.join(directory,filename.split('.')[0]+"*"))
    print "renaming files to facilitate sorting"
    for filename in files:
        if '~' not in filename and "IC" not in filename: #I'm using the '~' to designate files that have already been renamed
            index = filename.split("_")[-1].split(".")[0]
            if index == '':
                index = 0
            
            new_filename = "_".join(filename.split("_")[:-1])+'_~{0:03d}~.tif'.format(int(index))
            print "renaming {}\nto       {}".format(filename,new_filename)
            os.rename(filename,new_filename)
    print ""

def match_names_to_ICs(first_tif_filename,IC_numbers):
    directory = os.path.split(first_tif_filename)[0]
    filename = os.path.split(first_tif_filename)[1]
    files = glob.glob(os.path.join(directory,filename.split('.')[0]+"*"))
    print "renaming files to match ICs"
    for ii,filename in enumerate(np.sort(files)):
        if "IC" not in filename:
            new_filename = filename.split("~")[0]+"IC{0:03d}".format(int(IC_numbers[ii]))+filename.split("~")[-1]
            print "renaming  {}\nto        {}".format(filename,new_filename)
            os.rename(filename,new_filename)



if __name__=='__main__':
    print "Navigate to the folder holding the traces CSV file"
    traces_CSV_file_path = tkFileDialog.askopenfilename(title = "Select CSV Traces file",filetypes = [("csv files","*.csv"),])
    IC_numbers = open_IC_CSV(traces_CSV_file_path)

    print ""
    print "Navigate to the first TIF mask file"
    initialdir = os.path.split(traces_CSV_file_path)[0]
    first_mask_file_path = tkFileDialog.askopenfilename(initialdir = initialdir,title = "Select first IC mask TIF file",filetypes = [("tif files","*.tif"),])
    rename_with_leading_zeros(first_mask_file_path)
    match_names_to_ICs(first_mask_file_path,IC_numbers)
