from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import warnings

def find_filenames(path,mouse_id=None,camera_config='widefield'):
    """ searches a path and returns a dictionary containing all of the necessary
        files to analyze an inscopix recording sessino

    Parameters
    ----------
    path : str
        path to search
    mouse_id : str
        mouse ID

    Returns
    -------
    filepaths : dictinoary containing names of files as keys, filepaths as values

    """

    if not mouse_id:
        mouse_id='none'
        warnings.warn('No mouse ID specified. Will not be able to locate mouse-specific files such as the behavior log file')

    #a list of tuples, with each tuple of the form (v0,v1) where:
    #   v0 = descriptive name of file
    #   v1 = list of strings to search for that uniquely identify the file

    if camera_config == 'widefield':
        camera_list = [
            ('behavior_video',['-0.avi']),
            ('behavior_video_timestamps',['-0.h5']),
            ('eye_video',['-1.avi']),
            ('eye_video_timestamps',['-1.h5']),
            ('face_video',['-2.avi']),
            ('face_video_timestamps',['-2.h5']),
        ]
    elif camera_config == 'laptop':
        camera_list = [
            ('laptop_video',['-0.avi']),
            ('laptop_video_timestamps',['-0.h5']),
            ('webcam_video',['-1.avi']),
            ('webcam_video_timestamps',['-1.h5']),
        ]
    else:
        camera_list = []

    things_to_look_for = [
        ('sync_file',['ync','.h5']),
        ('xml_file',['.xml']),
        ('behavior_pkl',[mouse_id,'.pkl']),
        ('traces',['races','.csv']),
        ('tif_list',['ecording','.tif']),
        ('downsampled_movie',['ownsampled','.h5']),
        ('zscored_movie',['scored','.h5']),
        ('IC_list',['IC','.tif'])
    ] + camera_list



    #initialize the filename dictionary
    filename_dict={}
    for file_description,string_list in things_to_look_for:
        filename_dict[file_description]=[]

    for directory, dirnames, filenames in os.walk(path):
        for file_description,string_list in things_to_look_for:
            for fn in filenames:
                bools = [fn.find(s)>0 for s in string_list]
                if sum(bools)==len(bools):
                    filename_dict[file_description].append(os.path.join(directory,fn))

    for key in filename_dict.keys():
        if len(filename_dict[key])==0:
            #nothing was found, return None
            filename_dict[key]=None
        elif len(filename_dict[key])==1 and key is not 'tif_list':
            #only one thing was found, return just the element in the list
            filename_dict[key]=filename_dict[key][0]

    #Mosaic appends a file identifier to every tif after the first.
    # #However, because the first doesn't have an identifier, it goes to the end of a sorted list
    # #The code below re-sorts to ensure that the first file that was recorded is at the front of the list
    if filename_dict['tif_list'] is not None:
        filename_dict['tif_list']=sorted(filename_dict['tif_list'])
        filename_dict['tif_list']=[filename_dict['tif_list'][-1]]+filename_dict['tif_list'][:-1]

    if filename_dict['IC_list'] is not None:
        filename_dict['IC_list']=sorted(filename_dict['IC_list'])


    return filename_dict

def mkdir(path):
    folder = os.path.split(path)[-1]
    basedir = os.path.split(path)[0]
    if folder not in os.listdir(basedir):
        os.mkdir(os.path.join(basedir,folder))

def quadratic_func(x, a, b, c):
    return a*x**2 + b*x + c

def detrend_movie(movie):
    from scipy.optimize import curve_fit
    detrended_movie = np.empty(np.shape(movie))
    for i in range(np.shape(movie)[0]):
        for j in range(np.shape(movie)[1]):
            ydata=movie[:,i,j]
            xdata=np.arange(len(ydata))
            popt, pcov = curve_fit(quadratic_func, xdata, ydata)
            detrended_movie[:,i,j]=ydata-quadratic_func(xdata, *popt)
        print(i,j)
    return detrended_movie

def parse_xml(xmlfile):
    '''
    Pulls attributes from auto-saved xml file, turns into Python dictionary
    '''
    attributes = {}
    f=open(xmlfile,'r')
    for line in f.readlines():
        if 'attr name' in line:
            key = line.split('"')[1]
            value = line.split('"')[2].split('</attr')[0][1:]
            attributes[key]=value

    #try turning attributes into floats
    for attritube in attributes.keys():
        try:
            attributes[attritube] = float(attributes[attritube])
        except Exception as e:
            pass
    return attributes

def save_h5(data,filename,dtype=float,keyname='data'):
    '''
    save an h5 file
    '''
    h5f = h5py.File(filename, 'w')
    h5f.create_dataset(keyname, data=data,dtype=dtype)
    h5f.close()

def load_h5(filename,keyname='data'):
    '''
    load an h5 file
    '''
    h5f = h5py.File(filename,'r')
    data = h5f['data'][:].astype(float)
    h5f.close()
    return data

def downsample_and_concatenate_tifs(tif_list,spatial_downsample_factor=4,temporal_downsample_factor=1):
    """
    opens each tif
    downsamples by downsample factor
    returns concatenated array
    """
    from skimage.measure import block_reduce
    from skimage import io as skio
    fsmall=[]
    if type(tif_list)==str:
        tif_list=[tif_list]
    for tif in tif_list:
        print('loading ',tif)
        f = skio.imread(tif)
        print('downsampling')
        fsmall.append((block_reduce(f,block_size=(temporal_downsample_factor,spatial_downsample_factor,spatial_downsample_factor),func=np.mean)))

    print('concatenating')
    dat=np.array(fsmall[0])
    for i in range(1,len(fsmall)):
        dat=np.concatenate((dat,fsmall[i]),axis=0)

    return dat

def open_traces(datapath):
    """
    a helper function for opening a CSV
    returns a dataframe
    """
    return pd.read_csv(datapath)

def make_traces_plot(data,N_ICs=60,spread_factor=1,scale_factor=0.25,pad=1,title=None,height_scale=0.75,colors=['blue'],fig=None,ax=None):
    """
    Extracts traces from dataframe that results from trace extraction in Mosaic
    Makes a simple plot

    """
    if ax is None:
        fig,ax=plt.subplots(figsize=(9,9*height_scale))

    ICs = [col for col in data.columns if 's.d.' in col]

    for col,IC in enumerate(ICs[:N_ICs]):
        color=colors[col%len(colors)]
        ax.plot(data['Time (s)']/60.,spread_factor*col+scale_factor*data[IC],color=color)

    ax.set_xlabel('Time (minutes)')
    ax.set_xlim(0,np.max(data['Time (s)'])/60.)
    ax.set_ylim(-spread_factor,N_ICs*spread_factor+pad)
    ax.set_yticks([])
    if title:
        ax.set_title(title)

    if ax is None:
        return fig,ax
    else:
        return ax

def heat_plot(traces,t=None,ax=None,colorbar=True,clim=[1,5],cmap='magma',label='z-scored activity'):
    if ax == None:
        fig,ax=plt.subplots(figsize=(9,5))
    if type(traces)==pd.core.frame.DataFrame:
        t=traces['Time (s)'].values
        heatmap=traces[[col for col in traces.columns if 'Time' not in col]].values.T
    else:
        heatmap=traces
    extent = [t[0],t[-1],np.shape(heatmap)[0],0]
    im=ax.imshow(heatmap,aspect='auto',extent=extent,clim=clim,cmap=cmap,interpolation='none')
    if colorbar==True:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size = "5%", pad = 0.05, aspect=2.3/0.15)
        plt.colorbar(im, cax = cax, extendfrac=20,label=label)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('IC Number')
