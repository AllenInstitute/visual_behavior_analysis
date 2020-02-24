import numpy as np
import matplotlib.pyplot as plt
from visual_behavior.visualization.qc import data_loading as dl
from visual_behavior.visualization.qc import data_processing as dp
import visual_behavior.plotting as vbp
import visual_behavior.database as db
from visual_behavior.utilities import EyeTrackingData
import os


################  OPHYS  ################ # NOQA: E402

def plot_max_intensity_projection_for_experiment(ophys_experiment_id, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    max_projection = dl.get_sdk_max_projection(ophys_experiment_id)
    ax.imshow(max_projection, cmap='gray', vmax=np.amax(max_projection)/4.)
    ax.axis('off')
    return ax

def plot_average_image_for_experiment(ophys_experiment_id, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    average_image = dl.get_sdk_ave_projection(ophys_experiment_id)
    ax.imshow(average_image, cmap='gray', vmax=np.amax(average_image)/2.)
    ax.axis('off')
    return ax

def plot_segmentation_mask_for_experiment(ophys_experiment_id, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    # segmentation_mask = dl.get_sdk_segmentation_mask_image(ophys_experiment_id)
    segmentation_mask = dl.get_valid_segmentation_mask(ophys_experiment_id)
    ax.imshow(segmentation_mask, cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
    return ax

def plot_segmentation_mask_overlay_for_experiment(ophys_experiment_id, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax = plot_max_intensity_projection_for_experiment(ophys_experiment_id, ax=ax)
    # segmentation_mask = dl.get_sdk_segmentation_mask_image(ophys_experiment_id)
    segmentation_mask = dl.get_valid_segmentation_mask(ophys_experiment_id)
    mask = np.zeros(segmentation_mask.shape)
    mask[:] = np.nan
    mask[segmentation_mask==1] = 1
    ax.imshow(mask, cmap='hsv', vmax=1, alpha=0.5)
    ax.axis('off')
    return ax

def plot_traces_heatmap_for_experiment(ophys_experiment_id, ax=None):
    dff_traces = dl.get_sdk_dff_traces_array(ophys_experiment_id)
    if ax is None:
        figsize = (14, 5)
        fig, ax = plt.subplots(figsize=figsize)
    cax = ax.pcolormesh(dff_traces, cmap='magma', vmin=0, vmax=0.5)
    ax.set_ylabel('cells')
    ax.set_xlabel('2P frames')
    return ax

def plot_average_intensity_timeseries_for_experiment(ophys_experiment_id, ax=None, color='gray'):
    average_intensity, frame_numbers = dp.get_average_intensity(ophys_experiment_id)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(frame_numbers, average_intensity, color=color, label=ophys_experiment_id)
    ax.set_ylabel('fluorescence value')
    ax.set_xlabel('frame #')
    return ax

def plot_motion_correction_xy_shift_for_experiment(ophys_experiment_id, ax=None):
    df = dl.load_rigid_motion_transform_csv(ophys_experiment_id)
    if ax is None:
        fig, ax = plt.subplots(figsize=(20,4))
    ax.plot(df.framenumber.values, df.x.values, color='red', label='x_shift')
    ax.plot(df.framenumber.values, df.y.values, color='blue', label='y_shift')
    ax.set_xlim(df.framenumber.values[0], df.framenumber.values[-1])
    ax.legend(fontsize='small', loc='upper right')
    ax.set_xlabel('2P frames')
    ax.set_ylabel('pixels')
    return ax

### BEHAVIOR ###

def make_eye_matrix_plot(ophys_experiment_id, ax):
    ax = np.array(ax)
    try:
        ophys_session_id = db.convert_id({'ophys_experiment_id':ophys_experiment_id},'ophys_session_id')
        ed = EyeTrackingData(ophys_session_id)

        frames = np.linspace(0,len(ed.ellipse_fits['pupil'])-1,len(ax.flatten())).astype(int)
        for ii,frame in enumerate(frames):
            axis = ax.flatten()[ii]
            axis.imshow(ed.get_annotated_frame(frame))
            axis.axis('off')
            axis.text(5,5,'frame {}'.format(frame),ha='left',va='top',color='yellow',fontsize=8)

        ax[0][0].set_title('ophys_experiment_id = {}, {} evenly spaced sample eye tracking frames'.format(ophys_experiment_id, len(frames)),ha='left')

    except Exception as e:
        for ii in range(len(ax.flatten())):
            axis = ax.flatten()[ii]
            axis.axis('off')

            error_text = 'could not generate pupil plot for ophys_experiment_id {}\n{}'.format(ophys_experiment_id, e)
            ax[0][0].set_title(error_text,ha='left')
    return ax

