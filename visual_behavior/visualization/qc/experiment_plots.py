import numpy as np
import matplotlib.pyplot as plt
from visual_behavior.visualization.qc import data_loading as dl
import visual_behavior.plotting as vbp
import visual_behavior.database as db
from visual_behavior.utilities import EyeTrackingData
import os

def plot_max_intensity_projection_for_experiment(ophys_experiment_id, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    max_projection = dl.get_sdk_max_projection(ophys_experiment_id)
    ax.imshow(max_projection, cmap='gray', vmax=np.amax(max_projection)/2.)
    ax.axis('off')
    return ax


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