import matplotlib.pyplot as plt
from visual_behavior.visualization import utils as ut
import visual_behavior.plotting as vbp
import visual_behavior.database as db
from visual_behavior.visualization.qc import data_loading as dl
from visual_behavior.visualization.qc import experiment_plots as ep
from visual_behavior.visualization.qc import session_plots as sp
import os

def plot_max_projection_images_for_container(ophys_container_id, save_figure=True):
    # function for getting experiment_ids belonging to container
    # create figure with sub-axes for each experiment
    # loop through experiments
    # load max projection using data_loading.py function
    # plot each on axis
    # return fig? or save fig?
    ophys_experiment_ids = dl.get_ophys_experiment_ids_for_ophys_container_id(ophys_container_id)

    figsize = (25, 5)
    fig, ax = plt.subplots(1, 6, figsize=figsize)
    for i, ophys_experiment_id in enumerate(ophys_experiment_ids):
        ax[i] = ep.plot_max_intensity_projection_for_experiment(ophys_experiment_id, ax=ax[i])
        ax[i].axis('off')
        session_type = dl.get_session_type_for_ophys_experiment_id(ophys_experiment_id)
        ax[i].set_title(str(ophys_experiment_id) + '\n' + session_type)

    if save_figure:
        ut.save_figure(fig, figsize, dl.get_container_plots_dir(), 'max_intensity_projection',
                       'container_' + str(ophys_container_id))


def plot_eye_tracking_sample_frames(ophys_container_id, save_figure=True):
    table = dl.get_filtered_ophys_experiment_table()
    ophys_experiment_ids = table.query('container_id == {}'.format(ophys_container_id)).sort_values(by='date_of_acquisition')['ophys_experiment_id']
    
    fig = plt.figure(figsize=(16, 5 * len(ophys_experiment_ids)))
    axes = []
    nplots = len(ophys_experiment_ids)
    buffer = 0.05
    for ii, ophys_experiment_id in enumerate(ophys_experiment_ids):
        print('on ophys_experiment_id {}, #{} of {}'.format(ophys_experiment_id, ii + 1, nplots))
        axes.append(vbp.placeAxesOnGrid(fig, dim=(3, 10), xspan=(0, 1), yspan=(ii / nplots + buffer, (ii + 1) / nplots)))
        axes[-1] = ep.make_eye_matrix_plot(ophys_experiment_id, axes[-1])

    savepath = os.path.join(dl.get_container_plots_dir(), 'eyetracking_sample_frames', 'container_{}.png'.format(ophys_container_id))
    fig.savefig(savepath, dpi=300, pad_inches=0.0, bbox_inches='tight')

    return fig, axes
