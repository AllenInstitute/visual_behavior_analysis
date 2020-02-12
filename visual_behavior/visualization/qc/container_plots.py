import matplotlib.pyplot as plt
from visual_behavior.visualization import utils as ut
from visual_behavior.visualization.qc import data_loading as dl
from visual_behavior.visualization.qc import experiment_plots as ep
from visual_behavior.visualization.qc import session_plots as sp



def plot_max_projection_images_for_container(ophys_container_id, save_figure=True):
    ## function for getting experiment_ids belonging to container
    ## create figure with sub-axes for each experiment
    ## loop through experiments
    ##### load max projection using data_loading.py function
    ## plot each on axis
    ## return fig? or save fig?
    ophys_experiment_ids = dl.get_ophys_experiment_ids_for_ophys_container_id(ophys_container_id)

    figsize = (25, 5)
    fig, ax = plt.subplots(1,6, figsize=figsize)
    for i,ophys_experiment_id in enumerate(ophys_experiment_ids):
        ax[i] = ep.plot_max_intensity_projection_for_experiment(ophys_experiment_id, ax=ax[i])
        ax[i].axis('off')
        session_type = dl.get_session_type_for_ophys_experiment_id(ophys_experiment_id)
        ax[i].set_title(str(ophys_experiment_id)+'\n'+session_type)

    if save_figure:
        ut.save_figure(fig, figsize, dl.get_container_plots_dir(), 'max_intensity_projection',
                       'container_'+str(ophys_container_id))








