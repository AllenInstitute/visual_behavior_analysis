import numpy as np
import matplotlib.pyplot as plt
from visual_behavior.visualization.qc import data_loading as dl



def plot_max_intensity_projection_for_experiment(ophys_experiment_id, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    max_projection = dl.get_sdk_max_projection(ophys_experiment_id)
    ax.imshow(max_projection, cmap='gray', vmax=np.amax(max_projection)/2.)
    ax.axis('off')
    return ax


