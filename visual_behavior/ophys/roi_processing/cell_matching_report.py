from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession
from allensdk.internal.api.behavior_ophys_api import BehaviorOphysLimsApi
import visual_behavior.ophys.roi_processing.segmentation_report as seg
import visual_behavior.ophys.roi_processing.roi_processing as roi
from allensdk.internal.api import PostgresQueryMixin
from psycopg2 import connect, extras

import matplotlib.pyplot as plt


import scipy.stats as stats

import SimpleITK as sitk
import networkx as nx
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os




# def plot_cell_zoom(roi_mask, max_projection, cell_specimen_id, spacex=10, spacey=10, show_mask=False, ax=None):
#     m = roi_mask 
#     (y, x) = np.where(m == 1)
#     xmin = np.min(x)
#     xmax = np.max(x)
#     ymin = np.min(y)
#     ymax = np.max(y)
#     mask = np.empty(m.shape)
#     mask[:] = np.nan
#     mask[y, x] = 1
#     if ax is None:
#         fig, ax = plt.subplots()
#     ax.imshow(max_projection, cmap='gray', vmin=0, vmax=np.amax(max_projection))
#     if show_mask:
#         ax.imshow(mask, cmap='jet', alpha=0.3, vmin=0, vmax=1)
#     ax.set_xlim(xmin - spacex, xmax + spacex)
#     ax.set_ylim(ymin - spacey, ymax + spacey)
#     ax.set_title('cell ' + str(cell_specimen_id))
#     ax.grid(False)
#     ax.axis('off')
#     return ax