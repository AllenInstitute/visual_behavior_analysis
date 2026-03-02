"""
Paper Figure 1 - Dataset Summary / Experimental Overview

This script generates Figure 1 which provides an overview of the platform dataset,
including cell counts, imaging depths, and example stimulus/behavior timeseries data.

Figure panels:
  - Panel A: Example max intensity projection and ROI outlines
  - Panel B: Stimulus blocks + all traces heatmap + behavior timeseries (combined multi-panel)
  - Panel C: Cell count histograms by depth for the platform dataset
  - Panel D: Number of cells per plane by depth
  - Panel E: Number of planes per depth
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache

import visual_behavior.data_access.utilities as utilities
from visual_behavior.data_access import loading as loading

import visual_behavior.visualization.utils as utils
import visual_behavior.visualization.ophys.platform_paper_figures as ppf
import visual_behavior.visualization.ophys.platform_single_cell_examples as pse


# ============================================================================
# CONFIGURATION
# ============================================================================
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})

# Get or create save directory using relative path
save_dir = os.path.join(os.getcwd(), 'platform_paper_figures', 'figure_1')
os.makedirs(save_dir, exist_ok=True)

folder = 'dataset_summary'


# ============================================================================
# DATA LOADING
# ============================================================================

# Load cache and get experiment/cell tables
platform_cache_dir = loading.get_platform_analysis_cache_dir()
cache = VisualBehaviorOphysProjectCache.from_local_cache(
    cache_dir=platform_cache_dir, use_static_cache=True
)

ophys_experiment_table = cache.get_ophys_experiment_table()
ophys_experiment_table = utilities.add_extra_columns_to_experiment_table(ophys_experiment_table)
ophys_cells_table = cache.get_ophys_cells_table()
ophys_cells_table = ophys_cells_table.merge(ophys_experiment_table, on='ophys_experiment_id')

# Load metadata tables from platform cache
experiments_table = pd.read_csv(
    os.path.join(platform_cache_dir, 'all_ophys_experiments_table.csv'), index_col=0
)
platform_experiments = pd.read_csv(
    os.path.join(platform_cache_dir, 'platform_paper_ophys_experiments_table.csv'), index_col=0
)
platform_cells_table = pd.read_csv(
    os.path.join(platform_cache_dir, 'platform_paper_ophys_cells_table.csv'), index_col=0
)
matched_cells_table = pd.read_csv(
    os.path.join(platform_cache_dir, 'platform_paper_matched_ophys_cells_table.csv'), index_col=0
)

# Get utility lists and dicts for plotting
cre_lines = np.sort(platform_cells_table.cre_line.unique())
cell_types = utilities.get_cell_types_dict(cre_lines, platform_experiments)
palette = utilities.get_experience_level_colors()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# add_cell_type_column is available via: utilities.add_cell_type_column()
# plot_max_intensity_projection is available via: ppf.plot_max_intensity_projection() or pse.plot_max_intensity_projection()
# plot_roi_mask_outlines is available via: pse.plot_roi_mask_outlines()


def plot_expt_count_by_depth_tmp(expts_table, suptitle=None, save_dir=None, folder=None, ax=None):
    """Plot experiment counts as histograms by imaging depth for each cell type."""
    colors = [sns.color_palette()[9], sns.color_palette()[0]]
    if ax is None:
        figsize = (12, 3)
        fig, ax = plt.subplots(1, 3, figsize=figsize, sharey=True)
    for i, cell_type in enumerate(utils.get_cell_types()):
        ax[i] = sns.histplot(
            data=expts_table[expts_table.cell_type == cell_type],
            bins=10,
            y='imaging_depth',
            palette=colors,
            multiple='stack',
            stat='count',
            ax=ax[i]
        )
        title = (cell_type + '\n' + str(len(expts_table[expts_table.cell_type == cell_type])) +
                ' cells, ' + str(len(expts_table[expts_table.cell_type == cell_type].mouse_id.unique())) + ' mice')
        ax[i].set_title(title)
        ax[i].invert_yaxis()
        ax[i].set_ylim(400, 50)
        ax[i].set_xlabel('Experiment count')
        ax[i].set_ylabel('')
    ax[0].set_ylabel('Imaging depth (um)')
    plt.subplots_adjust(wspace=0.3)
    plt.suptitle(suptitle, x=0.5, y=1.2)

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'expt_count_by_depth_' + suptitle)
    return ax


# ============================================================================
# PANEL A: Example max intensity projection and ROI outlines
# ============================================================================
# NOTE: This panel requires a specific dataset_dict which needs to be generated
# by loading data for a specific mouse and session. This is commented out as it
# requires substantial memory and specific dataset configuration.
#
# To generate this panel:
# 1. Select a mouse_id and ophys_session_id
# 2. Load the dataset_dict using loading.get_dataset_dict_for_containers()
# 3. Call plot_roi_mask_outlines() on the desired container/experiment

# Load dataset for example mouse and session
mouse_id = 449653
mouse_expts = experiments_table[experiments_table.mouse_id == mouse_id].sort_values(
    by=['date_of_acquisition', 'targeted_structure', 'imaging_depth']
)
session_expts = mouse_expts[(mouse_expts.session_type == 'OPHYS_4_images_B') &
                            (mouse_expts.experience_level == 'Novel 1')]
ophys_container_ids = session_expts.ophys_container_id.unique()

dataset_dict = loading.get_dataset_dict_for_containers(
    ophys_container_ids,
    session_expts,
    dataset_dict=None
)

# Plot max projections with ROI outlines for all FOVs across sessions for one mouse
session_id_for_area_depths = session_expts.ophys_session_id.unique()[0]
ppf.plot_all_planes_all_sessions_for_mouse(dataset_dict, mouse_expts, session_id_for_area_depths,
                                           save_dir=save_dir, folder='max_projection_images', ax=None)

# Plot matched ROI max projections for a specific container
ophys_container_id = session_expts.ophys_container_id.unique()[0]
ophys_experiment_id = session_expts[session_expts.ophys_container_id == ophys_container_id].index.values[0]
dataset = dataset_dict[ophys_container_id][ophys_experiment_id]

# Plot max projection with ROI outlines
pse.plot_max_and_roi_outlines_for_container(ophys_container_id, platform_experiments,
                                            matched_cells_table, dataset_dict,
                                            cells_to_label=None, save_dir=save_dir, ax=None)

# Plot ROI mask outlines on max projection
pse.plot_roi_mask_outlines(dataset, cell_specimen_ids=None, ax=None)


# ============================================================================
# PANEL B: Stimulus blocks + All traces heatmap + Behavior timeseries (combined)
# ============================================================================
# This section loads data for an example session and creates a combined figure
# with stimulus blocks, neural traces heatmap, and behavior timeseries

# Select mouse and filter to sessions with good data
mouse_id = 449653  # Vip example
mouse_expts = experiments_table[experiments_table.mouse_id == mouse_id].sort_values(
    by=['date_of_acquisition', 'targeted_structure', 'imaging_depth']
)

# Get sessions and load dataset
session_expts = mouse_expts[(mouse_expts.session_type == 'OPHYS_4_images_B') &
                             (mouse_expts.experience_level == 'Novel 1')]
ophys_container_ids = session_expts.ophys_container_id.unique()

dataset_dict = loading.get_dataset_dict_for_containers(
    ophys_container_ids,
    session_expts,
    dataset_dict=None
)

# Aggregate traces for this session
all_traces, session_metadata = ppf.aggregate_traces_for_session(
    dataset_dict, session_expts, trace_type='dff'
)

# Get a representative container for the combined plot
ophys_container_id = session_expts.ophys_container_id.unique()[0]
ophys_experiment_id = session_expts[session_expts.ophys_container_id == ophys_container_id].index.values[0]
ophys_session_id = session_expts.ophys_session_id.unique()[0]

dataset = dataset_dict[ophys_container_id][ophys_experiment_id]
stim_table = dataset.stimulus_presentations.copy()
timestamps = dataset.ophys_timestamps.copy()
xlim_seconds = [timestamps[0], timestamps[-1]]

# Create combined figure with stimulus blocks, traces heatmap, and behavior
figsize = [16, 12]
fig = plt.figure(figsize=figsize, facecolor='white')

# Sub-panel B1: Stimulus blocks
ax = utils.placeAxesOnGrid(fig, dim=(1, 1), xspan=(0, 0.75), yspan=(0, 0.08))
ax = ppf.add_stimulus_blocks(stim_table, xlim=xlim_seconds, annotate_blocks=True, ax=ax)
ax.set_xticklabels('')
ax.set_xlabel('')

# Sub-panel B2: All traces heatmap
ax = utils.placeAxesOnGrid(fig, dim=(1, 1), xspan=(0, 1), yspan=(0.12, 0.65))
ax = ppf.plot_all_traces_heatmap(all_traces, session_metadata, timestamps=timestamps,
                                  cmap='gray_r', save_dir=None, ax=ax)
ax.set_xticklabels('')
ax.set_xlabel('')

# Sub-panel B3: Simple behavior timeseries (non-stacked)
ax_behavior = utils.placeAxesOnGrid(fig, dim=(1, 1), xspan=(0, 0.75), yspan=(0.68, 0.72))
ax_behavior = ppf.plot_behavior_timeseries(dataset, start_time=timestamps[0],
                                           xlim_seconds=xlim_seconds, save_dir=None, ax=ax_behavior)
ax_behavior.set_xticklabels('')
ax_behavior.set_xlabel('')

# Sub-panel B4-B6: Behavior timeseries (stacked format with 4 subplots)
ax = utils.placeAxesOnGrid(fig, dim=(4, 1), xspan=(0, 0.75), yspan=(0.7, 1),
                            sharex=True, height_ratios=[1, 1, 3, 3])
ax = ppf.plot_behavior_timeseries_stacked(dataset, start_time=timestamps[0],
                                           xlim_seconds=xlim_seconds,
                                           show_images=False, label_changes=False,
                                           label_omissions=False, save_dir=None, ax=ax)

# Save the combined figure
folder_combined = 'combined_panels'
utils.save_figure(fig, figsize, save_dir, folder_combined,
                  str(mouse_id) + '_' + str(ophys_session_id) + '_traces_and_behavior')


# ============================================================================
# PANEL C: Cell count histograms by depth (Platform dataset)
# ============================================================================
cells = platform_cells_table.drop_duplicates('cell_specimen_id')
suptitle = 'Platform dataset'

ppf.plot_cell_count_by_depth(cells, suptitle=suptitle, save_dir=save_dir,
                             folder=folder, ax=None)


# ============================================================================
# PANEL D: Number of cells per plane by depth
# ============================================================================
tmp = platform_cells_table.copy()
ppf.plot_n_cells_per_plane_by_depth(tmp, suptitle=suptitle, save_dir=save_dir,
                                     folder=folder, ax=None)


# ============================================================================
# PANEL E: Number of planes per depth
# ============================================================================
ppf.plot_n_planes_per_depth(tmp, suptitle=suptitle, save_dir=save_dir,
                            folder=folder, ax=None)


# ============================================================================
# COMPOSITE FIGURE 1: All panels on a single figure axis
# ============================================================================
# Layout based on published Figure 1 (Allen Brain Observatory Visual Behavior 2P dataset):
#   Row 1 (~15%): Panel A (pipeline schematic, ~60%) | Panel D (active/passive sessions, ~20%) | Panel E (image sets, ~20%)
#   Row 2 (~25%): Panel B (GCaMP6 cell images, ~25%) | Panel C (multi-plane longitudinal imaging, ~75%)
#   Row 3 (~55%): Panel F (full session: stimulus blocks + traces heatmap + behavior, full width)
#
# NOTE: Panels A, D, E are schematics created outside of code — these are left as
# labeled placeholders for placement in Illustrator. Panels B, C, and F are generated
# from data and plotted directly onto the composite figure.

def plot_figure_1_composite(dataset_dict, session_expts, mouse_expts, mouse_id,
                            ophys_session_id, ophys_container_id, ophys_experiment_id,
                            all_traces, session_metadata, dataset, stim_table, timestamps,
                            platform_cells_table, platform_experiments, matched_cells_table,
                            save_dir=None, folder='composite_figures'):
    """
    Generate a composite Figure 1 with all panels arranged in the published layout.
    Saved as PDF for Illustrator editing.
    """
    figsize = (22, 22)
    fig = plt.figure(figsize=figsize, facecolor='white')

    # ---- Row 1: Panels A, D, E (schematics — placeholders) ----
    ax_A = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 0.55], yspan=[0, 0.12])
    ax_A.text(0.5, 0.5, 'A — Pipeline schematic\n(place manually)', transform=ax_A.transAxes,
              ha='center', va='center', fontsize=11, color='gray')
    ax_A.set_title('A', loc='left', fontweight='bold', fontsize=16)
    ax_A.axis('off')

    ax_D = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.58, 0.78], yspan=[0, 0.12])
    ax_D.text(0.5, 0.5, 'D — Active/passive\nsessions schematic', transform=ax_D.transAxes,
              ha='center', va='center', fontsize=10, color='gray')
    ax_D.set_title('D', loc='left', fontweight='bold', fontsize=16)
    ax_D.axis('off')

    ax_E = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.80, 1.0], yspan=[0, 0.12])
    ax_E.text(0.5, 0.5, 'E — Familiar/novel\nimage sets', transform=ax_E.transAxes,
              ha='center', va='center', fontsize=10, color='gray')
    ax_E.set_title('E', loc='left', fontweight='bold', fontsize=16)
    ax_E.axis('off')

    # ---- Row 2: Panel B (GCaMP images) | Panel C (multi-plane longitudinal) ----
    ax_B = utils.placeAxesOnGrid(fig, dim=[1, 3], xspan=[0, 0.25], yspan=[0.15, 0.32],
                                  wspace=0.1)
    try:
        # Plot max projection with ROI outlines for the 3 cell types
        pse.plot_roi_mask_outlines(dataset, cell_specimen_ids=None, ax=ax_B[0] if isinstance(ax_B, list) else ax_B)
    except Exception:
        pass
    if isinstance(ax_B, list):
        ax_B[0].set_title('B', loc='left', fontweight='bold', fontsize=16)
    else:
        ax_B.set_title('B', loc='left', fontweight='bold', fontsize=16)

    ax_C = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.28, 1.0], yspan=[0.15, 0.32])
    ax_C.text(0.5, 0.5, 'C — Multi-plane longitudinal imaging\n(place max projection grid manually)',
              transform=ax_C.transAxes, ha='center', va='center', fontsize=11, color='gray')
    ax_C.set_title('C', loc='left', fontweight='bold', fontsize=16)
    ax_C.axis('off')

    # ---- Row 3: Panel F (full session traces + behavior, full width) ----
    xlim_seconds_F = [timestamps[0], timestamps[-1]]

    # F1: Stimulus blocks
    ax_F1 = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.05, 0.85], yspan=[0.36, 0.40])
    ax_F1 = ppf.add_stimulus_blocks(stim_table, xlim=xlim_seconds_F, annotate_blocks=True, ax=ax_F1)
    ax_F1.set_xticklabels('')
    ax_F1.set_xlabel('')
    ax_F1.set_title('F', loc='left', fontweight='bold', fontsize=16)

    # F2: All traces heatmap
    ax_F2 = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.05, 1.0], yspan=[0.42, 0.75])
    ax_F2 = ppf.plot_all_traces_heatmap(all_traces, session_metadata, timestamps=timestamps,
                                         cmap='gray_r', save_dir=None, ax=ax_F2)
    ax_F2.set_xticklabels('')
    ax_F2.set_xlabel('')

    # F3: Behavior timeseries (stacked: stimulus, licking, running, pupil)
    ax_F3 = utils.placeAxesOnGrid(fig, dim=[4, 1], xspan=[0.05, 0.85], yspan=[0.77, 0.98],
                                   sharex=True, height_ratios=[1, 1, 3, 3])
    ax_F3 = ppf.plot_behavior_timeseries_stacked(dataset, start_time=timestamps[0],
                                                  xlim_seconds=xlim_seconds_F,
                                                  show_images=False, label_changes=False,
                                                  label_omissions=False, save_dir=None, ax=ax_F3)

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'figure_1_composite', formats=['.pdf'])
    return fig


# Generate composite figure
fig1 = plot_figure_1_composite(
    dataset_dict, session_expts, mouse_expts, mouse_id,
    session_id_for_area_depths, ophys_container_id, ophys_experiment_id,
    all_traces, session_metadata, dataset, stim_table, timestamps,
    platform_cells_table, platform_experiments, matched_cells_table,
    save_dir=save_dir, folder='composite_figures'
)
