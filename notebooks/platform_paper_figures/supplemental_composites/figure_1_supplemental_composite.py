"""
Supplemental Figures S1-S2 Composite Script (associated with Paper Figure 1)

This script generates composite figures for supplemental figures S1 and S2,
which show cell count analysis by depth and ROI size characteristics.

S1: Cell counts by depth
- Row 1 (A): 3 histograms for mesoscope data (Exc, Sst, Vip)
- Row 2 (B): 3 histograms for platform dataset (Exc, Sst, Vip)
- Row 3 (C): 3 histograms for full dataset excluding Ai94 (Exc, Sst, Vip)
- Row 4 (D): Vertical cell count by depth with all areas
- Row 5 (E-H): 4 columns, one per cohort (VisualBehavior, Task1B, Multiscope, 4areas)

S2: ROI size analysis
- Row 1 (A): 3 KDE plots of roi_size by depth (per cell type)
- Row 2 (B): Boxplot of roi_size by cell type and depth | (C): 3 KDE plots of roi_width_over_height
- Row 3 (D): Matched cell ROI examples (placeholder)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import visual_behavior.data_access.utilities as utilities
from visual_behavior.data_access import loading as loading

import visual_behavior.visualization.utils as utils
import visual_behavior.visualization.ophys.platform_paper_figures as ppf
import visual_behavior.visualization.ophys.platform_single_cell_examples as pse

from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache


# ============================================================================
# Configuration
# ============================================================================

sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})

save_dir = os.path.join(os.getcwd(), 'platform_paper_figures', 'supplemental_composites')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# ============================================================================
# Data Loading
# ============================================================================

platform_cache_dir = loading.get_platform_analysis_cache_dir()
cache = VisualBehaviorOphysProjectCache.from_local_cache(
    cache_dir=platform_cache_dir,
    use_static_cache=True
)

ophys_experiment_table = cache.get_ophys_experiment_table()
ophys_experiment_table = utilities.add_extra_columns_to_experiment_table(ophys_experiment_table)
ophys_cells_table = cache.get_ophys_cells_table()
ophys_cells_table = ophys_cells_table.merge(ophys_experiment_table, on='ophys_experiment_id')

experiments_table = pd.read_csv(
    os.path.join(platform_cache_dir, 'all_ophys_experiments_table.csv'),
    index_col=0
)
platform_experiments = pd.read_csv(
    os.path.join(platform_cache_dir, 'platform_paper_ophys_experiments_table.csv'),
    index_col=0
)
platform_cells_table = pd.read_csv(
    os.path.join(platform_cache_dir, 'platform_paper_ophys_cells_table.csv'),
    index_col=0
)
matched_cells_table = pd.read_csv(
    os.path.join(platform_cache_dir, 'platform_paper_matched_ophys_cells_table.csv'),
    index_col=0
)

matched_cells = matched_cells_table.cell_specimen_id.unique()
matched_experiments = matched_cells_table.ophys_experiment_id.unique()

cre_lines = np.sort(platform_cells_table.cre_line.unique())
cell_types = utilities.get_cell_types_dict(cre_lines, platform_experiments)

palette = utilities.get_experience_level_colors()
cell_types_list = utils.get_cell_types()

# Load cell specimen measurements for mesoscope
meso_cell_dir = os.path.join(platform_cache_dir, 'cell_specimen_tables')
cells_meso = pd.read_csv(
    os.path.join(meso_cell_dir, 'meso_cell_specimen_table.csv'),
    index_col=0
)

# Load GLM results
results_pivoted = pd.read_hdf(
    os.path.join(platform_cache_dir, 'glm_results', 'across_session_normalized_platform_results_pivoted.h5'),
    key='df'
)


# ============================================================================
# Helper Functions
# ============================================================================

def plot_expt_count_by_depth_tmp(expts_table, suptitle=None, save_dir=None, folder=None, ax=None):
    """Plot histogram of experiment counts by imaging depth"""
    colors = [sns.color_palette()[9], sns.color_palette()[0]]
    if ax is None:
        figsize = (12, 3)
        fig, ax = plt.subplots(1, 3, figsize=figsize, sharey=True)
    for i, cell_type in enumerate(utils.get_cell_types()):
        ax[i] = sns.histplot(
            data=expts_table[expts_table.cell_type == cell_type],
            bins=10, y='imaging_depth',
            palette=colors, multiple='stack', stat='count', ax=ax[i]
        )
        title = (cell_type + '\n' + str(len(expts_table[expts_table.cell_type == cell_type])) +
                 ' cells, ' + str(len(expts_table[expts_table.cell_type == cell_type].mouse_id.unique())) +
                 ' mice')
        ax[i].set_title(title)
        ax[i].invert_yaxis()
        ax[i].set_ylim(400, 50)
        ax[i].set_xlabel('Experiment count')
        ax[i].set_ylabel('')
    ax[0].set_ylabel('Imaging depth (um)')
    plt.subplots_adjust(wspace=0.3)
    plt.suptitle(suptitle, x=0.5, y=1.2)

    if save_dir:
        utils.save_figure(fig, (12, 3), save_dir, folder, 'expt_count_by_depth_' + suptitle)
    return ax


# ============================================================================
# COMPOSITE FIGURE S1: Cell counts by depth (multi-row layout)
# ============================================================================

def plot_figure_S1_composite(platform_cells_table, ophys_cells_table, cells_meso,
                             save_dir=None, folder='supplemental_composites'):
    """
    Generate composite Figure S1 showing cell counts by depth across different dataset subsets.

    Layout (5 rows):
    - Row 1 (A): Mesoscope data histograms (3 cell types)
    - Row 2 (B): Platform dataset histograms (3 cell types)
    - Row 3 (C): Full dataset (excluding Ai94) histograms (3 cell types)
    - Row 4 (D): Vertical cell count by depth with all areas
    - Row 5 (E-H): 4 cohort comparisons (one column each)
    """
    figsize = (22, 30)
    fig = plt.figure(figsize=figsize, facecolor='white')

    # ---- Row 1 (A): Mesoscope data - 3 cell type histograms ----
    cells = platform_cells_table.drop_duplicates('cell_specimen_id')
    cells_meso_subset = cells[cells.project_code == 'VisualBehaviorMultiscope']
    suptitle = 'VisualBehaviorMultiscope'

    ax_A = utils.placeAxesOnGrid(fig, dim=[1, 3], xspan=[0, 1], yspan=[0, 0.08], wspace=0.3)
    ppf.plot_cell_count_by_depth(cells_meso_subset, suptitle=None, save_dir=None, folder=None, ax=ax_A[0] if isinstance(ax_A, list) else ax_A)
    if isinstance(ax_A, list):
        ax_A[0].set_title('A  Mesoscope data', loc='left', fontweight='bold', fontsize=16)
    else:
        ax_A.set_title('A  Mesoscope data', loc='left', fontweight='bold', fontsize=16)

    # ---- Row 2 (B): Platform dataset - 3 cell type histograms ----
    cells = platform_cells_table.drop_duplicates('cell_specimen_id')
    suptitle = 'Platform dataset'

    ax_B = utils.placeAxesOnGrid(fig, dim=[1, 3], xspan=[0, 1], yspan=[0.10, 0.18], wspace=0.3)
    ppf.plot_cell_count_by_depth(cells, suptitle=None, save_dir=None, folder=None, ax=ax_B[0] if isinstance(ax_B, list) else ax_B)
    if isinstance(ax_B, list):
        ax_B[0].set_title('B  Platform dataset', loc='left', fontweight='bold', fontsize=16)
    else:
        ax_B.set_title('B  Platform dataset', loc='left', fontweight='bold', fontsize=16)

    # ---- Row 3 (C): Full dataset (excluding Ai94) - 3 cell type histograms ----
    cells = ophys_cells_table.drop_duplicates('cell_specimen_id')
    cells = cells[cells.reporter_line.str.contains('Ai94') == False]
    suptitle = 'full dataset (other than Ai94)'

    ax_C = utils.placeAxesOnGrid(fig, dim=[1, 3], xspan=[0, 1], yspan=[0.20, 0.28], wspace=0.3)
    ppf.plot_cell_count_by_depth(cells, suptitle=None, save_dir=None, folder=None, ax=ax_C[0] if isinstance(ax_C, list) else ax_C)
    if isinstance(ax_C, list):
        ax_C[0].set_title('C  Full dataset (excluding Ai94)', loc='left', fontweight='bold', fontsize=16)
    else:
        ax_C.set_title('C  Full dataset (excluding Ai94)', loc='left', fontweight='bold', fontsize=16)

    # ---- Row 4 (D): Vertical cell count by depth with all areas ----
    cells = ophys_cells_table.drop_duplicates('cell_specimen_id')
    cells = cells[cells.reporter_line.str.contains('Ai94') == False]
    suptitle = 'Full dataset (excluding Ai94)'
    suffix = '_full_dataset'

    ax_D = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 1], yspan=[0.30, 0.45])
    ax_D.text(0.5, 0.5, 'D — Vertical cell count by depth with all areas\n(figure created separately)',
              transform=ax_D.transAxes, ha='center', va='center', fontsize=11, color='gray')
    ax_D.set_title('D', loc='left', fontweight='bold', fontsize=16)
    ax_D.axis('off')

    # ---- Row 5 (E-H): 4 cohort comparisons ----
    cells = ophys_cells_table.drop_duplicates('cell_specimen_id')
    cells = cells[cells.reporter_line.str.contains('Ai94') == False]

    project_codes = ['VisualBehavior', 'VisualBehaviorTask1B', 'VisualBehaviorMultiscope', 'VisualBehaviorMultiscope4areasx2d']
    cohort_labels = ['E  VisualBehavior', 'F  Task1B', 'G  Multiscope', 'H  4areas']

    for i, (project_code, cohort_label) in enumerate(zip(project_codes, cohort_labels)):
        proj_cells = cells[cells.project_code == project_code]
        xstart = i * 0.25
        xend = (i + 1) * 0.25

        ax_cohort = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[xstart, xend], yspan=[0.47, 0.62])
        ax_cohort.text(0.5, 0.5, f'{cohort_label}\n(figure created separately)',
                       transform=ax_cohort.transAxes, ha='center', va='center', fontsize=10, color='gray')
        ax_cohort.set_title(cohort_label.split()[0], loc='left', fontweight='bold', fontsize=16)
        ax_cohort.axis('off')

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'figure_S1_composite', formats=['.pdf'])
    return fig


# ============================================================================
# COMPOSITE FIGURE S2: ROI size analysis (3-row layout)
# ============================================================================

def plot_figure_S2_composite(cells_meso, matched_cells_table, results_pivoted,
                             save_dir=None, folder='supplemental_composites'):
    """
    Generate composite Figure S2 showing ROI size and morphology analysis.

    Layout (3 rows):
    - Row 1 (A): KDE plots of roi_size by depth (3 cell types)
    - Row 2 (B): Boxplot of roi_size by cell type and depth | (C): KDE plots of roi_width_over_height (3 cell types)
    - Row 3 (D): Matched cell ROI examples (placeholder, grid layout)
    """
    figsize = (22, 20)
    fig = plt.figure(figsize=figsize, facecolor='white')

    # ---- Row 1 (A): KDE plots of roi_size by depth per cell type ----
    ax_A = utils.placeAxesOnGrid(fig, dim=[1, 3], xspan=[0, 1], yspan=[0, 0.15], wspace=0.3)

    for i, cell_type in enumerate(utils.get_cell_types()):
        ct_cells = cells_meso[cells_meso.cell_type == cell_type]
        ax_current = ax_A[i] if isinstance(ax_A, list) else ax_A
        ax_current = sns.kdeplot(data=ct_cells, y='imaging_depth', x='roi_size', ax=ax_current)
        ax_current.set_title(cell_type)
        ax_current.invert_yaxis()
        ax_current.set_xlabel('Cell size (pixels)')
        ax_current.set_ylabel('Imaging depth (um)')

    ax_A_first = ax_A[0] if isinstance(ax_A, list) else ax_A
    ax_A_first.set_title('A  ROI size by depth (KDE)', loc='left', fontweight='bold', fontsize=16)

    # ---- Row 2: Boxplot (B) and KDE plots (C) ----
    # Panel B: Boxplot of roi_size by cell type and binned depth
    ax_B = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 0.45], yspan=[0.18, 0.38])
    ax_B = sns.boxplot(data=cells_meso, x='cell_type', y='roi_size', hue='binned_depth', ax=ax_B)
    ax_B.set_ylabel('Cell size (pixels)')
    ax_B.set_xlabel('Cell type')
    ax_B.legend(bbox_to_anchor=(1, 1), fontsize='xx-small')
    ax_B.set_title('B  ROI size by cell type and depth', loc='left', fontweight='bold', fontsize=16)

    # Panel C: KDE plots of roi_width_over_height by depth per cell type
    ax_C = utils.placeAxesOnGrid(fig, dim=[1, 3], xspan=[0.50, 1], yspan=[0.18, 0.38], wspace=0.3)

    for i, cell_type in enumerate(utils.get_cell_types()):
        ct_cells = cells_meso[cells_meso.cell_type == cell_type]
        ax_current = ax_C[i] if isinstance(ax_C, list) else ax_C
        ax_current = sns.kdeplot(data=ct_cells, y='imaging_depth', x='roi_width_over_height', ax=ax_current)
        ax_current.set_title(cell_type)
        ax_current.invert_yaxis()
        ax_current.set_xlabel('Width/Height ratio')
        ax_current.set_ylabel('Imaging depth (um)')

    ax_C_first = ax_C[0] if isinstance(ax_C, list) else ax_C
    ax_C_first.set_title('C  ROI width/height by depth (KDE)', loc='left', fontweight='bold', fontsize=16)

    # ---- Row 3 (D): Matched cell ROI examples (placeholder 2x3 grid) ----
    ax_D = utils.placeAxesOnGrid(fig, dim=[2, 3], xspan=[0, 1], yspan=[0.42, 0.95], wspace=0.2, hspace=0.3)

    # Create placeholders for the 2x3 grid of matched cell examples
    flat_axes = ax_D if isinstance(ax_D, list) else [ax_D]
    for idx, ax_placeholder in enumerate(flat_axes):
        ax_placeholder.text(0.5, 0.5, f'Matched cell example {idx + 1}',
                           transform=ax_placeholder.transAxes, ha='center', va='center',
                           fontsize=10, color='gray')
        ax_placeholder.set_title(chr(68 + idx // 3), loc='left', fontweight='bold', fontsize=14)  # D, E, F, etc.
        ax_placeholder.axis('off')

    # Set the main title for row 3
    if isinstance(ax_D, list) and len(ax_D) > 0:
        ax_D[0].set_title('D  Matched cell ROI examples\n(figures created separately)',
                          loc='left', fontweight='bold', fontsize=16)

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'figure_S2_composite', formats=['.pdf'])
    return fig


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    # Generate S1 composite
    fig_S1 = plot_figure_S1_composite(
        platform_cells_table, ophys_cells_table, cells_meso,
        save_dir=save_dir, folder='supplemental_composites'
    )
    print("Figure S1 composite generated and saved.")

    # Generate S2 composite
    fig_S2 = plot_figure_S2_composite(
        cells_meso, matched_cells_table, results_pivoted,
        save_dir=save_dir, folder='supplemental_composites'
    )
    print("Figure S2 composite generated and saved.")

    print(f"\nAll composite figures saved to: {save_dir}")
