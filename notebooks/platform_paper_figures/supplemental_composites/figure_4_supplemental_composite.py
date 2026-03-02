"""
Composite Figure Scripts for Supplemental Figures S10-S15 (Figure 4 supplemental).

This script creates composite figures combining multiple GLM analysis panels:
- S10: GLM dropout summaries (4 panels)
- S11: Kernel heatmaps for images (3 panels)
- S12: Kernel heatmaps for omissions (3 panels)
- S13: Extended kernel analysis (4 panels)
- S14: Coding by area and depth (4 panels)
- S15: Coding score distributions (4 panels)

Layout strategy: Each figure creates placeholder axes using placeAxesOnGrid
to match the expected panel layout, with manual plot placement or function calls
for figures that generate their own output.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import visual_behavior.data_access.loading as loading
import visual_behavior.data_access.utilities as utilities
import visual_behavior.visualization.utils as utils
from visual_behavior.dimensionality_reduction.clustering import processing
from visual_behavior.dimensionality_reduction.clustering import plotting

import visual_behavior.visualization.ophys.glm_example_plots as gep
import visual_behavior_glm.GLM_visualization_tools as gvt


# ============================================================================
# Configuration
# ============================================================================

sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 1})

save_dir = os.path.join(os.getcwd(), 'platform_paper_figures', 'supplemental_composites')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Color mappings
experience_level_colors = utils.get_experience_level_colors()
cre_line_colors = utils.get_cre_line_colors()
experience_levels = utils.get_new_experience_levels()
cell_types = utils.get_cell_types()


# ============================================================================
# Data Loading
# ============================================================================

# Metadata tables
experiments_table = pd.read_csv(
    os.path.join(loading.get_metadata_tables_dir(), 'all_ophys_experiments_table.csv'),
    index_col=0
)
platform_experiments = pd.read_csv(
    os.path.join(loading.get_metadata_tables_dir(), 'platform_paper_ophys_experiments_table.csv'),
    index_col=0
)
platform_cells_table = pd.read_csv(
    os.path.join(loading.get_metadata_tables_dir(), 'platform_paper_ophys_cells_table.csv'),
    index_col=0
)
matched_cells_table = pd.read_csv(
    os.path.join(loading.get_metadata_tables_dir(), 'platform_paper_matched_ophys_cells_table.csv'),
    index_col=0
)

# Define model version and load GLM results
glm_version = '24_events_all_L2_optimize_by_session'

run_params = pd.read_pickle(
    os.path.join(loading.get_glm_results_dir(), glm_version + '_run_params.pkl')
)
run_params['figure_dir'] = os.path.join(save_dir, 'coding_scores_and_kernels')

kernels = run_params['kernels']
all_weights_df = pd.read_hdf(
    os.path.join(loading.get_glm_results_dir(), 'weights_df.h5'),
    key='df'
)
all_results = pd.read_hdf(
    os.path.join(loading.get_glm_results_dir(), 'all_results.h5'),
    key='df'
)
all_results_pivoted = pd.read_hdf(
    os.path.join(loading.get_glm_results_dir(), 'results_pivoted.h5'),
    key='df'
)
all_results_pivoted = processing.set_negative_one_values_to_zero(all_results_pivoted)

# Split out cell_specimen_id and ophys_experiment_id for results table
all_results['cell_specimen_id'] = [
    int(identifier.split('_')[1])
    for identifier in all_results.identifier.values
]
all_results['ophys_experiment_id'] = [
    int(identifier.split('_')[0])
    for identifier in all_results.identifier.values
]

# Get results and weights just for platform paper experiments
results_pivoted = pd.read_hdf(
    os.path.join(loading.get_glm_results_dir(), 'platform_results_pivoted.h5'),
    key='df'
)
results_pivoted = processing.set_negative_one_values_to_zero(results_pivoted)

# Make coding scores positive
results_pivoted[processing.get_features_for_clustering()] = np.abs(
    results_pivoted[processing.get_features_for_clustering()]
)

weights_df = pd.read_hdf(
    os.path.join(loading.get_glm_results_dir(), 'platform_results_weights_df.h5'),
    key='df'
)

# Get lists of matched cells and experiments
matched_cells = matched_cells_table.cell_specimen_id.unique()
matched_experiments = matched_cells_table.ophys_experiment_id.unique()

# Get cre_lines and cell types for plot labels
cre_lines = np.sort(platform_cells_table.cre_line.unique())
cell_types_dict = utilities.get_cell_types_dict(cre_lines, platform_experiments)

# Filter for non-passive sessions
results_pivoted_filtered = results_pivoted.query('not passive').copy()
results = all_results[
    all_results.ophys_experiment_id.isin(platform_experiments.index.unique())
].copy()


# ============================================================================
# Figure S10 - GLM Dropout Summaries (4 panels)
# ============================================================================

def plot_figure_S10_composite():
    """
    Create composite Figure S10 with GLM dropout summaries.

    Layout:
    - Row 1: A — Dropout summary population plot
    - Row 2: B — Dropout individual population | C — Dropout individual (single)
    - Row 3: D — Variance explained matched cells
    figsize=(22, 22)
    """
    figsize = (22, 22)
    fig = plt.figure(figsize=figsize)

    # Create main grid (3 rows, 2 columns to accommodate B and C side-by-side)
    ax_a = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 1], yspan=[2/3, 1],
                                  wspace=0.3, hspace=0.3)[0, 0]
    ax_b = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 0.5], yspan=[1/3, 2/3],
                                  wspace=0.3, hspace=0.3)[0, 0]
    ax_c = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.5, 1], yspan=[1/3, 2/3],
                                  wspace=0.3, hspace=0.3)[0, 0]
    ax_d = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 1], yspan=[0, 1/3],
                                  wspace=0.3, hspace=0.3)[0, 0]

    # Panel A - Dropout summary population plot
    ax_a.text(0.5, 0.5, 'A — Dropout summary population plot\n(place manually)',
             transform=ax_a.transAxes, ha='center', va='center',
             fontsize=11, color='gray')
    ax_a.set_title('A', loc='left', fontweight='bold', fontsize=16)
    ax_a.axis('off')

    # Panel B - Dropout individual population
    ax_b.text(0.5, 0.5, 'B — Dropout individual population\n(place manually)',
             transform=ax_b.transAxes, ha='center', va='center',
             fontsize=11, color='gray')
    ax_b.set_title('B', loc='left', fontweight='bold', fontsize=16)
    ax_b.axis('off')

    # Panel C - Dropout individual (single)
    ax_c.text(0.5, 0.5, 'C — Dropout individual (single)\n(place manually)',
             transform=ax_c.transAxes, ha='center', va='center',
             fontsize=11, color='gray')
    ax_c.set_title('C', loc='left', fontweight='bold', fontsize=16)
    ax_c.axis('off')

    # Panel D - Variance explained matched cells
    ax_d.text(0.5, 0.5, 'D — Variance explained matched cells\n(place manually)',
             transform=ax_d.transAxes, ha='center', va='center',
             fontsize=11, color='gray')
    ax_d.set_title('D', loc='left', fontweight='bold', fontsize=16)
    ax_d.axis('off')

    plt.tight_layout()
    utils.save_figure(fig, figsize, save_dir, 'composites', 'S10_dropout_summaries', formats=['.pdf'])
    plt.close()
    print("Figure S10 composite created: S10_dropout_summaries.pdf")


# ============================================================================
# Figure S11 - Kernel Heatmaps (Images, 3 panels)
# ============================================================================

def plot_figure_S11_composite():
    """
    Create composite Figure S11 with kernel heatmaps for images.

    Layout:
    - Row 1: A — Image kernel heatmap with dropout by experience
    - Row 2: B — Image kernel heatmap Familiar only
    - Row 3: C — all-images weights and coding score heatmaps
    figsize=(22, 22)
    """
    figsize = (22, 22)
    fig = plt.figure(figsize=figsize)

    # Create axes for 3 rows
    ax_a = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 1], yspan=[2/3, 1],
                                  wspace=0.3, hspace=0.3)[0, 0]
    ax_b = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 1], yspan=[1/3, 2/3],
                                  wspace=0.3, hspace=0.3)[0, 0]
    ax_c = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 1], yspan=[0, 1/3],
                                  wspace=0.3, hspace=0.3)[0, 0]

    # Panel A - Image kernel heatmap with dropout by experience
    ax_a.text(0.5, 0.5, 'A — Image kernel heatmap with dropout\nby experience (Vip, Sst, Slc)',
             transform=ax_a.transAxes, ha='center', va='center',
             fontsize=11, color='gray')
    ax_a.set_title('A', loc='left', fontweight='bold', fontsize=16)
    ax_a.axis('off')

    # Panel B - Image kernel heatmap Familiar only
    ax_b.text(0.5, 0.5, 'B — Image kernel heatmap Familiar only\n(place manually)',
             transform=ax_b.transAxes, ha='center', va='center',
             fontsize=11, color='gray')
    ax_b.set_title('B', loc='left', fontweight='bold', fontsize=16)
    ax_b.axis('off')

    # Panel C - all-images weights and coding score heatmaps
    ax_c.text(0.5, 0.5, 'C — all-images weights and coding score heatmaps\n(place manually)',
             transform=ax_c.transAxes, ha='center', va='center',
             fontsize=11, color='gray')
    ax_c.set_title('C', loc='left', fontweight='bold', fontsize=16)
    ax_c.axis('off')

    plt.tight_layout()
    utils.save_figure(fig, figsize, save_dir, 'composites', 'S11_kernel_heatmaps_images', formats=['.pdf'])
    plt.close()
    print("Figure S11 composite created: S11_kernel_heatmaps_images.pdf")


# ============================================================================
# Figure S12 - Kernel Heatmaps (Omissions, 3 panels)
# ============================================================================

def plot_figure_S12_composite():
    """
    Create composite Figure S12 with kernel heatmaps for omissions.

    Layout:
    - Row 1: A — Omissions kernel heatmap with dropout by experience
    - Row 2: B — Omissions kernel heatmap Familiar only
    - Row 3: C — omissions weights and coding score heatmaps
    figsize=(22, 22)
    """
    figsize = (22, 22)
    fig = plt.figure(figsize=figsize)

    # Create axes for 3 rows
    ax_a = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 1], yspan=[2/3, 1],
                                  wspace=0.3, hspace=0.3)[0, 0]
    ax_b = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 1], yspan=[1/3, 2/3],
                                  wspace=0.3, hspace=0.3)[0, 0]
    ax_c = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 1], yspan=[0, 1/3],
                                  wspace=0.3, hspace=0.3)[0, 0]

    # Panel A - Omissions kernel heatmap with dropout by experience
    ax_a.text(0.5, 0.5, 'A — Omissions kernel heatmap with dropout\nby experience (Vip, Sst, Slc)',
             transform=ax_a.transAxes, ha='center', va='center',
             fontsize=11, color='gray')
    ax_a.set_title('A', loc='left', fontweight='bold', fontsize=16)
    ax_a.axis('off')

    # Panel B - Omissions kernel heatmap Familiar only
    ax_b.text(0.5, 0.5, 'B — Omissions kernel heatmap Familiar only\n(place manually)',
             transform=ax_b.transAxes, ha='center', va='center',
             fontsize=11, color='gray')
    ax_b.set_title('B', loc='left', fontweight='bold', fontsize=16)
    ax_b.axis('off')

    # Panel C - omissions weights and coding score heatmaps
    ax_c.text(0.5, 0.5, 'C — omissions weights and coding score heatmaps\n(place manually)',
             transform=ax_c.transAxes, ha='center', va='center',
             fontsize=11, color='gray')
    ax_c.set_title('C', loc='left', fontweight='bold', fontsize=16)
    ax_c.axis('off')

    plt.tight_layout()
    utils.save_figure(fig, figsize, save_dir, 'composites', 'S12_kernel_heatmaps_omissions', formats=['.pdf'])
    plt.close()
    print("Figure S12 composite created: S12_kernel_heatmaps_omissions.pdf")


# ============================================================================
# Figure S13 - Extended Kernel Analysis (4 panels)
# ============================================================================

def plot_figure_S13_composite():
    """
    Create composite Figure S13 with extended kernel analysis.

    Layout:
    - Row 1: A — hits kernel weights/coding heatmap
    - Row 2: B — misses kernel
    - Row 3: C — running kernel
    - Row 4: D — pupil/licks kernels
    figsize=(22, 28)
    """
    figsize = (22, 28)
    fig = plt.figure(figsize=figsize)

    # Create axes for 4 rows
    ax_a = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 1], yspan=[3/4, 1],
                                  wspace=0.3, hspace=0.3)[0, 0]
    ax_b = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 1], yspan=[2/4, 3/4],
                                  wspace=0.3, hspace=0.3)[0, 0]
    ax_c = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 1], yspan=[1/4, 2/4],
                                  wspace=0.3, hspace=0.3)[0, 0]
    ax_d = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 1], yspan=[0, 1/4],
                                  wspace=0.3, hspace=0.3)[0, 0]

    # Panel A - hits kernel weights/coding heatmap
    ax_a.text(0.5, 0.5, 'A — hits kernel weights/coding heatmap\n(place manually)',
             transform=ax_a.transAxes, ha='center', va='center',
             fontsize=11, color='gray')
    ax_a.set_title('A', loc='left', fontweight='bold', fontsize=16)
    ax_a.axis('off')

    # Panel B - misses kernel
    ax_b.text(0.5, 0.5, 'B — misses kernel weights/coding heatmap\n(place manually)',
             transform=ax_b.transAxes, ha='center', va='center',
             fontsize=11, color='gray')
    ax_b.set_title('B', loc='left', fontweight='bold', fontsize=16)
    ax_b.axis('off')

    # Panel C - running kernel
    ax_c.text(0.5, 0.5, 'C — running kernel weights/coding heatmap\n(place manually)',
             transform=ax_c.transAxes, ha='center', va='center',
             fontsize=11, color='gray')
    ax_c.set_title('C', loc='left', fontweight='bold', fontsize=16)
    ax_c.axis('off')

    # Panel D - pupil/licks kernels
    ax_d.text(0.5, 0.5, 'D — pupil/licks kernels weights/coding\n(place manually)',
             transform=ax_d.transAxes, ha='center', va='center',
             fontsize=11, color='gray')
    ax_d.set_title('D', loc='left', fontweight='bold', fontsize=16)
    ax_d.axis('off')

    plt.tight_layout()
    utils.save_figure(fig, figsize, save_dir, 'composites', 'S13_extended_kernel_analysis', formats=['.pdf'])
    plt.close()
    print("Figure S13 composite created: S13_extended_kernel_analysis.pdf")


# ============================================================================
# Figure S14 - Coding by Area and Depth (4 panels)
# ============================================================================

def plot_figure_S14_composite():
    """
    Create composite Figure S14 with coding scores by area and depth.

    Layout:
    - Row 1: A — Population averages by depth
    - Row 2: B — Population averages by area
    - Row 3: C — ANOVA results images | D — ANOVA results omissions
    figsize=(22, 22)
    """
    figsize = (22, 22)
    fig = plt.figure(figsize=figsize)

    # Create axes
    ax_a = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 1], yspan=[2/3, 1],
                                  wspace=0.3, hspace=0.3)[0, 0]
    ax_b = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 1], yspan=[1/3, 2/3],
                                  wspace=0.3, hspace=0.3)[0, 0]
    ax_c = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 0.5], yspan=[0, 1/3],
                                  wspace=0.3, hspace=0.3)[0, 0]
    ax_d = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.5, 1], yspan=[0, 1/3],
                                  wspace=0.3, hspace=0.3)[0, 0]

    # Panel A - Population averages by depth
    ax_a.text(0.5, 0.5, 'A — Population averages by depth\n(place manually)',
             transform=ax_a.transAxes, ha='center', va='center',
             fontsize=11, color='gray')
    ax_a.set_title('A', loc='left', fontweight='bold', fontsize=16)
    ax_a.axis('off')

    # Panel B - Population averages by area
    ax_b.text(0.5, 0.5, 'B — Population averages by area\n(place manually)',
             transform=ax_b.transAxes, ha='center', va='center',
             fontsize=11, color='gray')
    ax_b.set_title('B', loc='left', fontweight='bold', fontsize=16)
    ax_b.axis('off')

    # Panel C - ANOVA results images
    ax_c.text(0.5, 0.5, 'C — ANOVA results images\n(place manually)',
             transform=ax_c.transAxes, ha='center', va='center',
             fontsize=11, color='gray')
    ax_c.set_title('C', loc='left', fontweight='bold', fontsize=16)
    ax_c.axis('off')

    # Panel D - ANOVA results omissions
    ax_d.text(0.5, 0.5, 'D — ANOVA results omissions\n(place manually)',
             transform=ax_d.transAxes, ha='center', va='center',
             fontsize=11, color='gray')
    ax_d.set_title('D', loc='left', fontweight='bold', fontsize=16)
    ax_d.axis('off')

    plt.tight_layout()
    utils.save_figure(fig, figsize, save_dir, 'composites', 'S14_coding_by_area_depth', formats=['.pdf'])
    plt.close()
    print("Figure S14 composite created: S14_coding_by_area_depth.pdf")


# ============================================================================
# Figure S15 - Coding Score Distributions (4 panels)
# ============================================================================

def plot_figure_S15_composite():
    """
    Create composite Figure S15 with coding score distributions.

    Layout:
    - Row 1: A-D — Feature coding by experience (images, omissions, behavior, task) — 1x4 grid
    - Row 2: E — Dropout summary CDF
    - Row 3: F — Metric-feature coding relationship
    figsize=(22, 20)
    """
    figsize = (22, 20)
    fig = plt.figure(figsize=figsize)

    # Create axes for 1x4 grid in Row 1
    ax_a = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 0.25], yspan=[2/3, 1],
                                  wspace=0.3, hspace=0.3)[0, 0]
    ax_b = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.25, 0.5], yspan=[2/3, 1],
                                  wspace=0.3, hspace=0.3)[0, 0]
    ax_c = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.5, 0.75], yspan=[2/3, 1],
                                  wspace=0.3, hspace=0.3)[0, 0]
    ax_d = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.75, 1], yspan=[2/3, 1],
                                  wspace=0.3, hspace=0.3)[0, 0]

    # Row 2 axes
    ax_e = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 1], yspan=[1/3, 2/3],
                                  wspace=0.3, hspace=0.3)[0, 0]

    # Row 3 axes
    ax_f = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 1], yspan=[0, 1/3],
                                  wspace=0.3, hspace=0.3)[0, 0]

    # Panel A - Feature coding by experience (images)
    ax_a.text(0.5, 0.5, 'A — Feature coding\nimages',
             transform=ax_a.transAxes, ha='center', va='center',
             fontsize=11, color='gray')
    ax_a.set_title('A', loc='left', fontweight='bold', fontsize=16)
    ax_a.axis('off')

    # Panel B - Feature coding by experience (omissions)
    ax_b.text(0.5, 0.5, 'B — Feature coding\nomissions',
             transform=ax_b.transAxes, ha='center', va='center',
             fontsize=11, color='gray')
    ax_b.set_title('B', loc='left', fontweight='bold', fontsize=16)
    ax_b.axis('off')

    # Panel C - Feature coding by experience (behavior)
    ax_c.text(0.5, 0.5, 'C — Feature coding\nbehavior',
             transform=ax_c.transAxes, ha='center', va='center',
             fontsize=11, color='gray')
    ax_c.set_title('C', loc='left', fontweight='bold', fontsize=16)
    ax_c.axis('off')

    # Panel D - Feature coding by experience (task)
    ax_d.text(0.5, 0.5, 'D — Feature coding\ntask',
             transform=ax_d.transAxes, ha='center', va='center',
             fontsize=11, color='gray')
    ax_d.set_title('D', loc='left', fontweight='bold', fontsize=16)
    ax_d.axis('off')

    # Panel E - Dropout summary CDF
    ax_e.text(0.5, 0.5, 'E — Dropout summary CDF\n(place manually)',
             transform=ax_e.transAxes, ha='center', va='center',
             fontsize=11, color='gray')
    ax_e.set_title('E', loc='left', fontweight='bold', fontsize=16)
    ax_e.axis('off')

    # Panel F - Metric-feature coding relationship
    ax_f.text(0.5, 0.5, 'F — Metric-feature coding relationship\n(place manually)',
             transform=ax_f.transAxes, ha='center', va='center',
             fontsize=11, color='gray')
    ax_f.set_title('F', loc='left', fontweight='bold', fontsize=16)
    ax_f.axis('off')

    plt.tight_layout()
    utils.save_figure(fig, figsize, save_dir, 'composites', 'S15_coding_distributions', formats=['.pdf'])
    plt.close()
    print("Figure S15 composite created: S15_coding_distributions.pdf")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    print("Creating composite figures for Figure 4 supplemental (S10-S15)...")
    print()

    plot_figure_S10_composite()
    plot_figure_S11_composite()
    plot_figure_S12_composite()
    plot_figure_S13_composite()
    plot_figure_S14_composite()
    plot_figure_S15_composite()

    print()
    print(f"All composite figures created successfully in: {save_dir}")
    print("Composite PDFs are located in the 'composites' subfolder.")
