"""
Supplemental Figures S3-S5 Composite - Extended behavioral analysis (associated with Figure 2)

This script generates composite figures combining multiple supplemental figure panels
showing detailed behavioral training history, performance across cohorts, and behavioral
timeseries analysis. The composites are designed for publication and Illustrator editing.

Figure S3 - Training history and behavior performance:
  - Panel A: Training history for mice (colored by behavior stage)
  - Panel B: Days in stage (bar chart)
  - Panel C: Prior exposures (bar chart)
  - Panel D: Stimulus exposure line plot
  - Panel E: Example mouse behavior over time
  - Panel F: Behavior metrics across stages (mean d-prime)
  - Panel G: Hit rate / FA rate

Figure S4 - Cohort-level behavioral performance:
  - Panel A: Response rate by trial type (3 panels: platform, ophys, handoff)
  - Panel B: D-prime by cohort (1x3 grid)
  - Panel C: Response latency by cohort (1x3 grid)
  - Panel D: Response probability heatmaps

Figure S5 - Behavioral timeseries and performance details:
  - Panel A: d-prime distribution by experience
  - Panel B: max d-prime by experience
  - Panel C: Fraction engaged by experience
  - Panel D: Response latency by experience
  - Panel E: Ophys history for all mice
  - Panel F: Ophys history for platform mice
"""

# ============================================================================
# IMPORTS
# ============================================================================
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})

# Data access
from visual_behavior.data_access import loading as loading
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache

# Utilities
import visual_behavior.utilities as vbu
import visual_behavior.visualization.utils as utils
import visual_behavior.data_access.utilities as utilities
import visual_behavior.visualization.behavior

# Visualization modules
import visual_behavior.visualization.ophys.platform_paper_figures as ppf
import visual_behavior.visualization.ophys.summary_figures as sf
import visual_behavior.visualization.ophys.experiment_summary_figures as df

# ============================================================================
# CONFIGURATION
# ============================================================================
save_dir = os.path.join(os.getcwd(), 'platform_paper_figures', 'supplemental_composites')
os.makedirs(save_dir, exist_ok=True)

folder = 'behavior_metrics'

# ============================================================================
# DATA LOADING
# ============================================================================
# Initialize cache
cache_dir = loading.get_platform_analysis_cache_dir()
cache = VisualBehaviorOphysProjectCache.from_local_cache(cache_dir=cache_dir, use_static_cache=True)

# Get behavior session tables
all_behavior_sessions = cache.get_behavior_session_table()
all_behavior_sessions['mouse_id'] = [int(mouse_id) for mouse_id in all_behavior_sessions.mouse_id.values]
all_behavior_sessions['experience_level'] = [utils.convert_experience_level(experience_level)
                                             for experience_level in all_behavior_sessions.experience_level.values]

# Get ophys experiment tables
all_experiments_table = cache.get_ophys_experiment_table()
all_experiments_table['mouse_id'] = [int(mouse_id) for mouse_id in all_experiments_table.mouse_id.values]
all_experiments_table['experience_level'] = [utils.convert_experience_level(experience_level)
                                            for experience_level in all_experiments_table.experience_level.values]

# Get platform paper specific tables
platform_experiments = loading.get_platform_paper_experiment_table(limit_to_closest_active=True, include_4x2_data=False)
platform_experiments['mouse_id'] = [int(mouse_id) for mouse_id in platform_experiments.mouse_id.values]
platform_experiments['experience_level'] = [utils.convert_experience_level(experience_level)
                                           for experience_level in platform_experiments.experience_level.values]

platform_behavior_sessions = loading.get_platform_paper_behavior_session_table()
platform_behavior_sessions['mouse_id'] = [int(mouse_id) for mouse_id in platform_behavior_sessions.mouse_id.values]
platform_behavior_sessions['experience_level'] = [utils.convert_experience_level(experience_level)
                                                 for experience_level in platform_behavior_sessions.experience_level.values]

# Limit behavior sessions to platform experiment mice
behavior_sessions = platform_behavior_sessions[platform_behavior_sessions.mouse_id.isin(platform_experiments.mouse_id.unique())]

# Add useful columns for filtering
original_behavior_sessions = behavior_sessions.copy()
behavior_sessions['ophys_container_id'] = behavior_sessions.mouse_id.values
behavior_sessions = utilities.add_date_string(behavior_sessions)
behavior_sessions = utilities.add_n_relative_to_first_novel_column(behavior_sessions)
behavior_sessions = utilities.add_first_novel_column(behavior_sessions)
behavior_sessions = utilities.add_second_novel_active_column(behavior_sessions)
behavior_sessions = utilities.add_last_familiar_active_column(behavior_sessions)

# Get utility lists and colormaps
experience_levels = utils.get_experience_levels()
new_experience_levels = utils.get_new_experience_levels()
cre_lines = utils.get_cre_lines()
cell_types = utils.get_cell_types()
experience_colors = utils.get_experience_level_colors()

# Load behavior statistics
method = 'sdk'
behavior_session_ids = behavior_sessions.index.values
original_behavior_stats_sdk, problem_sessions = vbu.get_behavior_stats_for_sessions(behavior_session_ids, behavior_sessions,
                                                method=method, engaged_only=False, per_image=False)
behavior_stats_sdk = original_behavior_stats_sdk.merge(behavior_sessions, on='behavior_session_id')

# Load stimulus-based behavior metrics (engaged only)
method = 'stimulus_based'
original_engaged_behavior_stats_stim, problem_sessions = vbu.get_behavior_stats_for_sessions(behavior_session_ids, behavior_sessions,
                                                method=method, engaged_only=True, per_image=False)
engaged_behavior_stats_stim = original_engaged_behavior_stats_stim.merge(behavior_sessions, on='behavior_session_id')
engaged_behavior_stats_stim = engaged_behavior_stats_stim[engaged_behavior_stats_stim.project_code != 'VisualBehaviorMultiscope4areasx2d']

# Load platform behavior stats for stimulus-based metrics
platform_behavior_stats = pd.read_csv(os.path.join(cache_dir, 'behavior_performance', 'platform_behavior_stats_engaged.csv'), index_col=0)


# ============================================================================
# COMPOSITE FIGURE S3: Training history and behavior performance
# ============================================================================

def plot_figure_S3_composite(behavior_sessions_input, behavior_stats_sdk_input, engaged_behavior_stats_stim_input,
                             platform_experiments_input, save_dir=None, folder='behavior_metrics'):
    """
    Generate composite Figure S3 with all training history and behavior performance panels.

    Layout:
      Row 1 (~15%): A — Training history for mice (full width placeholder)
      Row 2 (~20%): B — Days in stage | C — Prior exposures
      Row 3 (~20%): D — Stimulus exposure | E — Example mouse behavior
      Row 4 (~25%): F — Mean d-prime across stages | G — Hit/FA rate
    """
    figsize = (24, 28)
    fig = plt.figure(figsize=figsize, facecolor='white')

    # ---- Row 1: Panel A (Training history for mice) ----
    ax_A = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 1.0], yspan=[0, 0.15])
    ax_A.text(0.5, 0.5, 'A — Training history for mice colored by behavior stage',
              transform=ax_A.transAxes, ha='center', va='center', fontsize=11, color='gray')
    ax_A.set_title('A', loc='left', fontweight='bold', fontsize=16)
    ax_A.axis('off')

    # ---- Row 2: Panel B and C ----
    ax_B = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 0.48], yspan=[0.16, 0.32])
    ax_B.text(0.5, 0.5, 'B — Days in stage',
              transform=ax_B.transAxes, ha='center', va='center', fontsize=11, color='gray')
    ax_B.set_title('B', loc='left', fontweight='bold', fontsize=16)
    ax_B.axis('off')

    ax_C = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.52, 1.0], yspan=[0.16, 0.32])
    ax_C.text(0.5, 0.5, 'C — Prior exposures to image sets',
              transform=ax_C.transAxes, ha='center', va='center', fontsize=11, color='gray')
    ax_C.set_title('C', loc='left', fontweight='bold', fontsize=16)
    ax_C.axis('off')

    # ---- Row 3: Panel D and E ----
    ax_D = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 0.48], yspan=[0.34, 0.50])
    ax_D.text(0.5, 0.5, 'D — Stimulus exposure prior to imaging',
              transform=ax_D.transAxes, ha='center', va='center', fontsize=11, color='gray')
    ax_D.set_title('D', loc='left', fontweight='bold', fontsize=16)
    ax_D.axis('off')

    ax_E = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.52, 1.0], yspan=[0.34, 0.50])
    ax_E.text(0.5, 0.5, 'E — Example mouse behavior over time',
              transform=ax_E.transAxes, ha='center', va='center', fontsize=11, color='gray')
    ax_E.set_title('E', loc='left', fontweight='bold', fontsize=16)
    ax_E.axis('off')

    # ---- Row 4: Panel F and G ----
    ax_F = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 0.48], yspan=[0.52, 0.75])
    ax_F.text(0.5, 0.5, 'F — Mean d-prime across behavior stages',
              transform=ax_F.transAxes, ha='center', va='center', fontsize=11, color='gray')
    ax_F.set_title('F', loc='left', fontweight='bold', fontsize=16)
    ax_F.axis('off')

    ax_G = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.52, 1.0], yspan=[0.52, 0.75])
    ax_G.text(0.5, 0.5, 'G — Hit rate and False alarm rate',
              transform=ax_G.transAxes, ha='center', va='center', fontsize=11, color='gray')
    ax_G.set_title('G', loc='left', fontweight='bold', fontsize=16)
    ax_G.axis('off')

    # Empty space for additional content if needed
    ax_extra = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 1.0], yspan=[0.75, 1.0])
    ax_extra.axis('off')

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'figure_S3_composite', formats=['.pdf'])
    return fig


# ============================================================================
# COMPOSITE FIGURE S4: Cohort behavioral performance
# ============================================================================

def plot_figure_S4_composite(behavior_sessions_input, behavior_stats_sdk_input, engaged_behavior_stats_stim_input,
                             platform_experiments_input, save_dir=None, folder='behavior_metrics'):
    """
    Generate composite Figure S4 with cohort-level behavioral performance panels.

    Layout:
      Row 1 (~20%): A — Response rate by trial type (3 panels: platform, ophys, handoff) - placeholder
      Row 2 (~25%): B — D-prime by cohort (1x3 grid)
      Row 3 (~25%): C — Response latency by cohort (1x3 grid)
      Row 4 (~30%): D — Response probability heatmaps - placeholder
    """
    figsize = (22, 22)
    fig = plt.figure(figsize=figsize, facecolor='white')

    # ---- Row 1: Panel A (Response rate by trial type) ----
    ax_A = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 1.0], yspan=[0, 0.20])
    ax_A.text(0.5, 0.5, 'A — Response rate by trial type (platform, ophys, handoff sessions)',
              transform=ax_A.transAxes, ha='center', va='center', fontsize=11, color='gray')
    ax_A.set_title('A', loc='left', fontweight='bold', fontsize=16)
    ax_A.axis('off')

    # ---- Row 2: Panel B (D-prime by cohort) ----
    ax_B = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 1.0], yspan=[0.22, 0.47])
    ax_B.text(0.5, 0.5, 'B — D-prime by cohort (Cohort 1, 2, 3)',
              transform=ax_B.transAxes, ha='center', va='center', fontsize=11, color='gray')
    ax_B.set_title('B', loc='left', fontweight='bold', fontsize=16)
    ax_B.axis('off')

    # ---- Row 3: Panel C (Response latency by cohort) ----
    ax_C = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 1.0], yspan=[0.49, 0.74])
    ax_C.text(0.5, 0.5, 'C — Response latency by cohort (Cohort 1, 2, 3)',
              transform=ax_C.transAxes, ha='center', va='center', fontsize=11, color='gray')
    ax_C.set_title('C', loc='left', fontweight='bold', fontsize=16)
    ax_C.axis('off')

    # ---- Row 4: Panel D (Response probability heatmaps) ----
    ax_D = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 1.0], yspan=[0.76, 1.0])
    ax_D.text(0.5, 0.5, 'D — Response probability heatmaps for cohorts',
              transform=ax_D.transAxes, ha='center', va='center', fontsize=11, color='gray')
    ax_D.set_title('D', loc='left', fontweight='bold', fontsize=16)
    ax_D.axis('off')

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'figure_S4_composite', formats=['.pdf'])
    return fig


# ============================================================================
# COMPOSITE FIGURE S5: Behavioral timeseries and performance details
# ============================================================================

def plot_figure_S5_composite(behavior_sessions_input, behavior_stats_sdk_input, engaged_behavior_stats_stim_input,
                             platform_experiments_input, all_behavior_sessions_input, all_experiments_table_input,
                             save_dir=None, folder='behavior_metrics'):
    """
    Generate composite Figure S5 with behavioral timeseries and performance detail panels.

    Layout:
      Row 1 (~15%): A — d-prime distribution by experience | B — max d-prime by experience
      Row 2 (~15%): C — Fraction engaged by experience | D — Response latency by experience
      Row 3 (~30%): E — Ophys history for all mice (full width placeholder)
      Row 4 (~25%): F — Ophys history for platform mice (full width placeholder)
    """
    figsize = (22, 24)
    fig = plt.figure(figsize=figsize, facecolor='white')

    # ---- Row 1: Panels A and B ----
    ax_A = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 0.48], yspan=[0, 0.15])
    ax_A.text(0.5, 0.5, 'A — d-prime distribution by experience',
              transform=ax_A.transAxes, ha='center', va='center', fontsize=11, color='gray')
    ax_A.set_title('A', loc='left', fontweight='bold', fontsize=16)
    ax_A.axis('off')

    ax_B = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.52, 1.0], yspan=[0, 0.15])
    ax_B.text(0.5, 0.5, 'B — max d-prime by experience',
              transform=ax_B.transAxes, ha='center', va='center', fontsize=11, color='gray')
    ax_B.set_title('B', loc='left', fontweight='bold', fontsize=16)
    ax_B.axis('off')

    # ---- Row 2: Panels C and D ----
    ax_C = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 0.48], yspan=[0.17, 0.32])
    ax_C.text(0.5, 0.5, 'C — Fraction engaged by experience',
              transform=ax_C.transAxes, ha='center', va='center', fontsize=11, color='gray')
    ax_C.set_title('C', loc='left', fontweight='bold', fontsize=16)
    ax_C.axis('off')

    ax_D = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.52, 1.0], yspan=[0.17, 0.32])
    ax_D.text(0.5, 0.5, 'D — Response latency by experience',
              transform=ax_D.transAxes, ha='center', va='center', fontsize=11, color='gray')
    ax_D.set_title('D', loc='left', fontweight='bold', fontsize=16)
    ax_D.axis('off')

    # ---- Row 3: Panel E (Ophys history for all mice) ----
    ax_E = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 1.0], yspan=[0.34, 0.64])
    ax_E.text(0.5, 0.5, 'E — Ophys history for all mice',
              transform=ax_E.transAxes, ha='center', va='center', fontsize=11, color='gray')
    ax_E.set_title('E', loc='left', fontweight='bold', fontsize=16)
    ax_E.axis('off')

    # ---- Row 4: Panel F (Ophys history for platform mice) ----
    ax_F = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 1.0], yspan=[0.66, 1.0])
    ax_F.text(0.5, 0.5, 'F — Ophys history for platform mice',
              transform=ax_F.transAxes, ha='center', va='center', fontsize=11, color='gray')
    ax_F.set_title('F', loc='left', fontweight='bold', fontsize=16)
    ax_F.axis('off')

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'figure_S5_composite', formats=['.pdf'])
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("Generating Supplemental Figures S3-S5 Composite...")

    # Create output directory if it doesn't exist
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    print("\nGenerating S3 Composite - Training history and behavior performance...")
    fig_S3 = plot_figure_S3_composite(
        behavior_sessions, behavior_stats_sdk, engaged_behavior_stats_stim,
        platform_experiments, save_dir=save_dir, folder=folder
    )

    print("\nGenerating S4 Composite - Cohort behavioral performance...")
    fig_S4 = plot_figure_S4_composite(
        behavior_sessions, behavior_stats_sdk, engaged_behavior_stats_stim,
        platform_experiments, save_dir=save_dir, folder=folder
    )

    print("\nGenerating S5 Composite - Behavioral timeseries and performance details...")
    fig_S5 = plot_figure_S5_composite(
        behavior_sessions, behavior_stats_sdk, engaged_behavior_stats_stim,
        platform_experiments, all_behavior_sessions, all_experiments_table,
        save_dir=save_dir, folder=folder
    )

    print("\nSupplemental Figures S3-S5 composite generation complete!")
    print(f"Figures saved to: {save_dir}")
