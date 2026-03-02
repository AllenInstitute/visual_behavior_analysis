"""
Supplemental Figures S6-S9 Composite - Extended population response analysis (associated with Paper Figure 3)

This script generates composite figures combining multiple panels for supplemental figures:
- S6: Cell-matched population responses (4 rows of panels)
- S7: Extended response characterization (4 rows of panels)
- S8: Omission response characterization (4 rows of panels)
- S9: Statistics and depth analysis (3 rows of panels)

These are composite/placeholder versions meant for manual assembly of individual plots.
"""

# ============================================================================
# IMPORTS
# ============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import seaborn as sns
sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})

# Statistical analysis
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from scipy.stats import chi2

# Data access
from visual_behavior.data_access import loading as loading
import visual_behavior.data_access.utilities as utilities
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache

# Utilities
import visual_behavior.visualization.utils as utils
import visual_behavior.visualization.ophys.platform_paper_figures as ppf

# ============================================================================
# CONFIGURATION
# ============================================================================
save_dir = os.path.join(os.getcwd(), 'platform_paper_figures', 'supplemental_composites')
folder = 'population_activity'

# Create save directories if they don't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if not os.path.exists(os.path.join(save_dir, folder)):
    os.makedirs(os.path.join(save_dir, folder))

# ============================================================================
# DATA LOADING
# ============================================================================
# Initialize cache
platform_cache_dir = loading.get_platform_analysis_cache_dir()
cache = VisualBehaviorOphysProjectCache.from_local_cache(cache_dir=platform_cache_dir, use_static_cache=True)

# Load experiments and cells tables
experiments_table = pd.read_csv(os.path.join(loading.get_metadata_tables_dir(), 'all_ophys_experiments_table.csv'), index_col=0)
platform_experiments = pd.read_csv(os.path.join(loading.get_metadata_tables_dir(), 'platform_paper_ophys_experiments_table.csv'), index_col=0)
platform_cells_table = pd.read_csv(os.path.join(loading.get_metadata_tables_dir(), 'platform_paper_ophys_cells_table.csv'), index_col=0)
matched_cells_table = pd.read_csv(os.path.join(loading.get_metadata_tables_dir(), 'platform_paper_matched_ophys_cells_table.csv'), index_col=0)

# Get lists of matched cells and experiments
matched_cells = matched_cells_table.cell_specimen_id.unique()
matched_experiments = matched_cells_table.ophys_experiment_id.unique()

# Get cre_lines and cell types for plot labels
cre_lines = np.sort(platform_cells_table.cre_line.unique())
cell_types = utils.get_cell_types()

# Get useful utilities
palette = utilities.get_experience_level_colors()
experience_levels = utils.get_new_experience_levels()
experience_level_colors = utils.get_experience_level_colors()

# Load response metrics table
import visual_behavior.dimensionality_reduction.clustering.processing as processing

data_type = 'filtered_events'
session_subset = 'full_session'
inclusion_criteria = 'platform_experiment_table'

filtered_events_metrics_table = processing.generate_merged_table_of_model_free_metrics(
    data_type=data_type, session_subset=session_subset,
    inclusion_criteria=inclusion_criteria, save_dir=loading.get_metadata_tables_dir()
)

metrics_table = filtered_events_metrics_table.copy()

# Create event-type specific dataframes
image_mdf = metrics_table[(metrics_table.event_type == 'images')]
change_mdf = metrics_table[(metrics_table.event_type == 'changes')]
omission_mdf = metrics_table[(metrics_table.event_type == 'omissions')]

# Create each_image specific dataframe
each_image_mdf = metrics_table[(metrics_table.event_type == 'images') &
                               (metrics_table.image_id.notna())].copy()

# ============================================================================
# HIERARCHICAL STATISTICS FUNCTIONS
# ============================================================================

def likelihood_ratio_test(model_restricted, model_full, df_diff):
    """
    Performs a Likelihood Ratio Test (LRT) to compare two nested models.

    Args:
        model_restricted: The simpler model (e.g., Random Intercept).
        model_full: The more complex model (e.g., Random Intercept + Slope).
        df_diff: The difference in the number of estimated parameters (degrees of freedom).

    Returns:
        A dictionary with LRT statistics.
    """
    # Extract Log-Likelihoods
    LL_restricted = model_restricted.llf
    LL_full = model_full.llf

    # Calculate the LRT statistic (Test Statistic)
    LRT_stat = -2 * (LL_restricted - LL_full)

    # Calculate the p-value using the chi-squared distribution
    p_value = chi2.sf(LRT_stat, df_diff)

    return {
        'LRT Statistic ($\chi^2$)': LRT_stat,
        'Degrees of Freedom (df)': df_diff,
        'P-value': p_value
    }


def extract_mixedlm_summary(model_results):
    """
    Extracts key results from a fitted statsmodels MixedLM model.
    """
    # 1. Extract Fixed Effects Table (The Coef. table)
    fixed_effects_df = model_results.summary().tables[0]
    fixed_effects_dict = fixed_effects_df.to_dict(orient='dict')

    # 2. Extract Random Effects Table (The Variance/Covariance table)
    random_effects_df = model_results.summary().tables[1]
    random_effects_dict = random_effects_df.to_dict(orient='dict')

    # 3. Extract Model Fit and Group Statistics
    model_stats = {
        'Log-Likelihood': model_results.llf,
        'AIC': model_results.aic,
        'BIC': model_results.bic,
        'No. Observations': model_results.nobs,
        'No. Groups': model_results.summary().tables[0].iloc[2][1],
        'Scale_Residual_Variance': model_results.scale
    }

    # 4. Combine into a Single Dictionary
    full_results_dict = {
        'Model_Type': 'Mixed Linear Model (MixedLM)',
        'Model_Statistics': model_stats,
        'Fixed_Effects': fixed_effects_dict,
        'Random_Effects_Variance': random_effects_dict
    }

    return full_results_dict


def extract_ols_summary(model_A_fit):
    """
    Extracts key results from a fitted statsmodels OLS model.
    """
    # 1. Extract Fixed Effects (Coefficients)
    fixed_effects_df = pd.DataFrame({
        'Coef.': model_A_fit.params,
        'Std. Err.': model_A_fit.bse,
        't': model_A_fit.tvalues,
        'P>|t|': model_A_fit.pvalues,
        'Lower CI': model_A_fit.conf_int()[0],
        'Upper CI': model_A_fit.conf_int()[1]
    })
    fixed_effects_dict = fixed_effects_df.to_dict(orient='index')

    # 2. Extract Model Fit Statistics
    model_stats = {
        'R-squared': model_A_fit.rsquared,
        'Adj. R-squared': model_A_fit.rsquared_adj,
        'F-statistic': model_A_fit.fvalue,
        'Prob (F-statistic)': model_A_fit.f_pvalue,
        'Log-Likelihood': model_A_fit.llf,
        'AIC': model_A_fit.aic,
        'BIC': model_A_fit.bic,
        'No. Observations': model_A_fit.nobs
    }

    # 3. Combine into a Single Dictionary
    full_results_dict = {
        'Model_Type': 'OLS Regression (Fixed Effects Only)',
        'Model_Statistics': model_stats,
        'Fixed_Effects': fixed_effects_dict
    }

    return full_results_dict


def run_mixed_linear_model_stats_for_metric(data, metric):
    '''
    Function to select and run a mixed linear model for nested stats to account for cells coming from specific mice
    Will test to see if mouse_id explains any residual variance after fitting data to cell metric values across experience levels

    data should be a dataframe with columns for the metric value, experience level, and mouse_id
    metric is a string, name of column in data to compute stats for
    '''
    formula = metric+' ~ experience_level'

    ##### Fit 3 model types then do likelihood ratio test to figure out which is appropriate

    # --- Model A: Fixed Effects Only (OLS equivalent) ---
    model_A_fit = ols(formula=formula, data=data).fit()

    # --- Model B: Random Intercept (RI) ---
    model_B = smf.mixedlm(formula=formula, data=data,
        groups=data["mouse_id"],
        vc_formula={'mouse_id': '0 + C(mouse_id)'}
    )
    model_B_fit = model_B.fit(reml=False)

    # --- Model C: Random Intercept and Random Slope (RI+RS) ---
    model_C = smf.mixedlm(
        formula=formula,
        data=data,
        groups=data["mouse_id"],
        re_formula='~ 1 + experience_level',
        vc_formula={'mouse_id': '0 + C(mouse_id)'}
    )
    model_C_fit = model_C.fit(reml=False)

    #### likelihood ratio tests
    # Test: B vs A (does random intercept help?)
    lrt_B_vs_A = likelihood_ratio_test(model_A_fit, model_B_fit, df_diff=1)

    # Test: C vs B (does random slope help?)
    lrt_C_vs_B = likelihood_ratio_test(model_B_fit, model_C_fit, df_diff=3)

    return {
        'model_A': model_A_fit,
        'model_B': model_B_fit,
        'model_C': model_C_fit,
        'LRT_B_vs_A': lrt_B_vs_A,
        'LRT_C_vs_B': lrt_C_vs_B,
    }


# ============================================================================
# SUPPLEMENTAL FIGURE S6: Cell-matched population analysis composite
# ============================================================================

def plot_figure_S6_composite(save_dir=save_dir, folder=folder):
    """
    S6 Composite - Cell-matched population analysis

    Layout (4 rows):
    Row 1: A — Population averages images matched (1x3: Exc, Sst, Vip) — placeholder
    Row 2: B — Response heatmaps images matched (3x3 grid: cell_type × experience) — placeholder
    Row 3: C — Population averages changes matched (1x3) — placeholder
    Row 4: D — Response heatmaps changes matched (3x3 grid) — placeholder

    figsize=(22, 28)
    """
    figsize = (22, 28)
    fig = plt.figure(figsize=figsize)

    # Row 1: A - Population averages for images (1x3)
    ax = utils.placeAxesOnGrid(fig, dim=(1, 3), xspan=(0.05, 0.95), yspan=(0.85, 0.95), wspace=0.3)
    for i, a in enumerate(ax.ravel()):
        a.text(0.5, 0.5, f'A.{i+1} — Pop. avg. images matched\n({cell_types[i]})\n(place manually)',
                transform=a.transAxes, ha='center', va='center', fontsize=11, color='gray')
        a.set_title(f'A.{i+1}', loc='left', fontweight='bold', fontsize=16)
        a.axis('off')

    # Row 2: B - Response heatmaps for images (3x3 grid)
    ax = utils.placeAxesOnGrid(fig, dim=(3, 3), xspan=(0.05, 0.95), yspan=(0.65, 0.84), wspace=0.2, hspace=0.3)
    for i, a in enumerate(ax.ravel()):
        ct_idx = i // 3
        exp_idx = i % 3
        a.text(0.5, 0.5, f'B — Heatmap matched images\n({cell_types[ct_idx]} × {experience_levels[exp_idx]})\n(place manually)',
                transform=a.transAxes, ha='center', va='center', fontsize=10, color='gray')
        a.set_title('B', loc='left', fontweight='bold', fontsize=14)
        a.axis('off')

    # Row 3: C - Population averages for changes (1x3)
    ax = utils.placeAxesOnGrid(fig, dim=(1, 3), xspan=(0.05, 0.95), yspan=(0.52, 0.62), wspace=0.3)
    for i, a in enumerate(ax.ravel()):
        a.text(0.5, 0.5, f'C.{i+1} — Pop. avg. changes matched\n({cell_types[i]})\n(place manually)',
                transform=a.transAxes, ha='center', va='center', fontsize=11, color='gray')
        a.set_title(f'C.{i+1}', loc='left', fontweight='bold', fontsize=16)
        a.axis('off')

    # Row 4: D - Response heatmaps for changes (3x3 grid)
    ax = utils.placeAxesOnGrid(fig, dim=(3, 3), xspan=(0.05, 0.95), yspan=(0.02, 0.51), wspace=0.2, hspace=0.3)
    for i, a in enumerate(ax.ravel()):
        ct_idx = i // 3
        exp_idx = i % 3
        a.text(0.5, 0.5, f'D — Heatmap matched changes\n({cell_types[ct_idx]} × {experience_levels[exp_idx]})\n(place manually)',
                transform=a.transAxes, ha='center', va='center', fontsize=10, color='gray')
        a.set_title('D', loc='left', fontweight='bold', fontsize=14)
        a.axis('off')

    plt.tight_layout()
    utils.save_figure(fig, figsize, save_dir, folder, 'figure_S6_composite', formats=['.pdf'])
    print(f"S6 composite saved to {os.path.join(save_dir, folder, 'figure_S6_composite.pdf')}")
    return fig


# ============================================================================
# SUPPLEMENTAL FIGURE S7: Extended response characterization composite
# ============================================================================

def plot_figure_S7_composite(save_dir=save_dir, folder=folder):
    """
    S7 Composite - Extended response characterization

    Layout (4 rows):
    Row 1: A — Mean response distribution | B — Reliability distribution | C — Lifetime sparseness
    Row 2: D — Metric distributions all conditions | E — Fraction responsive images
    Row 3: F — Fraction responsive changes | G — Experience modulation violins
    Row 4: H — Experience modulation annotated by cell type (changes and images)

    figsize=(24, 28)
    """
    figsize = (24, 28)
    fig = plt.figure(figsize=figsize)

    # Row 1: A, B, C - Metric distributions (1x3)
    ax = utils.placeAxesOnGrid(fig, dim=(1, 3), xspan=(0.05, 0.95), yspan=(0.85, 0.95), wspace=0.25)

    # A - Mean response distribution
    ax[0, 0].text(0.5, 0.5, 'A — Mean response distribution\nby experience level\n(place manually)',
                  transform=ax[0, 0].transAxes, ha='center', va='center', fontsize=11, color='gray')
    ax[0, 0].set_title('A', loc='left', fontweight='bold', fontsize=16)
    ax[0, 0].axis('off')

    # B - Reliability distribution
    ax[0, 1].text(0.5, 0.5, 'B — Reliability distribution\nby experience level\n(place manually)',
                  transform=ax[0, 1].transAxes, ha='center', va='center', fontsize=11, color='gray')
    ax[0, 1].set_title('B', loc='left', fontweight='bold', fontsize=16)
    ax[0, 1].axis('off')

    # C - Lifetime sparseness
    ax[0, 2].text(0.5, 0.5, 'C — Lifetime sparseness\nby experience level\n(place manually)',
                  transform=ax[0, 2].transAxes, ha='center', va='center', fontsize=11, color='gray')
    ax[0, 2].set_title('C', loc='left', fontweight='bold', fontsize=16)
    ax[0, 2].axis('off')

    # Row 2: D, E - Additional metric distributions
    ax = utils.placeAxesOnGrid(fig, dim=(1, 2), xspan=(0.05, 0.95), yspan=(0.70, 0.84), wspace=0.25)

    # D - Metric distributions all conditions
    ax[0, 0].text(0.5, 0.5, 'D — Metric distributions\nall conditions\n(place manually)',
                  transform=ax[0, 0].transAxes, ha='center', va='center', fontsize=11, color='gray')
    ax[0, 0].set_title('D', loc='left', fontweight='bold', fontsize=16)
    ax[0, 0].axis('off')

    # E - Fraction responsive images
    ax[0, 1].text(0.5, 0.5, 'E — Fraction responsive images\nby experience and cell type\n(place manually)',
                  transform=ax[0, 1].transAxes, ha='center', va='center', fontsize=11, color='gray')
    ax[0, 1].set_title('E', loc='left', fontweight='bold', fontsize=16)
    ax[0, 1].axis('off')

    # Row 3: F, G - Responsive cells and modulation
    ax = utils.placeAxesOnGrid(fig, dim=(1, 2), xspan=(0.05, 0.95), yspan=(0.55, 0.69), wspace=0.25)

    # F - Fraction responsive changes
    ax[0, 0].text(0.5, 0.5, 'F — Fraction responsive changes\nby experience and cell type\n(place manually)',
                  transform=ax[0, 0].transAxes, ha='center', va='center', fontsize=11, color='gray')
    ax[0, 0].set_title('F', loc='left', fontweight='bold', fontsize=16)
    ax[0, 0].axis('off')

    # G - Experience modulation violins
    ax[0, 1].text(0.5, 0.5, 'G — Experience modulation\nviolin plots\n(place manually)',
                  transform=ax[0, 1].transAxes, ha='center', va='center', fontsize=11, color='gray')
    ax[0, 1].set_title('G', loc='left', fontweight='bold', fontsize=16)
    ax[0, 1].axis('off')

    # Row 4: H - Experience modulation by cell type
    ax = utils.placeAxesOnGrid(fig, dim=(2, 3), xspan=(0.05, 0.95), yspan=(0.02, 0.54), wspace=0.25, hspace=0.35)

    label_idx = 0
    for event_type_idx, event_type in enumerate(['Changes', 'Images']):
        for ct_idx, cell_type in enumerate(cell_types):
            a = ax[event_type_idx, ct_idx]
            a.text(0.5, 0.5, f'H — Experience modulation\n({event_type}, {cell_type})\n(place manually)',
                    transform=a.transAxes, ha='center', va='center', fontsize=10, color='gray')
            a.set_title(f'H.{label_idx+1}', loc='left', fontweight='bold', fontsize=14)
            a.axis('off')
            label_idx += 1

    plt.tight_layout()
    utils.save_figure(fig, figsize, save_dir, folder, 'figure_S7_composite', formats=['.pdf'])
    print(f"S7 composite saved to {os.path.join(save_dir, folder, 'figure_S7_composite.pdf')}")
    return fig


# ============================================================================
# SUPPLEMENTAL FIGURE S8: Omission response characterization composite
# ============================================================================

def plot_figure_S8_composite(save_dir=save_dir, folder=folder):
    """
    S8 Composite - Omission response characterization

    Layout (4 rows):
    Row 1: A — Omission response heatmaps (3x3 grid) — placeholder
    Row 2: B — Matched omission heatmaps (3x3 grid) — placeholder
    Row 3: C — Population averages omissions (1x3) — placeholder
    Row 4: D — Omission modulation index | E — % Omission responsive cells

    figsize=(22, 30)
    """
    figsize = (22, 30)
    fig = plt.figure(figsize=figsize)

    # Row 1: A - Omission response heatmaps (3x3 grid)
    ax = utils.placeAxesOnGrid(fig, dim=(3, 3), xspan=(0.05, 0.95), yspan=(0.75, 0.95), wspace=0.2, hspace=0.3)
    for i, a in enumerate(ax.ravel()):
        ct_idx = i // 3
        exp_idx = i % 3
        a.text(0.5, 0.5, f'A — Omission response heatmaps\n({cell_types[ct_idx]} × {experience_levels[exp_idx]})\n(place manually)',
                transform=a.transAxes, ha='center', va='center', fontsize=10, color='gray')
        a.set_title('A', loc='left', fontweight='bold', fontsize=14)
        a.axis('off')

    # Row 2: B - Matched omission heatmaps (3x3 grid)
    ax = utils.placeAxesOnGrid(fig, dim=(3, 3), xspan=(0.05, 0.95), yspan=(0.55, 0.74), wspace=0.2, hspace=0.3)
    for i, a in enumerate(ax.ravel()):
        ct_idx = i // 3
        exp_idx = i % 3
        a.text(0.5, 0.5, f'B — Matched omission heatmaps\n({cell_types[ct_idx]} × {experience_levels[exp_idx]})\n(place manually)',
                transform=a.transAxes, ha='center', va='center', fontsize=10, color='gray')
        a.set_title('B', loc='left', fontweight='bold', fontsize=14)
        a.axis('off')

    # Row 3: C - Population averages omissions (1x3)
    ax = utils.placeAxesOnGrid(fig, dim=(1, 3), xspan=(0.05, 0.95), yspan=(0.40, 0.54), wspace=0.3)
    for i, a in enumerate(ax.ravel()):
        a.text(0.5, 0.5, f'C.{i+1} — Pop. avg. omissions\n({cell_types[i]})\n(place manually)',
                transform=a.transAxes, ha='center', va='center', fontsize=11, color='gray')
        a.set_title(f'C.{i+1}', loc='left', fontweight='bold', fontsize=16)
        a.axis('off')

    # Row 4: D, E - Omission modulation and responsive cells
    ax = utils.placeAxesOnGrid(fig, dim=(1, 2), xspan=(0.05, 0.95), yspan=(0.02, 0.39), wspace=0.25)

    # D - Omission modulation index
    ax[0, 0].text(0.5, 0.5, 'D — Omission modulation index\ndistribution by cell type\n(place manually)',
                  transform=ax[0, 0].transAxes, ha='center', va='center', fontsize=11, color='gray')
    ax[0, 0].set_title('D', loc='left', fontweight='bold', fontsize=16)
    ax[0, 0].axis('off')

    # E - % Omission responsive cells
    ax[0, 1].text(0.5, 0.5, 'E — % Omission responsive cells\nby experience and cell type\n(place manually)',
                  transform=ax[0, 1].transAxes, ha='center', va='center', fontsize=11, color='gray')
    ax[0, 1].set_title('E', loc='left', fontweight='bold', fontsize=16)
    ax[0, 1].axis('off')

    plt.tight_layout()
    utils.save_figure(fig, figsize, save_dir, folder, 'figure_S8_composite', formats=['.pdf'])
    print(f"S8 composite saved to {os.path.join(save_dir, folder, 'figure_S8_composite.pdf')}")
    return fig


# ============================================================================
# SUPPLEMENTAL FIGURE S9: Statistics and depth analysis composite
# ============================================================================

def plot_figure_S9_composite(save_dir=save_dir, folder=folder):
    """
    S9 Composite - Statistics and depth analysis

    Layout (3 rows):
    Row 1: A — Change response metric distribution | B — Image response metric distribution
    Row 2: C — % Responsive by depth with stats (1x3)
    Row 3: D — Change response by depth per cohort (grid of small plots) — placeholder

    figsize=(22, 24)
    """
    figsize = (22, 24)
    fig = plt.figure(figsize=figsize)

    # Row 1: A, B - Response metric distributions
    ax = utils.placeAxesOnGrid(fig, dim=(1, 2), xspan=(0.05, 0.95), yspan=(0.80, 0.95), wspace=0.25)

    # A - Change response metric distribution
    ax[0, 0].text(0.5, 0.5, 'A — Change response metric\ndistribution by experience\n(place manually)',
                  transform=ax[0, 0].transAxes, ha='center', va='center', fontsize=11, color='gray')
    ax[0, 0].set_title('A', loc='left', fontweight='bold', fontsize=16)
    ax[0, 0].axis('off')

    # B - Image response metric distribution
    ax[0, 1].text(0.5, 0.5, 'B — Image response metric\ndistribution by experience\n(place manually)',
                  transform=ax[0, 1].transAxes, ha='center', va='center', fontsize=11, color='gray')
    ax[0, 1].set_title('B', loc='left', fontweight='bold', fontsize=16)
    ax[0, 1].axis('off')

    # Row 2: C - % Responsive by depth with stats (1x3)
    ax = utils.placeAxesOnGrid(fig, dim=(1, 3), xspan=(0.05, 0.95), yspan=(0.60, 0.79), wspace=0.3)
    for i, a in enumerate(ax.ravel()):
        a.text(0.5, 0.5, f'C.{i+1} — % Responsive by depth\n({cell_types[i]}) with stats\n(place manually)',
                transform=a.transAxes, ha='center', va='center', fontsize=11, color='gray')
        a.set_title(f'C.{i+1}', loc='left', fontweight='bold', fontsize=16)
        a.axis('off')

    # Row 3: D - Change response by depth per cohort (grid)
    ax = utils.placeAxesOnGrid(fig, dim=(3, 3), xspan=(0.05, 0.95), yspan=(0.02, 0.59), wspace=0.3, hspace=0.4)
    for i, a in enumerate(ax.ravel()):
        ct_idx = i // 3
        cohort_idx = i % 3
        a.text(0.5, 0.5, f'D — Change response by depth\n({cell_types[ct_idx]}, Cohort {cohort_idx+1})\n(place manually)',
                transform=a.transAxes, ha='center', va='center', fontsize=10, color='gray')
        a.set_title(f'D.{i+1}', loc='left', fontweight='bold', fontsize=12)
        a.axis('off')

    plt.tight_layout()
    utils.save_figure(fig, figsize, save_dir, folder, 'figure_S9_composite', formats=['.pdf'])
    print(f"S9 composite saved to {os.path.join(save_dir, folder, 'figure_S9_composite.pdf')}")
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("Generating Supplemental Figures S6-S9 Composites...")

    print("\nGenerating S6 - Cell-matched population analysis composite...")
    fig_s6 = plot_figure_S6_composite()
    plt.close(fig_s6)

    print("\nGenerating S7 - Extended response characterization composite...")
    fig_s7 = plot_figure_S7_composite()
    plt.close(fig_s7)

    print("\nGenerating S8 - Omission response characterization composite...")
    fig_s8 = plot_figure_S8_composite()
    plt.close(fig_s8)

    print("\nGenerating S9 - Statistics and depth analysis composite...")
    fig_s9 = plot_figure_S9_composite()
    plt.close(fig_s9)

    print(f"\nAll supplemental figure composites saved to: {os.path.join(save_dir, folder)}")
