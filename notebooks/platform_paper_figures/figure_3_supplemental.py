"""
Supplemental Figures S6-S9 - Extended population response analysis (associated with Paper Figure 3)

This script generates supplemental figures showing:
- S6: Cell-matched population responses and decoding analysis
- S7: Extended response characterization (metrics distributions, responsive cell fractions, experience modulation)
- S8: Omission response characterization
- S9: Additional analysis with hierarchical statistics and annotations
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
save_dir = os.path.join(os.getcwd(), 'platform_paper_figures', 'figure_3_supplemental')
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
# SUPPLEMENTAL FIGURE S6: Cell-matched population analysis
# ============================================================================

def s6_cell_matched_population():
    """
    S6 - Cell-matched population analysis:
    - Population averages for cell-matched cells across conditions
    - Response heatmaps for matched cells
    """

    # Cell-matched population averages for images
    matched = utilities.limit_to_cell_specimen_ids_matched_in_all_experience_levels(image_mdf)
    xlim_seconds = [-0.25, 0.7]

    suffix = '_images_matched'
    axes_column = 'cell_type'
    hue_column = 'experience_level'

    ax = ppf.plot_population_averages_for_conditions(
        matched, data_type, 'all', axes_column, hue_column,
        horizontal=False, xlabel='Time from image onset (s)',
        ylabel='Average calcium event magnitude', legend=False,
        xlim_seconds=xlim_seconds, interval_sec=0.5,
        palette=palette, ax=None, title=None, suptitle='Matched cells - Images',
        save_dir=save_dir, folder=folder, suffix=suffix
    )

    # Cell-matched response heatmaps for images
    timestamps = image_mdf.trace_timestamps.values[0]
    row_condition = 'cell_type'
    col_condition = 'experience_level'
    cols_to_sort_by = ['mean_response']

    ppf.plot_response_heatmaps_for_conditions(
        matched, timestamps, data_type, 'all', row_condition, col_condition,
        cols_to_sort_by=None, suptitle='Matched cells - Images',
        xlabel='Time from image onset (s)',
        microscope=None, xlim_seconds=[-1, 1.5], match_cells=True, cbar=False,
        save_dir=save_dir, folder=folder, suffix='_matched', ax=None
    )

    # Cell-matched population averages for changes
    matched_changes = utilities.limit_to_cell_specimen_ids_matched_in_all_experience_levels(change_mdf)
    xlim_seconds = [-1., 1.5]

    suffix = '_changes_matched'

    ax = ppf.plot_population_averages_for_conditions(
        matched_changes, data_type, 'changes', axes_column, hue_column,
        ylabel='Calcium events', sharey=False,
        horizontal=True, xlabel='Time from image change (s)',
        xlim_seconds=xlim_seconds, interval_sec=0.5, legend=False,
        palette=palette, ax=None, title=None, suptitle='Matched cells - Changes',
        save_dir=save_dir, folder=folder, suffix=suffix
    )

    # Cell-matched response heatmaps for changes
    ppf.plot_response_heatmaps_for_conditions(
        matched_changes, timestamps, data_type, 'changes', row_condition, col_condition,
        cols_to_sort_by=None, suptitle='Matched cells - Changes',
        xlabel='Time from image change (s)',
        microscope=None, xlim_seconds=[-1, 1.5], match_cells=True, cbar=False,
        save_dir=save_dir, folder=folder, suffix='_matched_changes', ax=None
    )


# ============================================================================
# SUPPLEMENTAL FIGURE S7: Extended response characterization
# ============================================================================

def s7_extended_response_characterization():
    """
    S7 - Extended response characterization:
    - Metric distributions by experience level (multiple metrics)
    - Metric distributions across all conditions
    - Responsive cell fractions by depth and area
    - Experience modulation indices by cell type
    - Metrics across cohorts
    """

    # Metric distributions for various metrics
    metrics_to_plot = [
        ('mean_response', 'Mean response'),
        ('reliability', 'Reliability'),
        ('lifetime_sparseness', 'Lifetime sparseness'),
    ]

    for metric, ylabel in metrics_to_plot:
        if metric in metrics_table.columns:
            ppf.plot_metric_distribution_by_experience(
                metrics_table, metric, plot_type='pointplot',
                add_zero_line=False, event_type='images', data_type=data_type,
                ylabel=ylabel, ylims=None, save_dir=save_dir, ax=None
            )

    # Metric distributions across all conditions
    ppf.plot_metric_distribution_all_conditions(
        image_mdf, 'mean_response', 'images', data_type,
        ylabel='Mean image response', ylims=None,
        remove_outliers=True, add_zero_line=False, save_dir=save_dir
    )

    # Fraction of responsive cells (original fraction version, not percent)
    ppf.plot_fraction_responsive_cells(
        image_mdf, responsiveness_threshold=0.1, ylim=(-0.02, 1),
        ylabel='Fraction of image responsive cells',
        save_dir=save_dir, folder=folder, suffix='_images_responsive',
        horizontal=False, ax=None
    )

    # Fraction of responsive cells for changes
    ppf.plot_fraction_responsive_cells(
        change_mdf, responsiveness_threshold=0.1, ylim=(-0.02, 1),
        ylabel='Fraction of image change responsive cells',
        save_dir=save_dir, folder=folder, suffix='_changes_responsive',
        horizontal=False, ax=None
    )

    # Experience modulation index (non-annotated version)
    metric = 'mean_response'
    event_type = 'changes'
    xlabel = 'Experience modulation'

    # Plot non-annotated experience modulation index
    ppf.plot_experience_modulation_index(
        change_mdf, event_type, plot_type='violinplot', ylims=(-1.1, 1.1),
        include_all_comparisons=False, save_dir=save_dir, folder=folder,
        suffix='_changes_modulation'
    )

    # Plot annotated version for changes
    ppf.plot_experience_modulation_index_annotated_by_cell_type(
        change_mdf, event_type, metric, platform_cells_table,
        xlabel=xlabel, xlims=(-1.1, 1.1), all_comparisons=False, horiz=True,
        suptitle='Experience modulation - Changes', suffix='_changes',
        save_dir=save_dir, ax=None
    )

    # Experience modulation index for images
    event_type = 'images'

    # Plot non-annotated version for images
    ppf.plot_experience_modulation_index(
        image_mdf, event_type, plot_type='violinplot', ylims=(-1.1, 1.1),
        include_all_comparisons=False, save_dir=save_dir, folder=folder,
        suffix='_images_modulation'
    )

    ppf.plot_experience_modulation_index_annotated_by_cell_type(
        image_mdf, event_type, metric, platform_cells_table,
        xlabel=xlabel, xlims=(-1.1, 1.1), all_comparisons=False, horiz=True,
        suptitle='Experience modulation - Images', suffix='_images',
        save_dir=save_dir, ax=None
    )

    # Modulation index distributions
    metric = 'change_modulation_index'
    xlabel = 'Change modulation index'
    xlims = (-0.2, 0.5)

    ppf.plot_modulation_index_distribution(
        metrics_table, metric, label=xlabel, lims=xlims, horiz=True,
        annot=('left', 'right'), save_dir=save_dir, folder=folder, ax=None
    )

    # Metrics across cohorts by area and depth
    for metric, ylabel in [('mean_response_changes', 'Mean change response'),
                            ('change_modulation_index', 'Change modulation index')]:
        if metric in metrics_table.columns:
            ppf.plot_metric_across_cohorts_area_depth(
                metrics_table, metric, ylabel=ylabel, plot_type='barplot',
                save_dir=save_dir, folder=folder, ax=None
            )

            ppf.plot_metric_across_cohorts_area_depth(
                metrics_table, metric, ylabel=ylabel, plot_type='violinplot',
                save_dir=save_dir, folder=folder, ax=None
            )


# ============================================================================
# SUPPLEMENTAL FIGURE S8: Omission response characterization
# ============================================================================

def s8_omission_response_characterization():
    """
    S8 - Omission response characterization:
    - Population averages for omissions across conditions
    - Omission modulation index
    - Experience modulation for omissions
    - Responsive cell fractions for omissions
    """

    # Population averages for omissions - heatmaps
    timestamps = omission_mdf.trace_timestamps.values[0]
    row_condition = 'cell_type'
    col_condition = 'experience_level'

    ppf.plot_response_heatmaps_for_conditions(
        omission_mdf, timestamps, data_type, 'omissions', row_condition, col_condition,
        cols_to_sort_by=None, suptitle='Omission responses',
        xlabel='Time from omission (s)',
        microscope=None, xlim_seconds=[-1, 2], match_cells=False, cbar=False,
        save_dir=save_dir, folder=folder, suffix='_omissions', ax=None
    )

    # Cell-matched omission heatmaps
    matched_omissions = utilities.limit_to_cell_specimen_ids_matched_in_all_experience_levels(omission_mdf)

    ppf.plot_response_heatmaps_for_conditions(
        matched_omissions, timestamps, data_type, 'omissions', row_condition, col_condition,
        cols_to_sort_by=None, suptitle='Matched cells - Omissions',
        xlabel='Time from omission (s)',
        microscope=None, xlim_seconds=[-1, 2], match_cells=True, cbar=False,
        save_dir=save_dir, folder=folder, suffix='_matched_omissions', ax=None
    )

    # Population averages for omissions across conditions
    axes_column = 'cell_type'
    hue_column = 'experience_level'
    xlim_seconds = [-1., 1.5]

    ax = ppf.plot_population_averages_for_conditions(
        omission_mdf, data_type, 'omissions', axes_column, hue_column,
        horizontal=True, xlabel='Time from omission (s)',
        ylabel='Calcium events', legend=False,
        xlim_seconds=xlim_seconds, interval_sec=0.5,
        palette=palette, ax=None, title=None, suptitle=None,
        save_dir=save_dir, folder=folder, suffix='_omissions'
    )

    # Omission modulation index
    metric = 'omission_modulation_index'
    xlabel = 'Omission modulation index'
    xlims = (-0.2, 0.5)

    if metric in metrics_table.columns:
        ppf.plot_modulation_index_distribution(
            metrics_table, metric, label=xlabel, lims=xlims, horiz=True,
            annot=('left', 'right'), save_dir=save_dir, folder=folder, ax=None
        )

    # Experience modulation for omissions by cell type
    event_type = 'omissions'
    metric = 'mean_response'
    xlabel = 'Experience modulation'

    ppf.plot_experience_modulation_index_annotated_by_cell_type(
        omission_mdf, event_type, metric, platform_cells_table,
        xlabel=xlabel, xlims=(-1.1, 1.1), all_comparisons=False, horiz=True,
        suptitle='Experience modulation - Omissions', suffix='_omissions',
        save_dir=save_dir, ax=None
    )

    # Responsive cell fractions for omissions
    ppf.plot_percent_responsive_cells(
        omission_mdf, responsiveness_threshold=0.1, ylim=(-5, 105),
        ylabel='% Omission responsive cells',
        save_dir=save_dir, folder=folder, suffix='_omissions_responsive',
        horizontal=False, ax=None
    )


# ============================================================================
# SUPPLEMENTAL FIGURE S9: Additional analysis with statistics
# ============================================================================

def s9_additional_analysis_with_stats():
    """
    S9 - Additional analysis:
    - Hierarchical mixed linear model statistics for key metrics
    - Statistical annotations on response plots
    - Detailed comparisons across depth and area
    """

    # Run hierarchical stats for key metrics
    key_metrics = ['mean_response_changes', 'mean_response_images', 'reliability']

    for metric in key_metrics:
        if metric in metrics_table.columns:
            # Filter out NaN values
            data_clean = metrics_table[['mouse_id', 'experience_level', metric]].dropna()

            if len(data_clean) > 0 and len(data_clean.mouse_id.unique()) > 1:
                try:
                    stats_results = run_mixed_linear_model_stats_for_metric(data_clean, metric)
                    print(f"\n{metric} - Statistics Summary:")
                    print(f"LRT (B vs A - Random Intercept): {stats_results['LRT_B_vs_A']}")
                    print(f"LRT (C vs B - Random Slope): {stats_results['LRT_C_vs_B']}")
                except Exception as e:
                    print(f"Could not fit mixed model for {metric}: {e}")

    # Get descriptive stats for metrics
    cols_to_groupby = ['experience_level', 'cell_type']
    stats_summary = ppf.get_descriptive_stats_for_metric(metrics_table, 'mean_response_changes', cols_to_groupby)
    print("\nDescriptive stats for mean_response_changes:")
    print(stats_summary)

    # Plot metric distributions with statistical annotations for changes
    metric = 'mean_response_changes'
    ylabel = 'Mean image change response'
    event_type = 'changes'

    ppf.plot_metric_distribution_by_experience(
        metrics_table, metric, plot_type='pointplot', horiz=False,
        suptitle='Change response - All Cells',
        add_zero_line=False, event_type=event_type, data_type=data_type,
        ylabel=ylabel, ylims=None, save_dir=save_dir, ax=None
    )

    # Plot metric distributions for images
    metric = 'mean_response_images'
    ylabel = 'Mean image response'
    event_type = 'images'

    if metric in metrics_table.columns:
        ppf.plot_metric_distribution_by_experience(
            metrics_table, metric, plot_type='pointplot', horiz=False,
            suptitle='Image response - All Cells',
            add_zero_line=False, event_type=event_type, data_type=data_type,
            ylabel=ylabel, ylims=None, save_dir=save_dir, ax=None
        )

    # Add statistical annotations to plots
    ppf.add_stats_to_plot()
    ppf.add_stats_to_plot_xaxis()

    # Plot fraction responsive cells with depth/area breakdown
    fraction_matched = ppf.get_fraction_matched_cells(
        matched_cells_table, platform_cells_table,
        conditions=['cell_type', 'ophys_container_id']
    )
    fraction_matched = fraction_matched.reset_index()
    fraction_matched = fraction_matched.merge(
        platform_experiments[['ophys_container_id', 'binned_depth', 'imaging_depth']],
        on='ophys_container_id'
    )
    print("\nFraction of matched cells by cell type and depth:")
    print(fraction_matched)

    # Also get fraction of responsive cells
    fraction_responsive = ppf.get_fraction_responsive_cells(
        image_mdf, conditions=['cell_type', 'experience_level', 'ophys_container_id', 'ophys_experiment_id'],
        responsiveness_threshold=0.1
    )
    fraction_responsive = fraction_responsive.reset_index()
    fraction_responsive = fraction_responsive.merge(
        platform_experiments[['ophys_container_id', 'binned_depth', 'imaging_depth']],
        on='ophys_container_id'
    )
    fraction_responsive = fraction_responsive.drop_duplicates(subset=['ophys_container_id', 'ophys_experiment_id'])
    fraction_responsive['percent_responsive'] = fraction_responsive['fraction_responsive'] * 100

    # Plot by depth and cell type with statistical annotations
    depths = np.sort(fraction_responsive.binned_depth.unique())
    figsize = (10, 2.25)
    fig, ax = plt.subplots(1, 3, figsize=figsize, sharey=True)
    ax = ax.ravel()

    for i, cell_type in enumerate(cell_types):
        data = fraction_responsive[fraction_responsive.cell_type == cell_type]
        ax[i] = sns.pointplot(
            data=data, x='binned_depth', y='percent_responsive',
            hue='experience_level', linestyle='none', order=depths,
            markers='.', markersize=4, err_kws={'linewidth': 2},
            palette=experience_level_colors, dodge=0.2, ax=ax[i]
        )
        ax[i].set_title(cell_type)
        ax[i].get_legend().remove()
        ax[i].set_xlabel('Imaging depth (um)')
        ax[i].set_ylabel('% Image responsive cells')
        ax[i].set_ylim(0, 100)

        # Add statistical annotations to the plot
        ax[i], tukey_table = ppf.add_stats_to_plot_for_hues(
            data, 'percent_responsive', ax[i],
            xorder=depths, x='binned_depth', hue='experience_level'
        )

        # Add statistical annotations
        ax[i], tukey_table = ppf.add_stats_to_plot_for_hues(
            data, 'percent_responsive', ax[i],
            xorder=depths, x='binned_depth', hue='experience_level'
        )

    plt.subplots_adjust(wspace=0.1)
    filename = 'percent_responsive_cells_by_depth_stats'
    utils.save_figure(fig, figsize, save_dir, folder, filename)

    # Plot change response by depth across cohorts
    metric = 'mean_response_changes'
    ylabel = 'Mean image change response'

    project_codes = metrics_table.project_code.unique()
    for c, cell_type in enumerate(cell_types):
        ct_data = metrics_table[metrics_table.cell_type == cell_type]
        for p, project_code in enumerate(project_codes):
            data = ct_data[ct_data.project_code == project_code]
            imaging_depths = np.sort(data.binned_depth.unique())

            if len(imaging_depths) > 0:
                fig, ax = plt.subplots(1, 1, figsize=(4, 3))
                ax = sns.pointplot(
                    data=data, x='binned_depth', y=metric, hue='experience_level',
                    order=imaging_depths, hue_order=experience_levels,
                    palette=palette, dodge=0.3, linestyle='none',
                    markers='.', markersize=8, err_kws={'linewidth': 2},
                    errorbar=('ci', 95), ax=ax
                )
                ax.set_ylabel(ylabel)
                ax.set_xlabel('Imaging depth (um)')
                ax.set_title(f'{cell_type} - Cohort {p+1}')

                # Add statistical annotations
                ax, tukey_table = ppf.add_stats_to_plot_for_hues(
                    data, metric, ax,
                    xorder=imaging_depths, x='binned_depth', hue='experience_level'
                )
                plt.tight_layout()
                filename = f'change_response_by_depth_{cell_type}_Cohort{p+1}_stats'
                utils.save_figure(fig, (4, 3), save_dir, folder, filename)
                plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("Generating Supplemental Figures S6-S9...")

    print("\nGenerating S6 - Cell-matched population analysis...")
    s6_cell_matched_population()

    print("\nGenerating S7 - Extended response characterization...")
    s7_extended_response_characterization()

    print("\nGenerating S8 - Omission response characterization...")
    s8_omission_response_characterization()

    print("\nGenerating S9 - Additional analysis with statistics...")
    s9_additional_analysis_with_stats()

    print(f"\nAll supplemental figures saved to: {save_dir}")
