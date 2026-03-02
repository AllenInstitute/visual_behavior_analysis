# Code Extraction Summary - Figure 4 Scripts

## Overview

This document summarizes the complete extraction of code from 7 Figure 4-related scripts in the visual behavior analysis platform paper project.

**Files Analyzed:**
1. `251206_figure_4_updated_figures.py` (4094 lines) - MAIN LATEST
2. `241218_figure_4_updated_figures.py` (older version) - COMPARISON
3. `241218_figure_4_updated_figures_clean_run.py` (older version) - COMPARISON
4. `251206_figure_4_prediction_strength_control.py` (4094 lines) - VALIDATION CONTROL
5. `260228_clustering_shuffle_control.py` (744 lines) - SHUFFLE CONTROL
6. `240605_familiar_only_clustering_control.py` (349 lines) - EXPERIENCE LEVEL CONTROL
7. `240404_figure_4_generate_analysis_files.py` (379 lines) - DATA GENERATION

**Total Lines of Code:** ~14,600 lines

---

## Key Findings

### NEW FUNCTIONS IN LATEST VERSION (251206)

Functions added since 241218 version:

1. **`plotting.get_feature_colors_with_gray()`** - Returns color palette for features
2. **`plotting.plot_proportion_cells_area_depth_pie_chart(cluster_meta, save_dir=None, folder=None)`** - Distribution chart
3. **`plotting.plot_proportion_cells_per_depth_pie_chart(cluster_meta, save_dir=None, folder=None)`** - Distribution chart
4. **`ppf.plot_ophys_history_for_mice(sessions, color_column=..., color_map=..., save_dir=..., folder=...)`** - Mouse recording history
5. **`processing.shuffle_dropout_score(data, shuffle_type=..., separate_cre_lines_for_shuffle=...)`** - Shuffle control function

### COMPLETE IMPORT STATEMENT STRUCTURE

```
Standard library:
  - os, numpy, pandas, matplotlib.pyplot, pickle, seaborn

Visual behavior packages:
  - visual_behavior.visualization.utils
  - visual_behavior.data_access.loading
  - visual_behavior.data_access.utilities
  - visual_behavior.visualization.ophys.glm_example_plots
  - visual_behavior.visualization.ophys.platform_paper_figures (ppf)
  - visual_behavior.visualization.ophys.platform_single_cell_examples (pse)

Clustering packages:
  - visual_behavior.dimensionality_reduction.clustering.plotting
  - visual_behavior.dimensionality_reduction.clustering.processing

GLM packages:
  - visual_behavior_glm.GLM_fit_dev
  - visual_behavior_glm.GLM_analysis_tools
  - visual_behavior_glm.GLM_across_session

Scientific packages:
  - scipy.stats, sklearn.metrics.pairwise, sklearn.model_selection, umap
  - statsmodels.stats.multicomp (for Tukey HSD)
```

### DATA LOADING PIPELINE

1. **Load metadata tables (CSV)**
   - `all_ophys_experiments_table.csv`
   - `platform_paper_ophys_experiments_table.csv`
   - `platform_paper_ophys_cells_table.csv`
   - `platform_paper_matched_ophys_cells_table.csv`

2. **Load GLM results (HDF5)**
   - `across_session_normalized_platform_results_pivoted.h5`
   - `all_results.h5`

3. **Load clustering data (HDF5)**
   - `clustering_feature_matrix.h5`
   - `cluster_metadata_14_clusters.h5`

4. **Load model selection results (Pickle)**
   - Gap statistic scores
   - Silhouette scores
   - Eigengap values

5. **Load response data (generated from loading module)**
   - Image responses
   - Change responses
   - Omission responses

### CORE PROCESSING WORKFLOW

```
Feature Matrix → Gap/Eigengap/Silhouette → Select K=14 → Co-clustering
    ↓
Co-clustering Matrix → Hierarchical Clustering → Cluster Assignment
    ↓
Cluster Metadata → Add Metrics → Generate Response Tables
    ↓
Plotting Functions → Create Figures
```

### CONFIGURATION VARIABLES

| Variable | Value | Purpose |
|----------|-------|---------|
| `n_clusters` | 14 | Number of clusters selected |
| `glm_version` | '24_events_all_L2_optimize_by_session' | GLM model version |
| `data_type` | 'events' | Response data type |
| `metric` | 'euclidean' | Distance metric for clustering |
| `shuffle_type` | 'all' | Type of shuffle for gap statistic |
| `k_max` | 25 | Max clusters to test in selection |
| `output_sampling_rate` | 30 | Hz for response data interpolation |
| `inclusion_criteria` | 'platform_experiment_table' | Dataset filtering criteria |
| `threshold_percentile` | 5 | For outlier removal |

### DIRECTORY STRUCTURE

```
Platform Cache Dir/
├── all_ophys_experiments_table.csv
├── platform_paper_ophys_experiments_table.csv
├── platform_paper_ophys_cells_table.csv
├── platform_paper_matched_ophys_cells_table.csv
├── glm_results/
│   ├── 24_events_all_L2_optimize_by_session_run_params.pkl
│   ├── across_session_normalized_platform_results_pivoted.h5
│   └── all_results.h5
└── clustering/
    ├── clustering_feature_matrix.h5
    ├── cluster_metadata_14_clusters.h5
    ├── coclustering_matrix_n_14_clusters.h5
    ├── gap_scores_euclidean_24_events_all_L2_optimize_by_session_nb20_unshuffled_to_all.pkl
    ├── silhouette_score_3_24_clusters_metric_euclidean_nboots_20.pkl
    ├── eigengap_24_events_all_L2_optimize_by_session_kmax25_all.pkl
    └── shuffle_control/
        ├── shuffle_control/
        ├── shuffled_feature_matrices.pkl
        ├── cluster_meta_n_10_clusters.h5
        └── coclustering_matrix_n_10_clusters.h5

Figure Save Dir/
└── figure_4/
    ├── selecting_k_clusters/
    ├── clustering_results/
    ├── clustering_controls/
    ├── cluster_results/
    ├── single_cell_roi_and_coding_scores/
    └── (various .png files)
```

---

## PLOTTING MODULE ORGANIZATION

### Cluster Selection Plots
- `plot_gap_statistic()` - Gap statistic evaluation
- `plot_eigengap_values()` - Eigengap heuristic evaluation
- `plot_silhouette_scores()` - Silhouette score evaluation

### Cluster Validation Plots
- `plot_within_cluster_correlations()` - Within-cluster correlation distributions
- `plot_coclustering_matrix_sorted_by_cluster_size()` - Co-clustering probabilities

### Dimension Reduction Plots
- `plot_umap_for_clusters()` - UMAP colored by cluster ID
- `plot_umap_for_clusters_separately()` - Separate UMAP per cluster
- `plot_umap_for_features_separately()` - UMAP colored by features/cell types

### Heatmap Plots
- `plot_coding_score_heatmap_remapped()` - Coding score heatmaps
- `plot_cluster_means_remapped()` - Mean coding scores per cluster
- `plot_cre_line_means_remapped()` - Mean coding scores per cre line
- `plot_mean_cluster_heatmaps_remapped()` - Average heatmaps by cluster
- `plot_coding_score_heatmap_matched()` - Matched coding scores

### Population Response Plots
- `plot_population_average_response_for_clusters_as_rows_split()` - Population averages split by stimulus
- `plot_population_average_response_for_clusters_as_rows_all_response_types()` - Combined response types
- `plot_cell_response_heatmaps_for_clusters()` - Individual cell response heatmaps

### Cell Composition Plots
- `plot_percent_cells_per_cluster_all_cre()` - Cell distribution across clusters
- `plot_percent_cells_per_cluster_per_cre()` - Per-cre-line distribution
- `plot_percent_cells_per_cluster_per_cre_dominant_feature()` - Distribution by dominant feature
- `plot_percent_cells_per_cluster_per_cre_dominant_feature_xaxis()` - Alternative layout
- `plot_fraction_cells_per_cluster_per_cre()` - Fractional composition

### Anatomical Distribution Plots
- `plot_cluster_depth_distribution_by_cre_line_separately()` - Depth distribution per cre line
- `plot_cluster_depth_distribution_by_cre_lines()` - Comparison across cre lines
- `plot_proportion_cells_area_depth_pie_chart()` - Area and depth distribution (NEW)
- `plot_proportion_cells_per_depth_pie_chart()` - Depth distribution (NEW)
- `plot_cluster_info()` - General cluster information

### Modulation and Metrics Plots
- `plot_experience_modulation()` - Experience-dependent plasticity
- `plot_response_metrics_boxplot_by_cre()` - Response metrics boxplots
- `plot_response_metrics_boxplot_by_cre_as_cols()` - Response metrics with cre lines as columns
- `plot_experience_modulation()` - Experience modulation statistics
- `plot_tukey_diff_in_means_for_metric()` - Tukey HSD post-hoc comparisons
- `plot_difference_of_means_and_universal_CI()` - Difference of means with confidence intervals
- `plot_diff_of_means_cre_as_col_clusters_as_rows()` - Mean differences layout
- `plot_cluster_properties_combined()` - Combined cluster property visualization
- `plot_cluster_metrics_cre_as_col_clusters_as_rows()` - Metrics comparison layout

### Helper Functions
- `get_feature_colors_with_gray()` - Feature color palette (NEW)
- `get_pref_experience_level_colors_for_clusters()` - Experience preference colors

---

## PROCESSING MODULE FUNCTION CATEGORIES

### Data Preparation
- `get_feature_matrix_for_clustering()` - Create feature matrix for clustering
- `limit_results_pivoted_to_features_for_clustering()` - Filter to relevant features
- `flip_sign_of_dropouts()` - Normalize feature signs
- `get_cell_metadata_for_feature_matrix()` - Prepare cell metadata
- `get_cells_matched_in_3_familiar_active_sessions()` - Filter to matched cells

### Clustering
- `compute_gap()` - Compute gap statistic
- `get_coClust_matrix()` - Compute co-clustering probabilities
- `run_hierarchical_clustering_and_save_cluster_meta()` - Full clustering pipeline

### Cluster Validation
- `load_eigengap()` - Load/compute eigengap values
- `add_within_cluster_corr_to_cluster_meta()` - Add within-cluster correlations
- `get_umap_results()` - Compute/load UMAP embeddings

### Metrics Computation
- `generate_coding_score_metrics_table()` - Create metrics table
- `get_coding_score_metrics_for_clusters()` - Cluster-level metrics
- `generate_merged_table_of_coding_score_and_model_free_metrics()` - Comprehensive metrics table
- `get_coding_score_metrics()` - Cell-level coding score metrics
- `get_mean_dropout_scores_per_cluster()` - Mean dropout scores
- `get_fraction_cells_per_cluster_per_group()` - Cell composition

### Statistics
- `get_cluster_proportion_stats_for_locations()` - Statistical summaries by location
- `remove_outliers()` - Remove outlier responses

### Utilities
- `clean_cluster_meta()` - Filter sparse clusters
- `save_clustering_results()` - Save pickle files
- `get_silhouette_scores()` - Compute silhouette metrics
- `get_features_for_clustering()` - Feature list
- `get_cell_type_for_cre_line()` - Cell type mapping
- `shuffle_dropout_score()` - Shuffle control function (NEW)

---

## STATISTICAL ANALYSIS PATTERNS

### Model Selection
```
Gap Statistic → Eigengap Heuristic → Silhouette Score → Consensus → K=14
```

### Cross-validation
```
Train/Test Split (50/50) → Spectral Clustering on Both Sets →
Prediction Strength Calculation → Validation Score
```

### Post-hoc Comparisons
```
Measure Response Metric → One-way ANOVA → Tukey HSD →
Plot Difference of Means with CI
```

### Shuffle Controls
```
Original Data → Shuffle Cell IDs/Features → Compute Metrics →
Compare Against Shuffled Distribution
```

---

## CONTROL SCRIPTS OVERVIEW

### `251206_figure_4_prediction_strength_control.py`
- **Purpose:** Validate cluster selection using Tibshirani & Walther method
- **Key Functions:** Cross-validation of cluster assignments
- **Output:** Prediction strength scores for K=1 to 20

### `260228_clustering_shuffle_control.py`
- **Purpose:** Cell ID shuffle control for clustering robustness
- **Key Functions:** `get_mean_dropout_scores_per_cluster()`, `plot_mean_cluster_heatmaps_remapped()`
- **Output:** Comparison of original vs shuffled clustering results

### `240605_familiar_only_clustering_control.py`
- **Purpose:** Test clustering with familiar-only sessions
- **Key Variables:** `n_clusters = 10` (vs 14 for full data)
- **Key Functions:** `get_cells_matched_in_3_familiar_active_sessions()`
- **Output:** Supplemental figure 24

### `240404_figure_4_generate_analysis_files.py`
- **Purpose:** Generate all necessary data files for figure generation
- **Output:** All .h5 and .pkl files used by main scripts

---

## DATA FLOW DIAGRAM

```
Raw Experiments (ophys recordings)
        ↓
GLM Analysis (24_events_all_L2_optimize_by_session)
        ↓
Across-Session Normalization
        ↓
Feature Matrix (3,000 cells × 4 features)
        ↓
        ├→ Gap Statistic (K=1-25)
        ├→ Eigengap (K=1-25)
        └→ Silhouette (K=3-24)
        ↓
Select K=14 (consensus from methods)
        ↓
Spectral Clustering (150 iterations)
        ↓
Co-clustering Matrix (3,000 × 3,000)
        ↓
Agglomerative Hierarchical Clustering
        ↓
Cluster Assignment (14 clusters, 2,980 cells)
        ↓
        ├→ Cluster Metadata (cell properties)
        ├→ UMAP (embedding)
        ├→ Within-cluster correlations
        └→ Metrics (coding scores, response properties)
        ↓
Plotting Module (50+ plotting functions)
        ↓
Figure 4 Panels (A-N + supplemental S17-S24)
```

---

## PARAMETER SENSITIVITY

### Critical Parameters

| Parameter | Range Tested | Selected Value | Impact |
|-----------|--------------|---|--------|
| K (n_clusters) | 1-25 | 14 | Defines granularity of cell types |
| Distance metric | euclidean, others | euclidean | Spectral clustering metric |
| Shuffle type | 'all', 'cell_id' | 'all' | Gap statistic reference |
| Outlier percentile | 0-10 | 5 | Response filtering |
| n_iterations (co-clustering) | 50-500 | 150 | Stability of assignments |

### Robustness Tests
1. **Prediction Strength** - Cross-validation on train/test splits
2. **Shuffle Controls** - Cell ID and feature shuffles
3. **Familiar-only** - Experience level controls
4. **Alternative K values** - Eigengap and silhouette validation

---

## OUTPUT SPECIFICATIONS

### File Types Generated

**HDF5 Tables:**
- Cluster metadata with cell properties
- Coding score metrics
- Co-clustering probability matrices
- UMAP embeddings

**Pickle Files:**
- Gap statistic scores
- Silhouette scores
- Eigengap values
- Shuffled feature matrices

**Figures (PNG/PDF):**
- K selection plots (3 methods)
- Cluster validation plots
- UMAP plots (2 layouts × 2 colorings)
- Heatmaps (5+ variations)
- Population response traces (4 response types)
- Box plots (8+ metrics)
- Cell composition pie charts
- Depth distribution plots

---

## Code Quality Notes

### Strong Points
- Comprehensive data pipeline documentation
- Multiple validation/control scripts
- Clear variable naming conventions
- Extensive comments and markdown cells
- Modular function design (plotting, processing modules)

### Areas for Refactoring
- Some code repetition across plotting calls
- Long scripts with mixed concerns (data loading, analysis, plotting)
- Magic numbers (k_max=25, n_boots=20) could be parameterized
- Some inline plotting code mixed with function calls

---

## Recommendations for New Figure Scripts

When creating new Figure scripts based on this code:

1. **Follow the import structure** - Use consistent module imports
2. **Centralize configuration** - Define all directories and parameters at top
3. **Separate concerns** - Keep data loading, processing, and plotting separate
4. **Use consistent naming** - Match the data frame/variable naming conventions
5. **Add comprehensive docstrings** - Document function purposes and parameters
6. **Include validation steps** - Check data shapes and value ranges
7. **Cache intermediate results** - Use pickle/HDF5 for expensive computations
8. **Implement error handling** - Graceful failures for missing files
