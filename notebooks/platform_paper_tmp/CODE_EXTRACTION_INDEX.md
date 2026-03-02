# Code Extraction Index - Figure 4 Scripts

## Document Overview

This is a complete extraction of all code from 7 Figure 4-related Python scripts in the visual behavior platform paper project. Three comprehensive documents have been created:

### 1. **COMPLETE_CODE_EXTRACTION.md** (MAIN REFERENCE)
   - **Purpose:** Exhaustive listing of all code elements
   - **Contains:**
     - All import statements (exact text)
     - All configuration variables and directory paths
     - All data loading code (exact text)
     - All plotting function calls with arguments (complete)
     - All processing function calls (complete)
     - Unique functions in latest version vs older version
     - Custom matplotlib/seaborn code

   **Use this when:** You need the exact code for any function call or data loading operation

### 2. **ADDITIONAL_CODE_DETAILS.md** (SNIPPETS & PATTERNS)
   - **Purpose:** Contextual code snippets and usage patterns
   - **Contains:**
     - Data loading workflows with full context
     - Clustering initialization and selection methods
     - Cross-validation code examples
     - Statistical analysis patterns
     - Individual cell analysis code
     - Experience modulation analysis
     - Response metrics statistical tests
     - Color utility functions
     - Figure customization examples
     - Common filtering and merging patterns

   **Use this when:** You want to understand HOW functions are used together or need example snippets

### 3. **CODE_EXTRACTION_SUMMARY.md** (OVERVIEW & ORGANIZATION)
   - **Purpose:** High-level summary and organization
   - **Contains:**
     - Key findings and new functions in latest version
     - Complete import structure overview
     - Data loading pipeline workflow
     - Core processing workflow
     - Configuration variables table
     - Directory structure diagram
     - Plotting module organization (by category)
     - Processing module organization (by category)
     - Statistical analysis patterns
     - Control scripts overview
     - Data flow diagram
     - Parameter sensitivity analysis
     - Output specifications

   **Use this when:** You want to understand the overall architecture or find which functions belong to which category

---

## Source Files Analyzed

### Main Scripts

**`251206_figure_4_updated_figures.py` (4094 lines)**
- Latest version
- Contains all main figure plotting code
- Includes new functions: pie charts, ophys history, shuffle control
- Complete clustering analysis workflow

**`251206_figure_4_prediction_strength_control.py` (4094 lines)**
- Duplicate for testing Tibshirani & Walther prediction strength method
- Same imports and data loading as 251206_figure_4_updated_figures.py

### Control Scripts

**`260228_clustering_shuffle_control.py` (744 lines)**
- Cell ID shuffle control validation
- Tests robustness of clustering to cell ID permutations

**`240605_familiar_only_clustering_control.py` (349 lines)**
- Experience level control analysis
- Tests clustering with familiar-only sessions (n_clusters=10)
- Supplemental figure 24

**`240404_figure_4_generate_analysis_files.py` (379 lines)**
- Data file generation script
- Creates all .h5 and .pkl cache files

### Older Versions (for comparison)

**`241218_figure_4_updated_figures.py`**
- Older version for comparison
- Missing: pie charts, ophys history, shuffle control functions

**`241218_figure_4_updated_figures_clean_run.py`**
- Older clean run version

---

## Quick Reference by Task

### I need the imports for my script
→ See **COMPLETE_CODE_EXTRACTION.md**, Section "ALL IMPORT STATEMENTS"

### I need to load data files
→ See **COMPLETE_CODE_EXTRACTION.md**, Section "CONFIGURATION VARIABLES AND DATA LOADING"
OR **ADDITIONAL_CODE_DETAILS.md**, Section "DETAILED DATA LOADING SECTION"

### I need to call a plotting function
→ See **CODE_EXTRACTION_SUMMARY.md**, Section "PLOTTING MODULE ORGANIZATION"
For exact arguments: See **COMPLETE_CODE_EXTRACTION.md**, Section "PLOTTING FUNCTION CALLS"

### I need to call a processing function
→ See **CODE_EXTRACTION_SUMMARY.md**, Section "PROCESSING MODULE FUNCTION CATEGORIES"
For exact arguments: See **COMPLETE_CODE_EXTRACTION.md**, Section "CLUSTERING-RELATED PROCESSING FUNCTION CALLS"

### I need example code for a specific analysis
→ See **ADDITIONAL_CODE_DETAILS.md**, Sections:
- "Clustering Initialization and Selection Code"
- "Prediction Strength Cross-Validation"
- "Experience Level Modulation Analysis"
- "Response Metrics Statistical Comparisons"

### I need to understand the overall workflow
→ See **CODE_EXTRACTION_SUMMARY.md**, Sections:
- "Data Loading Pipeline"
- "Core Processing Workflow"
- "Data Flow Diagram"

### I need to know what changed between versions
→ See **CODE_EXTRACTION_SUMMARY.md**, Section "NEW FUNCTIONS IN LATEST VERSION (251206)"
OR **COMPLETE_CODE_EXTRACTION.md**, Section "UNIQUE FUNCTIONS FOUND IN 251206_figure_4_updated_figures.py"

### I need directory paths
→ See **COMPLETE_CODE_EXTRACTION.md**, Section "KEY CONFIGURATION DIRECTORIES"
OR **CODE_EXTRACTION_SUMMARY.md**, Section "DIRECTORY STRUCTURE"

### I need to know function categories
→ See **CODE_EXTRACTION_SUMMARY.md**, Sections:
- "PLOTTING MODULE ORGANIZATION"
- "PROCESSING MODULE FUNCTION CATEGORIES"

### I need statistical analysis code
→ See **ADDITIONAL_CODE_DETAILS.md**, Sections:
- "Response Metrics Statistical Comparisons"
- "Box plots by cre line"
- "Tukey HSD post-hoc tests"

### I need color/style configuration
→ See **ADDITIONAL_CODE_DETAILS.md**, Section "PLOTTING HELPER FUNCTIONS AND COLOR UTILITIES"
OR **ADDITIONAL_CODE_DETAILS.md**, Section "INLINE PLOTTING AND FIGURE CUSTOMIZATION"

---

## File Organization Map

```
COMPLETE_CODE_EXTRACTION.md
├── FILE 1: 251206_figure_4_updated_figures.py (LATEST)
│   ├── ALL IMPORT STATEMENTS
│   ├── CONFIGURATION VARIABLES
│   ├── DATA LOADING CODE
│   ├── CLUSTERING PROCESSING CALLS
│   ├── PLOTTING CALLS (plotting module)
│   ├── PLOTTING CALLS (ppf module)
│   ├── PLOTTING CALLS (pse module)
│   └── CUSTOM MATPLOTLIB/SEABORN CODE
├── FILE 2: 251206_figure_4_prediction_strength_control.py
├── FILE 3: 260228_clustering_shuffle_control.py
├── FILE 4: 240605_familiar_only_clustering_control.py
├── FILE 5: 240404_figure_4_generate_analysis_files.py
├── UNIQUE FUNCTIONS (in latest, not in older)
├── KEY CONFIGURATION DIRECTORIES
└── DATA FILES LOADED/SAVED

ADDITIONAL_CODE_DETAILS.md
├── DETAILED DATA LOADING SECTION
├── CLUSTERING INITIALIZATION AND SELECTION CODE
├── PREDICTION STRENGTH CROSS-VALIDATION
├── INDIVIDUAL CELL EXAMPLES CODE
├── EXPERIENCE LEVEL MODULATION ANALYSIS
├── RESPONSE METRICS STATISTICAL COMPARISONS
├── PLOTTING HELPER FUNCTIONS AND COLOR UTILITIES
├── INLINE PLOTTING AND FIGURE CUSTOMIZATION
├── HELPER PROCESSING FUNCTIONS
├── COMMON DATA FILTERING AND MERGING PATTERNS
└── VARIABLES USED IN PLOTS

CODE_EXTRACTION_SUMMARY.md
├── Overview and Key Findings
├── COMPLETE IMPORT STATEMENT STRUCTURE
├── DATA LOADING PIPELINE
├── CORE PROCESSING WORKFLOW
├── CONFIGURATION VARIABLES
├── DIRECTORY STRUCTURE
├── PLOTTING MODULE ORGANIZATION
│   ├── Cluster Selection Plots
│   ├── Cluster Validation Plots
│   ├── Dimension Reduction Plots
│   ├── Heatmap Plots
│   ├── Population Response Plots
│   ├── Cell Composition Plots
│   ├── Anatomical Distribution Plots
│   ├── Modulation and Metrics Plots
│   └── Helper Functions
├── PROCESSING MODULE FUNCTION CATEGORIES
│   ├── Data Preparation
│   ├── Clustering
│   ├── Cluster Validation
│   ├── Metrics Computation
│   ├── Statistics
│   └── Utilities
├── STATISTICAL ANALYSIS PATTERNS
├── CONTROL SCRIPTS OVERVIEW
├── DATA FLOW DIAGRAM
├── PARAMETER SENSITIVITY
├── OUTPUT SPECIFICATIONS
├── CODE QUALITY NOTES
└── RECOMMENDATIONS FOR NEW FIGURE SCRIPTS

CODE_EXTRACTION_INDEX.md (this file)
├── Document Overview
├── Source Files Analyzed
├── Quick Reference by Task
├── File Organization Map
├── Key Variables and Constants
├── Data Structures Overview
└── Common Patterns and Conventions
```

---

## Key Variables and Constants

### Clustering Parameters
```python
n_clusters = 14                    # Number of optimal clusters
glm_version = '24_events_all_L2_optimize_by_session'
metric = 'euclidean'               # Distance metric
shuffle_type = 'all'               # Shuffle type for gap statistic
k_max = 25                         # Max clusters to test
```

### Data Loading Parameters
```python
data_type = 'events'
interpolate = True
output_sampling_rate = 30          # Hz
inclusion_criteria = 'platform_experiment_table'
event_type = 'all'
threshold_percentile = 5           # For outlier removal
```

### Response Data Conditions
```python
conditions = ['cell_specimen_id', 'is_change']    # For change responses
conditions = ['cell_specimen_id', 'omitted']      # For omission responses
```

### Plotting Parameters
```python
figsize = (7, 4)                   # Standard figure size
font_scale = 1.5                   # Seaborn context
lines.markeredgewidth = 1          # Marker style
hue_column = 'experience_level'    # For multi-level plots
axes_column = 'cluster_id'         # For subplot organization
```

---

## Data Structures Overview

### Main DataFrames

| DataFrame | Rows | Columns | Purpose |
|-----------|------|---------|---------|
| `platform_experiments` | ~260 | experiment metadata | Experiment information |
| `platform_cells_table` | ~3000 | cell properties | Cell metadata |
| `matched_cells_table` | ~2980 | cells matched across sessions | Matched cells across 3 experience levels |
| `feature_matrix` | ~3000 | 4 (images, omissions, task, behavior) | Clustering input features |
| `cluster_meta` | ~2980 | 10+ columns | Cluster assignments and properties |
| `results_pivoted` | ~3000 | coding scores across features | GLM results normalized across sessions |
| `change_mdf` | ~50M rows | response properties | Image change responses |
| `image_mdf` | ~50M rows | response properties | Repeated image responses |
| `omission_mdf` | ~20M rows | response properties | Omitted stimulus responses |
| `metrics` | ~2980 | 20+ metrics | Merged coding scores and model-free metrics |
| `response_metrics` | varies | response statistics | Statistical summaries |
| `umap_df` | ~3000 | 2 (x, y) | UMAP embedding coordinates |

### Key Dictionaries

| Dictionary | Contents | Purpose |
|-----------|----------|---------|
| `gap_dict` | gap scores for K=1-25 | Gap statistic results |
| `experience_level_colors` | colors for experience levels | Visualization mapping |
| `cre_line_colors` | colors for cre lines | Visualization mapping |
| `coclustering_dict` | co-clustering matrices by cre line | Clustering probabilities |

---

## Common Patterns and Conventions

### Data Filtering Pattern
```python
# Subset by condition
df = mdf[mdf.cre_line == 'Vip']
df = cluster_meta[cluster_meta.cluster_id < 10]
df = response_metrics[response_metrics['metric'] > threshold]

# Merge with additional metadata
df = df.merge(cluster_meta.reset_index()[['cell_specimen_id', 'cluster_id']],
              on='cell_specimen_id')
```

### Function Call Pattern
```python
# Standard plotting with optional save
plotting.plot_function_name(data, metric='column_name',
                           hue='group_column',
                           save_dir=save_dir,
                           folder='subfolder')

# Processing function with multiple steps
data = processing.step1(data, param1=value1)
data = processing.step2(data, param2=value2)
result = processing.step3(data)
```

### Directory Path Pattern
```python
# Construct paths
cache_dir = loading.get_platform_analysis_cache_dir()
data_dir = os.path.join(cache_dir, 'clustering')
save_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\...'

# Check and load/create
filename = os.path.join(data_dir, 'filename.pkl')
if os.path.exists(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
else:
    data = compute_expensive_operation()
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
```

### Figure Creation Pattern
```python
# Create figure
fig, ax = plt.subplots(figsize=(7, 4))

# Plot with seaborn
ax = sns.scatterplot(data=df, x='col1', y='col2', hue='group', ax=ax)

# Adjust and save
plt.subplots_adjust(hspace=0.3)
if save_dir:
    utils.save_figure(fig, figsize, save_dir, folder, name)
```

---

## Statistics Functions Used

### ANOVA and Post-hoc Tests
```python
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison

f_stat, p_value = f_oneway(group1, group2, group3)
tukey = pairwise_tukeyhsd(endog=data, groups=groups, alpha=0.05)
```

### Correlation Functions
```python
from scipy.stats import spearmanr, pearsonr

correlation, pvalue = spearmanr(x, y)
```

### Distance Metrics
```python
from sklearn.metrics.pairwise import euclidean_distances

distances = euclidean_distances(X, Y)
```

---

## Module Aliases and Conventions

| Alias | Full Module | Convention |
|-------|------------|-----------|
| `utils` | visual_behavior.visualization.utils | Utility colors and info |
| `loading` | visual_behavior.data_access.loading | Data loading functions |
| `plotting` | visual_behavior.dimensionality_reduction.clustering.plotting | All plotting |
| `processing` | visual_behavior.dimensionality_reduction.clustering.processing | Data processing |
| `ppf` | visual_behavior.visualization.ophys.platform_paper_figures | Platform-specific plots |
| `pse` | visual_behavior.visualization.ophys.platform_single_cell_examples | Single cell plots |
| `sns` | seaborn | Statistical visualization |
| `plt` | matplotlib.pyplot | Basic plotting |

---

## Next Steps

Use these documents for:
1. **Script Development:** Reference the exact imports and function calls
2. **Understanding Architecture:** Review the data flow diagram and workflows
3. **Code Reuse:** Copy code snippets from ADDITIONAL_CODE_DETAILS.md
4. **Problem Solving:** Use Quick Reference by Task to find relevant sections
5. **Version Control:** Track which functions are new/changed between versions

All code is extracted completely and exactly as written in the source scripts.
