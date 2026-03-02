"""
Supplemental Figures S1-S2 associated with Paper Figure 1 (Dataset summary).

This script generates supplemental figures for the platform paper, including:
- S1: Cell counts by depth for different subsets
- S2: ROI size analysis by depth and matched cells
- VIP/Sst correlation analysis across imaging planes and sessions
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

save_dir = os.path.join(os.getcwd(), 'platform_paper_figures', 'figure_1_supplemental')
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

# add_cell_type_column is available via: utilities.add_cell_type_column()


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
# S1: Cell counts by depth - Mesoscope only
# ============================================================================

cells = platform_cells_table.drop_duplicates('cell_specimen_id')
cells_meso = cells[cells.project_code == 'VisualBehaviorMultiscope']
suptitle = 'VisualBehaviorMultiscope'

ppf.plot_cell_count_by_depth(cells_meso, suptitle=suptitle, save_dir=save_dir, folder='S1', ax=None)

tmp = platform_cells_table.copy()
tmp = tmp[tmp.project_code == 'VisualBehaviorMultiscope']
ppf.plot_n_cells_per_plane_by_depth(tmp, suptitle=suptitle, save_dir=save_dir, folder='S1', ax=None)
ppf.plot_n_planes_per_depth(tmp, suptitle=suptitle, save_dir=save_dir, folder='S1', ax=None)


# ============================================================================
# S1: Cell counts by depth - Platform dataset
# ============================================================================

cells = platform_cells_table.drop_duplicates('cell_specimen_id')
suptitle = 'Platform dataset'

ppf.plot_cell_count_by_depth(cells, suptitle=suptitle, save_dir=save_dir, folder='S1', ax=None)
ppf.plot_n_cells_per_plane_by_depth(cells, suptitle=suptitle, save_dir=save_dir, folder='S1', ax=None)
ppf.plot_n_planes_per_depth(cells, suptitle=suptitle, save_dir=save_dir, folder='S1', ax=None)


# ============================================================================
# S1: Cell counts by depth - Full dataset
# ============================================================================

cells = ophys_cells_table.drop_duplicates('cell_specimen_id')
cells = cells[cells.reporter_line.str.contains('Ai94') == False]
suptitle = 'full dataset (other than Ai94)'

ppf.plot_cell_count_by_depth(cells, suptitle=suptitle, save_dir=save_dir, folder='S1', ax=None)

tmp = ophys_cells_table.copy()
tmp = tmp[tmp.reporter_line.str.contains('Ai94') == False]
ppf.plot_n_cells_per_plane_by_depth(tmp, suptitle=suptitle, save_dir=save_dir, folder='S1', ax=None)
ppf.plot_n_planes_per_depth(tmp, suptitle=suptitle, save_dir=save_dir, folder='S1', ax=None)


# ============================================================================
# S1: Cell counts by depth - Vertically organized with all areas
# ============================================================================

cells = ophys_cells_table.drop_duplicates('cell_specimen_id')
cells = cells[cells.reporter_line.str.contains('Ai94') == False]
suptitle = 'Full dataset (excluding Ai94)'
suffix = '_full_dataset'
ppf.plot_cell_count_by_depth(
    cells, project_code=None, suptitle=suptitle, horiz=False,
    save_dir=save_dir, folder='S1', suffix=suffix, ax=None
)


# ============================================================================
# S1: Cell counts by project code
# ============================================================================

cells = ophys_cells_table.drop_duplicates('cell_specimen_id')
cells = cells[cells.reporter_line.str.contains('Ai94') == False]

project_codes = ['VisualBehavior', 'VisualBehaviorTask1B', 'VisualBehaviorMultiscope', 'VisualBehaviorMultiscope4areasx2d']

for i, project_code in enumerate(project_codes):
    proj_cells = cells[cells.project_code == project_code]
    suptitle = 'Cohort ' + str(i + 1)
    suffix = '_' + project_code
    ppf.plot_cell_count_by_depth(
        proj_cells, project_code=project_code, suptitle=suptitle, horiz=False,
        save_dir=save_dir, folder='S1', suffix=suffix, ax=None
    )


# ============================================================================
# S1: Platform dataset with horizontal layout
# ============================================================================

cells = platform_cells_table.drop_duplicates('cell_specimen_id')
suptitle = 'Platform dataset'
suffix = '_platform_cells_table'
ppf.plot_cell_count_by_depth(
    cells, project_code='VisualBehaviorMultiscope', suptitle=suptitle, horiz=True,
    save_dir=save_dir, folder='S1', suffix=suffix, ax=None
)


# ============================================================================
# S1: All acquired data
# ============================================================================

filepath = os.path.join(platform_cache_dir, '..', 'all_acquired_vb_ophys_experiments.xlsx')
if os.path.exists(filepath):
    df = pd.read_excel(filepath)
    df = utilities.add_cell_type_column(df)
    plot_expt_count_by_depth_tmp(df, suptitle='all acquired data', save_dir=save_dir, folder='S1', ax=None)


# ============================================================================
# S2: ROI size analysis by depth
# ============================================================================

# KDE plot of roi_size by imaging_depth per cell type
figsize = (9, 3)
fig, ax = plt.subplots(1, 3, figsize=figsize, sharey=True, sharex=True)

for i, cell_type in enumerate(utils.get_cell_types()):
    ct_cells = cells_meso[cells_meso.cell_type == cell_type]
    ax[i] = sns.kdeplot(data=ct_cells, y='imaging_depth', x='roi_size', ax=ax[i])
    ax[i].set_title(cell_type)
    ax[i].invert_yaxis()
    ax[i].set_xlabel('Cell size (pixels)')
    ax[i].set_ylabel('Imaging depth (um)')

if save_dir:
    utils.save_figure(fig, figsize, save_dir, 'S2', 'roi_size_by_depth_kdeplot')


# Boxplot of roi_size by cell_type and binned_depth
figsize = (3, 3)
fig, ax = plt.subplots(figsize=figsize)
ax = sns.boxplot(data=cells_meso, x='cell_type', y='roi_size', hue='binned_depth', ax=ax)
ax.set_ylabel('Cell size (pixels)')
ax.set_xlabel('Cell type')
ax.legend(bbox_to_anchor=(1, 1), fontsize='xx-small')

if save_dir:
    utils.save_figure(fig, figsize, save_dir, 'S2', 'roi_size_by_cell_type_and_depth_boxplot')


# KDE plot of roi_width_over_height by imaging_depth per cell type
figsize = (9, 3)
fig, ax = plt.subplots(1, 3, figsize=figsize, sharey=True, sharex=True)

for i, cell_type in enumerate(utils.get_cell_types()):
    ct_cells = cells_meso[cells_meso.cell_type == cell_type]
    ax[i] = sns.kdeplot(data=ct_cells, y='imaging_depth', x='roi_width_over_height', ax=ax[i])
    ax[i].set_title(cell_type)
    ax[i].invert_yaxis()

if save_dir:
    utils.save_figure(fig, figsize, save_dir, 'S2', 'roi_width_over_height_by_depth_kdeplot')


# ============================================================================
# S2: Matched cells ROI and coding scores
# ============================================================================

matched_cells_meso = cells_meso[cells_meso.cell_specimen_id.isin(matched_cells_table.cell_specimen_id.unique())]
std = matched_cells_meso.groupby(['cell_specimen_id']).std()
small_diff_cells = std[std.roi_size < 20].index.unique()

for cell_specimen_id in small_diff_cells[:20]:
    cell_metadata = matched_cells_table[matched_cells_table.cell_specimen_id == cell_specimen_id]
    cell_dropouts = results_pivoted[results_pivoted.cell_specimen_id == cell_specimen_id]
    pse.plot_matched_roi_and_coding_scores(
        cell_metadata, cell_dropouts, platform_experiments,
        save_dir=os.path.join(save_dir, 'S2')
    )


# ============================================================================
# VIP Correlation Analysis - Single mouse (Vip example)
# ============================================================================

mouse_id = 449653
mouse_expts = experiments_table[experiments_table.mouse_id == mouse_id].sort_values(
    by=['date_of_acquisition', 'targeted_structure', 'imaging_depth']
)
ophys_session_ids = [872592724, 873247524, 876303107]

for ophys_session_id in ophys_session_ids:
    ophys_experiment_ids = experiments_table[
        (experiments_table.ophys_session_id == ophys_session_id)
    ].index.values

    data_dict = loading.get_data_dict(ophys_experiment_ids, data_types=['filtered_events'])

    sdf = pd.DataFrame()
    for ophys_experiment_id in ophys_experiment_ids:
        tmp = data_dict[ophys_experiment_id]['filtered_events']['stimulus_response_df']
        tmp['ophys_experiment_id'] = ophys_experiment_id
        sdf = pd.concat([sdf, tmp])

    data = sdf[['cell_specimen_id', 'stimulus_presentations_id', 'mean_response', 'ophys_experiment_id']]
    data = data.merge(mouse_expts[['imaging_depth']], on='ophys_experiment_id')

    pivot = data.pivot_table(index='stimulus_presentations_id', values='mean_response', columns='cell_specimen_id')
    corr = pivot.corr()

    imaging_depths = data[['cell_specimen_id', 'imaging_depth']]
    imaging_depths = imaging_depths.drop_duplicates()
    imaging_depths = imaging_depths.set_index('cell_specimen_id')
    imaging_depths = imaging_depths.loc[corr.index.values]

    lut = dict(zip(np.sort(imaging_depths.imaging_depth.unique()), sns.color_palette('RdBu')))
    row_colors = imaging_depths.imaging_depth.map(lut)

    g = sns.clustermap(pivot.corr(), cmap='magma', vmin=0, vmax=0.5, row_colors=row_colors)
    session_type = mouse_expts[mouse_expts.ophys_session_id == ophys_session_id].session_type.unique()[0]
    g.fig.suptitle(session_type, y=1.01)

    figdir = os.path.join(save_dir, 'vip_correlations')
    if not os.path.exists(figdir):
        os.makedirs(figdir)
    g.savefig(os.path.join(figdir, f'Vip_449653_{ophys_session_id}_{session_type}.png'))


# ============================================================================
# VIP Correlation Analysis - Multiple Vip mice
# ============================================================================

vip_mice_cells = ophys_cells_table[
    (ophys_cells_table.cell_type == 'Vip Inhibitory') &
    (ophys_cells_table.project_code != 'VisualBehaviorMultiscope4areasx2d')
].groupby(['mouse_id', 'cell_specimen_id']).count()[['ophys_experiment_id']].reset_index().groupby('mouse_id').count().sort_values(
    by='cell_specimen_id', ascending=False
)

for mouse_idx in [3, 4]:
    mouse_id = int(vip_mice_cells.index.values[mouse_idx])
    mouse_expts = platform_experiments[platform_experiments.mouse_id == mouse_id].sort_values(
        by=['date_of_acquisition', 'targeted_structure', 'imaging_depth']
    )
    sessions = mouse_expts.groupby(['ophys_session_id', 'session_type', 'date_of_acquisition']).count()
    ophys_session_ids = sessions.sort_values(by='date_of_acquisition').reset_index().ophys_session_id.values

    ophys_experiment_ids = mouse_expts.index.values
    data_dict = loading.get_data_dict(ophys_experiment_ids, data_types=['filtered_events'])

    for ophys_session_id in ophys_session_ids:
        ophys_experiment_ids = mouse_expts[
            (mouse_expts.ophys_session_id == ophys_session_id)
        ].index.values

        sdf = pd.DataFrame()
        for ophys_experiment_id in ophys_experiment_ids:
            tmp = data_dict[ophys_experiment_id]['filtered_events']['stimulus_response_df']
            tmp['ophys_experiment_id'] = ophys_experiment_id
            sdf = pd.concat([sdf, tmp])

        data = sdf[['cell_specimen_id', 'stimulus_presentations_id', 'mean_response', 'ophys_experiment_id']]
        data = data.merge(mouse_expts[['imaging_depth']], on='ophys_experiment_id')

        pivot = data.pivot_table(index='stimulus_presentations_id', values='mean_response', columns='cell_specimen_id')
        corr = pivot.corr()

        imaging_depths = data[['cell_specimen_id', 'imaging_depth']]
        imaging_depths = imaging_depths.drop_duplicates()
        imaging_depths = imaging_depths.set_index('cell_specimen_id')
        imaging_depths = imaging_depths.loc[corr.index.values]

        lut = dict(zip(np.sort(imaging_depths.imaging_depth.unique()), sns.color_palette('RdBu')))
        row_colors = imaging_depths.imaging_depth.map(lut)

        g = sns.clustermap(pivot.corr(), cmap='magma', vmin=0, vmax=0.5, row_colors=row_colors)
        session_type = mouse_expts[mouse_expts.ophys_session_id == ophys_session_id].session_type.unique()[0]
        cell_type_abbr = mouse_expts.cell_type.unique()[0][:3]
        g.fig.suptitle(f'{cell_type_abbr}_{mouse_id}_{ophys_session_id}\n{session_type}', y=1.05)

        figdir = os.path.join(save_dir, 'vip_correlations')
        if not os.path.exists(figdir):
            os.makedirs(figdir)
        g.savefig(os.path.join(figdir, f'{cell_type_abbr}_{mouse_id}_{ophys_session_id}_{session_type}.png'))


# ============================================================================
# Sst Correlation Analysis - Multiple mice
# ============================================================================

sst_mice = ophys_cells_table[
    (ophys_cells_table.cell_type == 'Sst Inhibitory') &
    (ophys_cells_table.project_code == 'VisualBehaviorMultiscope')
].groupby(['mouse_id', 'cell_specimen_id']).count()[['ophys_experiment_id']].reset_index().groupby('mouse_id').count().sort_values(
    by='cell_specimen_id', ascending=False
)

for mouse_id in sst_mice.index.values[:5]:
    mouse_id = int(mouse_id)
    mouse_expts = experiments_table[experiments_table.mouse_id == mouse_id].sort_values(
        by=['date_of_acquisition', 'targeted_structure', 'imaging_depth']
    )
    sessions = mouse_expts.groupby(['ophys_session_id', 'session_type', 'date_of_acquisition']).count()
    ophys_session_ids = sessions.sort_values(by='date_of_acquisition').reset_index().ophys_session_id.values

    ophys_experiment_ids = mouse_expts.index.values
    data_dict = loading.get_data_dict(ophys_experiment_ids, data_types=['filtered_events'])

    for ophys_session_id in ophys_session_ids:
        ophys_experiment_ids = mouse_expts[
            (mouse_expts.ophys_session_id == ophys_session_id)
        ].index.values

        sdf = pd.DataFrame()
        for ophys_experiment_id in ophys_experiment_ids:
            tmp = data_dict[ophys_experiment_id]['filtered_events']['stimulus_response_df']
            tmp['ophys_experiment_id'] = ophys_experiment_id
            sdf = pd.concat([sdf, tmp])

        data = sdf[['cell_specimen_id', 'stimulus_presentations_id', 'mean_response', 'ophys_experiment_id']]
        data = data.merge(mouse_expts[['imaging_depth']], on='ophys_experiment_id')

        pivot = data.pivot_table(index='stimulus_presentations_id', values='mean_response', columns='cell_specimen_id')
        corr = pivot.corr()

        imaging_depths = data[['cell_specimen_id', 'imaging_depth']]
        imaging_depths = imaging_depths.drop_duplicates()
        imaging_depths = imaging_depths.set_index('cell_specimen_id')
        imaging_depths = imaging_depths.loc[corr.index.values]

        lut = dict(zip(np.sort(imaging_depths.imaging_depth.unique()), sns.color_palette('RdBu')))
        row_colors = imaging_depths.imaging_depth.map(lut)

        g = sns.clustermap(pivot.corr(), cmap='magma', vmin=0, vmax=0.5, row_colors=row_colors)
        session_type = mouse_expts[mouse_expts.ophys_session_id == ophys_session_id].session_type.unique()[0]
        cell_type_abbr = mouse_expts.cell_type.unique()[0][:3]
        g.fig.suptitle(f'{cell_type_abbr}_{mouse_id}_{ophys_session_id}\n{session_type}', y=1.05)

        figdir = os.path.join(save_dir, 'sst_correlations')
        if not os.path.exists(figdir):
            os.makedirs(figdir)
        g.savefig(os.path.join(figdir, f'{cell_type_abbr}_{mouse_id}_{ophys_session_id}_{session_type}.png'))
