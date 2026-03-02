"""
Paper Figure 2 - Behavioral performance across experience levels

This script generates Figure 2 from the platform paper, showing behavior metrics
across experience levels (Familiar, Novel, Novel+) including d-prime, hit rate,
response latency, running speed, pupil width, and lick rate.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns

sns.set_context('notebook', font_scale=1.5, rc={'lines.markeredgewidth': 2})

import visual_behavior.utilities as vbu
import visual_behavior.data_access.utilities as utilities
from visual_behavior.data_access import loading as loading

import visual_behavior.visualization.utils as utils
import visual_behavior.visualization.ophys.platform_paper_figures as ppf

from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache

# ===== Configuration =====
save_dir = os.path.join(os.getcwd(), 'platform_paper_figures', 'figure_2')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

palette = utils.get_experience_level_colors()
experience_levels = utils.get_new_experience_levels()
cell_types = utils.get_cell_types()

# ===== Data Loading =====

# Get cache and metadata tables
cache_dir = loading.get_platform_analysis_cache_dir()
cache = VisualBehaviorOphysProjectCache.from_local_cache(cache_dir=cache_dir, use_static_cache=True)

# Load metadata CSV files
platform_experiments = pd.read_csv(os.path.join(cache_dir, 'platform_paper_ophys_experiments_table.csv'), index_col=0)
platform_experiments['mouse_id'] = [int(mouse_id) for mouse_id in platform_experiments.mouse_id.values]
platform_experiments['experience_level'] = [utils.convert_experience_level(experience_level)
                                           for experience_level in platform_experiments.experience_level.values]

# Get behavior session table
platform_behavior_sessions = loading.get_platform_paper_behavior_session_table()
platform_behavior_sessions['mouse_id'] = [int(mouse_id) for mouse_id in platform_behavior_sessions.mouse_id.values]
platform_behavior_sessions['experience_level'] = [utils.convert_experience_level(experience_level)
                                                  for experience_level in platform_behavior_sessions.experience_level.values]

# Limit behavior sessions to mice in platform experiments
behavior_sessions = platform_behavior_sessions[platform_behavior_sessions.mouse_id.isin(platform_experiments.mouse_id.unique())]

# Add useful columns for filtering
behavior_sessions = utilities.add_date_string(behavior_sessions)
behavior_sessions = utilities.add_n_relative_to_first_novel_column(behavior_sessions)
behavior_sessions = utilities.add_first_novel_column(behavior_sessions)
behavior_sessions = utilities.add_second_novel_active_column(behavior_sessions)
behavior_sessions = utilities.add_last_familiar_active_column(behavior_sessions)
behavior_sessions['ophys_container_id'] = behavior_sessions.mouse_id.values

# ===== Load behavior performance statistics =====

# SDK behavior stats
behavior_session_ids = behavior_sessions.index.values
behavior_stats_sdk, _ = vbu.get_behavior_stats_for_sessions(behavior_session_ids, behavior_sessions,
                                                            method='sdk', engaged_only=False, per_image=False)
behavior_stats_sdk = behavior_stats_sdk.merge(behavior_sessions, on='behavior_session_id')

# Stimulus-based behavior stats for engaged trials
engaged_behavior_stats_stim, _ = vbu.get_behavior_stats_for_sessions(behavior_session_ids, behavior_sessions,
                                                                     method='stimulus_based', engaged_only=True, per_image=False)
engaged_behavior_stats_stim = engaged_behavior_stats_stim.merge(behavior_sessions, on='behavior_session_id')
# Remove 4x2 sessions
engaged_behavior_stats_stim = engaged_behavior_stats_stim[engaged_behavior_stats_stim.project_code != 'VisualBehaviorMultiscope4areasx2d']

# ===== Load multi-session dataframes for behavioral metrics =====

inclusion_criteria = 'platform_experiment_table'

# Running speed
event_type = 'all'
data_type = 'running_speed'
conditions = ['ophys_experiment_id', 'is_change']
running_speed_mdf = loading.get_multi_session_df_for_conditions(data_type, event_type, conditions, inclusion_criteria,
                                                               interpolate=False, output_sampling_rate=None,
                                                               epoch_duration_mins=None)
running_speed_mdf_changes = running_speed_mdf[running_speed_mdf.is_change == True]
running_speed_mdf_changes = running_speed_mdf_changes.drop_duplicates(subset='behavior_session_id')

# Running speed - epochs
conditions_epochs = ['ophys_experiment_id', 'is_change', 'epoch']
running_speed_mdf_epochs = loading.get_multi_session_df_for_conditions(data_type, 'all', conditions_epochs, inclusion_criteria,
                                                                       interpolate=False, output_sampling_rate=None,
                                                                       epoch_duration_mins=5)
running_speed_mdf_changes_epochs = running_speed_mdf_epochs[running_speed_mdf_epochs.is_change == True]
running_speed_mdf_images_epochs = running_speed_mdf_epochs[running_speed_mdf_epochs.is_change == False]

# Pupil width
data_type = 'pupil_width'
conditions = ['ophys_experiment_id', 'is_change']
full_pupil_mdf = loading.get_multi_session_df_for_conditions(data_type, 'all', conditions, inclusion_criteria,
                                                            interpolate=True, output_sampling_rate=30,
                                                            epoch_duration_mins=None)
pupil_mdf = full_pupil_mdf[full_pupil_mdf.is_change == True]
pupil_mdf = pupil_mdf.drop_duplicates(subset='behavior_session_id')
pupil_mdf = pupil_mdf[pupil_mdf.behavior_session_id.isin(platform_experiments.behavior_session_id.unique())]

# Pupil width - epochs
conditions_epochs = ['ophys_experiment_id', 'is_change', 'epoch']
pupil_mdf_epochs = loading.get_multi_session_df_for_conditions(data_type, 'all', conditions_epochs, inclusion_criteria,
                                                              interpolate=True, output_sampling_rate=30,
                                                              epoch_duration_mins=5)
pupil_mdf_changes_epochs = pupil_mdf_epochs[pupil_mdf_epochs.is_change == True]

# Lick rate
data_type = 'lick_rate'
conditions = ['ophys_experiment_id', 'is_change']
lick_rate_mdf = loading.get_multi_session_df_for_conditions(data_type, 'all', conditions, inclusion_criteria,
                                                           interpolate=False, output_sampling_rate=None,
                                                           epoch_duration_mins=None)
lick_rate_mdf = lick_rate_mdf[lick_rate_mdf.is_change == True]
lick_rate_mdf = lick_rate_mdf.drop_duplicates(subset='behavior_session_id')
lick_rate_mdf = lick_rate_mdf[lick_rate_mdf.behavior_session_id.isin(platform_experiments.behavior_session_id.unique())]

# Lick rate - epochs
conditions_epochs = ['ophys_experiment_id', 'is_change', 'epoch']
lick_rate_mdf_epochs = loading.get_multi_session_df_for_conditions(data_type, 'all', conditions_epochs, inclusion_criteria,
                                                                  interpolate=False, output_sampling_rate=None,
                                                                  epoch_duration_mins=5)
lick_rate_mdf_changes_epochs = lick_rate_mdf_epochs[lick_rate_mdf_epochs.is_change == True]
lick_rate_mdf_images_epochs = lick_rate_mdf_epochs[lick_rate_mdf_epochs.is_change == False]

# Load behavior performance CSV
platform_behavior_stats = pd.read_csv(os.path.join(cache_dir, 'behavior_performance', 'platform_behavior_stats_engaged.csv'),
                                     index_col=0)

# ===== Panel A: d-prime by experience =====
metric = 'mean_dprime_engaged'
stats = behavior_stats_sdk.copy()
stats = stats[stats[metric] > 0]
stats = stats[stats.behavior_session_id.isin(platform_experiments.behavior_session_id.unique())]

ppf.plot_behavior_metric_by_experience(stats, metric, title='behavior\nperformance', ylabel='d-prime', ylims=[-0.1, 3.3],
                                      best_image=False, show_mice=False, plot_stats=True,
                                      stripplot=True, show_ns=False,
                                      save_dir=save_dir, folder='behavior_metrics', suffix='_mean_dprime_engaged')

# ===== Panel B: hit rate and response latency by experience (horizontal) =====
metric = 'max_dprime'
stats = behavior_stats_sdk.copy()
stats = stats[stats.behavior_session_id.isin(platform_experiments.behavior_session_id.unique())]

ppf.plot_behavior_metric_by_experience_horiz(stats, metric, title='Behavior performance', xlabel='d-prime', xlims=[-0.1, 3.3],
                                            best_image=False, show_containers=False, plot_stats=False, stripplot=True, show_ns=False,
                                            save_dir=save_dir, folder='behavior_metrics', suffix='_max_dprime_horiz')

# ===== Panel C: Population averages across experience levels =====
df = running_speed_mdf_changes.copy()
df['expts'] = 'all'

axes_column = 'expts'
hue_column = 'experience_level'
xlim_seconds = [-1, 1.5]

ppf.plot_population_averages_across_experience(df, xlim_seconds=xlim_seconds,
                                               xlabel='Time from image change (sec)',
                                               ylabel='Running speed (cm/s)',
                                               data_type='running_speed', event_type='changes',
                                               interval_sec=0.5,
                                               palette=palette, ax=None, title=None,
                                               save_dir=save_dir, folder='behavior_metrics',
                                               suffix='_population_averages_running_speed')

# ===== Panel D: Running speed population averages =====
df = running_speed_mdf_changes.copy()
df['expts'] = 'all'

axes_column = 'expts'
hue_column = 'experience_level'
xlim_seconds = [-1, 1.5]

ax = ppf.plot_population_averages_for_conditions(df, 'running_speed', 'changes',
                                                axes_column, hue_column, horizontal=True,
                                                ylabel='Running speed (cm/s)',
                                                xlim_seconds=xlim_seconds, interval_sec=1,
                                                xlabel='Time from image change (sec)',
                                                palette=palette, ax=None, title='Running speed - changes',
                                                suptitle=None, save_dir=save_dir, folder='behavior_metrics', suffix='_running_speed_changes')

# ===== Panel E: Pupil width population averages =====
df = pupil_mdf.copy()
# Set baseline to 0 instead of 1
df['mean_trace'] = [mean_trace - 1 for mean_trace in df.mean_trace.values]
df['expts'] = 'all'

axes_column = 'expts'
hue_column = 'experience_level'
xlim_seconds = [-1, 1.5]

ax = ppf.plot_population_averages_for_conditions(df, 'pupil_width', 'changes',
                                                axes_column, hue_column, horizontal=True,
                                                xlim_seconds=xlim_seconds, interval_sec=1,
                                                palette=palette, ax=None, title='Pupil width - changes',
                                                suptitle=None, ylabel='Pupil width (norm.)',
                                                xlabel='Time from image change (sec)',
                                                save_dir=save_dir, folder='behavior_metrics', suffix='_pupil_width_changes')

# ===== Panel F: Lick rate population averages =====
df = lick_rate_mdf.copy()
# Put in units of licks/sec
df['mean_trace'] = [mean_trace * 10 for mean_trace in df.mean_trace.values]
df['expts'] = 'all'

axes_column = 'expts'
hue_column = 'experience_level'
xlim_seconds = [-1, 1.5]

suffix = '_overlay'
ax = ppf.plot_population_averages_for_conditions(df, 'lick_rate', 'changes',
                                                axes_column, hue_column, horizontal=True,
                                                xlim_seconds=xlim_seconds, interval_sec=1,
                                                palette=palette, ax=None, title='Lick rate - changes',
                                                suptitle=None, ylabel='Lick rate (licks/s)',
                                                xlabel='Time from image change (sec)',
                                                save_dir=save_dir, folder='behavior_metrics', suffix='_lick_rate_changes')

# ===== Panel G: Mean response by epoch =====
# Running speed by epoch
suffix = '_running_speed_changes_within_session'
ppf.plot_mean_response_by_epoch(running_speed_mdf_changes_epochs, metric='mean_response', horizontal=True, ymin=0,
                               ylabel='Image changes\nMean run speed (cm/s)', estimator=np.mean,
                               save_dir=save_dir, folder='epochs', max_epoch=11, suptitle=None, suffix=suffix, ax=None)

# Pupil width by epoch
suffix = '_pupil_width_changes_within_session'
ppf.plot_mean_response_by_epoch(pupil_mdf_changes_epochs, metric='mean_response', horizontal=True, ymin=0.9,
                               ylabel='Image changes\nMean pupil width (norm.)', estimator=np.mean,
                               save_dir=save_dir, folder='epochs', max_epoch=11, suptitle=None, suffix=suffix, ax=None)

# Lick rate by epoch
suffix = '_lick_rate_changes_within_session'
ppf.plot_mean_response_by_epoch(lick_rate_mdf_changes_epochs, metric='mean_response', horizontal=True, ymin=None,
                               ylabel='Image changes\nLick rate (licks/s)', estimator=np.mean, ymax=0.03,
                               save_dir=save_dir, folder='epochs', max_epoch=11, suptitle=None, suffix=suffix, ax=None)

# ===== Panel H: Lick raster for Familiar/Novel/Novel+ comparison =====
# Select example mouse with data at all experience levels
mouse_id = 425496

# Get datasets for each experience level
familiar_experiment_id = platform_experiments[(platform_experiments.mouse_id == mouse_id) &
                                             (platform_experiments.experience_level == 'Familiar')].index.values[0]
familiar_dataset = loading.get_ophys_dataset(familiar_experiment_id)

novel_experiment_id = platform_experiments[(platform_experiments.experience_level == 'Novel') &
                                         (platform_experiments.mouse_id == mouse_id)].index.values[0]
novel_dataset = loading.get_ophys_dataset(novel_experiment_id)

novel_plus_experiment_id = platform_experiments[(platform_experiments.experience_level == 'Novel +') &
                                              (platform_experiments.mouse_id == mouse_id)].index.values[0]
novel_plus_dataset = loading.get_ophys_dataset(novel_plus_experiment_id)

# Plot lick rasters for F/N/N+
colors = utils.get_experience_level_colors()
figsize = (12, 4)
fig, ax = plt.subplots(1, 3, figsize=figsize, sharey=True)

trials = familiar_dataset.trials.copy()
trials = trials[trials.go == True]
trials = trials[:100]
ax[0] = ppf.plot_lick_raster_for_trials(trials, title='Familiar', save_dir=None, suffix='_familiar', ax=ax[0])
ax[0].set_title('Familiar', color=colors[0])
ax[0].set_xlabel('')

trials = novel_dataset.trials.copy()
trials = trials[trials.go == True]
trials = trials[:100]
ax[1] = ppf.plot_lick_raster_for_trials(trials, title='Novel', save_dir=None, suffix='_novel', ax=ax[1])
ax[1].set_title('Novel', color=colors[1])
ax[1].set_ylabel('')

trials = novel_plus_dataset.trials.copy()
trials = trials[trials.go == True]
trials = trials[:100]
ax[2] = ppf.plot_lick_raster_for_trials(trials, title='Novel +', save_dir=None, suffix='_novel_plus', ax=ax[2])
ax[2].set_title('Novel +', color=colors[2])
ax[2].set_ylabel('')
ax[2].set_xlabel('')

plt.subplots_adjust(wspace=0.3)
plt.suptitle(mouse_id, x=0.6, y=1.03, fontsize=16)
utils.save_figure(fig, figsize, save_dir, 'lick_rasters', str(mouse_id) + '_F_N_N+_example_fig')

print("Figure 2 generation complete. Figures saved to:", save_dir)


# ============================================================================
# Load stimulus images for schematics (Panels A, B)
# ============================================================================
# Use the familiar experiment to get stimulus templates
stimulus_templates = familiar_dataset.stimulus_templates.copy()
image_pre = stimulus_templates.loc['im061']['unwarped'].copy()   # repeated image
image_post = stimulus_templates.loc['im065']['unwarped'].copy()  # change image


# ============================================================================
# Panel A schematic: Image change trial structure
# ============================================================================

def plot_figure_2A_schematic(image_pre=None, image_post=None, ax=None,
                              save_dir=None, folder=None, suffix=''):
    """
    Reproduce the Figure 2A image change schematic showing a sequence of
    repeated images (250 ms each) interleaved with gray screens (500 ms),
    followed by an image identity change and reward window.

    Parameters
    ----------
    image_pre : 2D array, optional
        The repeated image shown before the change (e.g., im061 unwarped).
    image_post : 2D array, optional
        The new image shown after the change (e.g., im065 unwarped).
    ax : matplotlib Axes, optional
        If provided, plot on this axes. Otherwise create a new figure.
    save_dir : str, optional
        Directory to save figure.
    folder : str, optional
        Subfolder within save_dir.
    suffix : str
        Suffix for saved filename.

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 2.8))
    else:
        fig = ax.figure

    ax.set_xlim(-1.5, 22)
    ax.set_ylim(-3.8, 4.5)
    ax.axis('off')

    # Layout parameters
    img_width = 1.8
    gray_width = 1.0
    img_height = 2.0
    y_center = 0
    y_bot = y_center - img_height / 2
    zoom = 0.18

    # Title
    ax.text(10, 3.8, 'Image change', fontsize=11, fontweight='bold',
            ha='center', va='center', color='#333333')

    # Ellipsis dots on far left
    for dx in [0, 0.5, 1.0]:
        ax.plot(-0.5 + dx, y_center, 'o', color='#999999', markersize=3, zorder=2)

    # Sequence: [img_pre] [gray] [img_pre] [gray] [img_pre] [gray] [img_post]
    x_cursor = 1.5
    n_repeats = 3

    positions_pre = []
    positions_gray = []

    for i in range(n_repeats):
        # Image frame
        positions_pre.append(x_cursor)
        rect = mpatches.FancyBboxPatch((x_cursor, y_bot), img_width, img_height,
                                        boxstyle='round,pad=0.02',
                                        facecolor='white', edgecolor='#cccccc',
                                        linewidth=0.5, zorder=1)
        ax.add_patch(rect)
        if image_pre is not None:
            im = OffsetImage(image_pre, zoom=zoom, cmap='gray')
            ab = AnnotationBbox(im, (x_cursor + img_width / 2, y_center),
                               frameon=False, zorder=3)
            ax.add_artist(ab)
        x_cursor += img_width + 0.15

        # Gray screen
        positions_gray.append(x_cursor)
        rect_gray = mpatches.FancyBboxPatch((x_cursor, y_bot), gray_width, img_height,
                                             boxstyle='round,pad=0.02',
                                             facecolor='#d0d0d0', edgecolor='#bbbbbb',
                                             linewidth=0.5, zorder=1)
        ax.add_patch(rect_gray)
        x_cursor += gray_width + 0.15

    # Change image (different identity) with red border
    change_x = x_cursor
    rect_change = mpatches.FancyBboxPatch((change_x, y_bot), img_width, img_height,
                                           boxstyle='round,pad=0.02',
                                           facecolor='white', edgecolor='#cc3333',
                                           linewidth=1.5, zorder=1)
    ax.add_patch(rect_change)
    if image_post is not None:
        im_post = OffsetImage(image_post, zoom=zoom, cmap='gray')
        ab_post = AnnotationBbox(im_post, (change_x + img_width / 2, y_center),
                                 frameon=False, zorder=3)
        ax.add_artist(ab_post)

    # Reward window bracket below change image
    bracket_y = y_bot - 0.7
    bracket_left = change_x
    bracket_right = change_x + img_width + 1.8
    ax.annotate('', xy=(bracket_left, bracket_y), xytext=(bracket_right, bracket_y),
                arrowprops=dict(arrowstyle='<->', color='#666666', lw=1.0))
    ax.text((bracket_left + bracket_right) / 2, bracket_y - 0.35,
            'Reward window\n750 ms', fontsize=7, ha='center', va='top',
            color='#666666')

    # Ellipsis dots after change
    for dx in [0, 0.5, 1.0]:
        ax.plot(change_x + img_width + 0.5 + dx, y_center, 'o',
                color='#999999', markersize=3, zorder=2)

    if save_dir is not None:
        figsize = fig.get_size_inches()
        utils.save_figure(fig, figsize, save_dir, folder,
                          'figure_2A_schematic' + suffix, formats=['.pdf', '.png'])

    return ax


# Generate Panel A schematic
plot_figure_2A_schematic(image_pre=image_pre, image_post=image_post,
                          save_dir=save_dir, folder='schematics', suffix='')


# ============================================================================
# Panel B schematic: Image omission trial structure
# ============================================================================

def plot_figure_2B_schematic(image=None, ax=None,
                              save_dir=None, folder=None, suffix=''):
    """
    Reproduce the Figure 2B image omission schematic showing a sequence of
    repeated images interleaved with gray screens, then an unexpected omission
    (gray where an image was expected).

    Parameters
    ----------
    image : 2D array, optional
        The repeated image (e.g., im061 unwarped).
    ax : matplotlib Axes, optional
        If provided, plot on this axes. Otherwise create a new figure.
    save_dir : str, optional
        Directory to save figure.
    folder : str, optional
        Subfolder within save_dir.
    suffix : str
        Suffix for saved filename.

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 2.8))
    else:
        fig = ax.figure

    ax.set_xlim(-1.5, 22)
    ax.set_ylim(-3.8, 4.5)
    ax.axis('off')

    # Layout parameters
    img_width = 1.8
    gray_width = 1.0
    img_height = 2.0
    y_center = 0
    y_bot = y_center - img_height / 2
    zoom = 0.18

    # Title
    ax.text(10, 3.8, 'Image omission', fontsize=11, fontweight='bold',
            ha='center', va='center', color='#333333')

    # Ellipsis dots on far left
    for dx in [0, 0.5, 1.0]:
        ax.plot(-0.5 + dx, y_center, 'o', color='#999999', markersize=3, zorder=2)

    x_cursor = 1.5

    # --- Image 1 ---
    img1_x = x_cursor
    rect1 = mpatches.FancyBboxPatch((x_cursor, y_bot), img_width, img_height,
                                     boxstyle='round,pad=0.02',
                                     facecolor='white', edgecolor='#cccccc',
                                     linewidth=0.5, zorder=1)
    ax.add_patch(rect1)
    if image is not None:
        im1 = OffsetImage(image, zoom=zoom, cmap='gray')
        ab1 = AnnotationBbox(im1, (x_cursor + img_width / 2, y_center),
                             frameon=False, zorder=3)
        ax.add_artist(ab1)
    x_cursor += img_width + 0.15

    # --- Gray 1 ---
    gray1_x = x_cursor
    rect_g1 = mpatches.FancyBboxPatch((x_cursor, y_bot), gray_width, img_height,
                                       boxstyle='round,pad=0.02',
                                       facecolor='#d0d0d0', edgecolor='#bbbbbb',
                                       linewidth=0.5, zorder=1)
    ax.add_patch(rect_g1)
    x_cursor += gray_width + 0.15

    # --- Image 2 ---
    img2_x = x_cursor
    rect2 = mpatches.FancyBboxPatch((x_cursor, y_bot), img_width, img_height,
                                     boxstyle='round,pad=0.02',
                                     facecolor='white', edgecolor='#cccccc',
                                     linewidth=0.5, zorder=1)
    ax.add_patch(rect2)
    if image is not None:
        im2 = OffsetImage(image, zoom=zoom, cmap='gray')
        ab2 = AnnotationBbox(im2, (x_cursor + img_width / 2, y_center),
                             frameon=False, zorder=3)
        ax.add_artist(ab2)
    x_cursor += img_width + 0.15

    # --- Gray 2 ---
    gray2_x = x_cursor
    rect_g2 = mpatches.FancyBboxPatch((x_cursor, y_bot), gray_width, img_height,
                                       boxstyle='round,pad=0.02',
                                       facecolor='#d0d0d0', edgecolor='#bbbbbb',
                                       linewidth=0.5, zorder=1)
    ax.add_patch(rect_g2)
    x_cursor += gray_width + 0.15

    # --- OMISSION (dashed cyan border, light gray fill) ---
    omission_x = x_cursor
    rect_omit = mpatches.FancyBboxPatch((x_cursor, y_bot), img_width, img_height,
                                         boxstyle='round,pad=0.02',
                                         facecolor='#e8e8e8', edgecolor='#44aacc',
                                         linewidth=1.5, linestyle='--', zorder=1)
    ax.add_patch(rect_omit)
    # Dashed cyan vertical line indicating expected stimulus timing
    ax.plot([x_cursor + img_width / 2, x_cursor + img_width / 2],
            [y_bot - 0.15, y_bot + img_height + 0.15],
            ls='--', color='#44aacc', lw=1.5, zorder=2, alpha=0.7)
    x_cursor += img_width + 0.15

    # --- Gray 3 (after omission) ---
    gray3_x = x_cursor
    rect_g3 = mpatches.FancyBboxPatch((x_cursor, y_bot), gray_width, img_height,
                                       boxstyle='round,pad=0.02',
                                       facecolor='#d0d0d0', edgecolor='#bbbbbb',
                                       linewidth=0.5, zorder=1)
    ax.add_patch(rect_g3)
    x_cursor += gray_width + 0.15

    # --- Image 3 (resumes after omission) ---
    img3_x = x_cursor
    rect3 = mpatches.FancyBboxPatch((x_cursor, y_bot), img_width, img_height,
                                     boxstyle='round,pad=0.02',
                                     facecolor='white', edgecolor='#cccccc',
                                     linewidth=0.5, zorder=1)
    ax.add_patch(rect3)
    if image is not None:
        im3 = OffsetImage(image, zoom=zoom, cmap='gray')
        ab3 = AnnotationBbox(im3, (x_cursor + img_width / 2, y_center),
                             frameon=False, zorder=3)
        ax.add_artist(ab3)
    x_cursor += img_width

    # Ellipsis dots after
    for dx in [0, 0.5, 1.0]:
        ax.plot(x_cursor + 0.5 + dx, y_center, 'o',
                color='#999999', markersize=3, zorder=2)

    # --- Timing annotations ---
    anno_y = y_bot - 0.5

    # "250 ms stimulus" under first image
    ax.text(img1_x + img_width / 2, anno_y, '250 ms\nstimulus', fontsize=6,
            ha='center', va='top', color='#666666')

    # Bracket spanning gray1 + image2 + gray2 = 1250 ms total cycle
    bracket_y2 = anno_y - 0.15
    bracket_left = gray1_x
    bracket_right = gray2_x + gray_width
    ax.annotate('', xy=(bracket_left, bracket_y2), xytext=(bracket_right, bracket_y2),
                arrowprops=dict(arrowstyle='<->', color='#666666', lw=0.8))
    ax.text((bracket_left + bracket_right) / 2, bracket_y2 - 0.3,
            '1250 ms', fontsize=6.5, ha='center', va='top', color='#666666')

    # "500 ms gray" under gray3
    ax.text(gray3_x + gray_width / 2, anno_y, '500 ms\ngray', fontsize=6,
            ha='center', va='top', color='#666666')

    if save_dir is not None:
        figsize = fig.get_size_inches()
        utils.save_figure(fig, figsize, save_dir, folder,
                          'figure_2B_schematic' + suffix, formats=['.pdf', '.png'])

    return ax


# Generate Panel B schematic
plot_figure_2B_schematic(image=image_pre,
                          save_dir=save_dir, folder='schematics', suffix='')


# ============================================================================
# Panel D schematic: Training timeline / experimental design
# ============================================================================

def plot_figure_2D_schematic(ax=None, save_dir=None, folder=None, suffix=''):
    """
    Reproduce the Figure 2D training timeline schematic showing
    the experimental design: behavior training sessions followed by
    2-photon imaging sessions (Familiar, Novel, Novel+).

    Parameters
    ----------
    ax : matplotlib Axes, optional
        If provided, plot on this axes. Otherwise create a new figure.
    save_dir : str, optional
        Directory to save figure. If None, figure is not saved.
    folder : str, optional
        Subfolder within save_dir.
    suffix : str
        Suffix for saved filename.

    Returns
    -------
    ax : matplotlib Axes
    """
    import matplotlib.patches as mpatches

    # --- Colors ---
    reds = sns.color_palette('Reds_r', 6)[:5][::2]
    blues = sns.color_palette('Blues_r', 6)[:5][::2]
    purples = sns.color_palette('Purples_r', 6)[:5][::2]
    familiar_color = blues[0]
    novel_color = reds[0]
    novel_plus_color = purples[0]
    training_color = '#3b3b3b'  # dark gray for training boxes
    text_gray = '#555555'

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 2.5))
    else:
        fig = ax.figure

    ax.set_xlim(-0.5, 30)
    ax.set_ylim(-2.8, 3.5)
    ax.axis('off')

    box_w = 0.95
    box_h = 1.2
    box_y = -box_h / 2  # center boxes at y=0
    gap = 0.25  # gap between boxes

    # Horizontal dotted timeline
    ax.plot([-0.3, 29], [0, 0], ls=':', color='#cccccc', lw=0.8, zorder=0)

    # ---- Behavior training (left side) ----

    # Ellipsis dots before training boxes
    for dx in [0, 0.45, 0.9]:
        ax.plot(dx, 0, 'o', color=training_color, markersize=3.5, zorder=2)

    # Training session boxes (6 boxes)
    training_start_x = 1.8
    n_training = 6
    for i in range(n_training):
        x = training_start_x + i * (box_w + gap)
        rect = mpatches.FancyBboxPatch((x, box_y), box_w, box_h,
                                        boxstyle='round,pad=0.05',
                                        facecolor=training_color, edgecolor='none',
                                        zorder=2)
        ax.add_patch(rect)

    training_end_x = training_start_x + n_training * (box_w + gap) - gap + box_w

    # Header
    training_center = (training_start_x + training_end_x) / 2
    ax.text(training_center, 2.7, 'Behavior training', fontsize=10, fontweight='bold',
            ha='center', va='center')
    ax.text(training_center, 2.0, 'Each box represents one session', fontsize=6.5,
            ha='center', va='center', fontstyle='italic', color=text_gray)

    # Label below training boxes
    ax.text(training_center, -1.3,
            'Image change detection\ncriterion performance', fontsize=6.5,
            ha='center', va='top', color='#444444')

    # ---- Transition arrow ----
    arrow_start_x = training_end_x + 0.4
    arrow_end_x = arrow_start_x + 2.2

    ax.annotate('', xy=(arrow_end_x, 0), xytext=(arrow_start_x, 0),
                arrowprops=dict(arrowstyle='->', color='#666666',
                                lw=1.2, ls='--'),
                zorder=2)

    ax.text((arrow_start_x + arrow_end_x) / 2, -0.95,
            'Transition after reaching\ncriterion performance', fontsize=5.5,
            ha='center', va='top', color='#777777', fontstyle='italic')

    # ---- 2P imaging — dark boxes (Familiar imaging sessions) ----
    imaging_start_x = arrow_end_x + 0.4
    n_familiar_imaging = 4
    for i in range(n_familiar_imaging):
        x = imaging_start_x + i * (box_w + gap)
        rect = mpatches.FancyBboxPatch((x, box_y), box_w, box_h,
                                        boxstyle='round,pad=0.05',
                                        facecolor=training_color, edgecolor='none',
                                        zorder=2)
        ax.add_patch(rect)

    familiar_imaging_end = imaging_start_x + n_familiar_imaging * (box_w + gap) - gap + box_w

    # Header centered over dark imaging boxes
    imaging_center = (imaging_start_x + familiar_imaging_end) / 2
    ax.text(imaging_center, 2.7, '2P imaging', fontsize=10, fontweight='bold',
            ha='center', va='center')
    ax.text(imaging_center, 2.0, 'Omissions added', fontsize=6.5,
            ha='center', va='center', fontstyle='italic', color=text_gray)

    # Ellipsis dots after imaging boxes
    dot_start = familiar_imaging_end + 0.5
    for dx in [0, 0.45, 0.9]:
        ax.plot(dot_start + dx, 0, 'o', color=training_color, markersize=3.5, zorder=2)

    # ---- Selected sessions: F, N, N+ colored boxes ----
    selected_start_x = dot_start + 1.8

    # Familiar box
    rect_F = mpatches.FancyBboxPatch((selected_start_x, box_y), box_w, box_h,
                                      boxstyle='round,pad=0.05',
                                      facecolor=familiar_color, edgecolor='none',
                                      zorder=2)
    ax.add_patch(rect_F)
    ax.text(selected_start_x + box_w / 2, 0, 'F', fontsize=10,
            fontweight='bold', ha='center', va='center', color='white')

    # Novel box
    novel_x = selected_start_x + box_w + gap
    rect_N = mpatches.FancyBboxPatch((novel_x, box_y), box_w, box_h,
                                      boxstyle='round,pad=0.05',
                                      facecolor=novel_color, edgecolor='none',
                                      zorder=2)
    ax.add_patch(rect_N)
    ax.text(novel_x + box_w / 2, 0, 'N', fontsize=10,
            fontweight='bold', ha='center', va='center', color='white')

    # Novel+ box
    novel_plus_x = novel_x + box_w + gap
    rect_Np = mpatches.FancyBboxPatch((novel_plus_x, box_y), box_w, box_h,
                                       boxstyle='round,pad=0.05',
                                       facecolor=novel_plus_color, edgecolor='none',
                                       zorder=2)
    ax.add_patch(rect_Np)
    ax.text(novel_plus_x + box_w / 2, 0, 'N+', fontsize=9,
            fontweight='bold', ha='center', va='center', color='white')

    # Labels below colored boxes
    label_y = -1.1
    ax.text(selected_start_x + box_w / 2, label_y, 'Familiar',
            fontsize=7.5, fontweight='bold', ha='center', va='top',
            color=familiar_color)
    ax.text(novel_x + box_w / 2, label_y, 'Novel',
            fontsize=7.5, fontweight='bold', ha='center', va='top',
            color=novel_color)
    ax.text(novel_plus_x + box_w / 2, label_y, 'Novel+',
            fontsize=7.5, fontweight='bold', ha='center', va='top',
            color=novel_plus_color)

    # Down arrow indicating selected sessions
    mid_selected = (selected_start_x + novel_plus_x + box_w) / 2
    ax.annotate('', xy=(mid_selected, box_h / 2 + 0.15),
                xytext=(mid_selected, 1.6),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.0),
                zorder=2)

    if save_dir is not None:
        figsize = fig.get_size_inches()
        utils.save_figure(fig, figsize, save_dir, folder,
                          'figure_2D_schematic' + suffix, formats=['.pdf', '.png'])

    return ax


# Generate Figure 2D schematic as standalone panel
plot_figure_2D_schematic(save_dir=save_dir, folder='schematics', suffix='')


# ============================================================================
# COMPOSITE FIGURE 2: All panels on a single figure axis
# ============================================================================
# Layout based on published Figure 2 (Change detection task with familiar and novel images):
#   Left column:  A (change schematic, top) | B (omission schematic, mid) | C (behavior traces, bottom)
#   Middle column: D (training timeline, top) | E (lick rasters F/N/N+, bottom ~60%)
#   Right column:  F (d-prime, top ~40%) | G (running speed, mid) | H (pupil width, bottom)
#
# Panels A, B are schematics — left as placeholders.
# Panels C-H are data-driven and plotted directly.

def plot_figure_2_composite(behavior_stats_sdk, platform_experiments, palette,
                            running_speed_mdf_changes, pupil_mdf, lick_rate_mdf,
                            familiar_dataset, novel_dataset, novel_plus_dataset,
                            colors, image_pre=None, image_post=None,
                            save_dir=None, folder='composite_figures'):
    """
    Generate a composite Figure 2 with all panels arranged in the published layout.
    Saved as PDF for Illustrator editing.

    Parameters
    ----------
    image_pre : 2D array, optional
        Repeated stimulus image for schematics (e.g., im061 unwarped).
    image_post : 2D array, optional
        Change stimulus image for Panel A schematic (e.g., im065 unwarped).
    """
    figsize = (24, 20)
    fig = plt.figure(figsize=figsize, facecolor='white')

    # ---- Left column ----
    # Panel A: Image change schematic (programmatic)
    ax_A = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 0.28], yspan=[0, 0.18])
    plot_figure_2A_schematic(image_pre=image_pre, image_post=image_post, ax=ax_A)
    ax_A.set_title('A', loc='left', fontweight='bold', fontsize=16)

    # Panel B: Omission schematic (programmatic)
    ax_B = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 0.28], yspan=[0.20, 0.38])
    plot_figure_2B_schematic(image=image_pre, ax=ax_B)
    ax_B.set_title('B', loc='left', fontweight='bold', fontsize=16)

    # Panel C: Behavior traces (running, pupil, licking during session)
    ax_C = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0, 0.28], yspan=[0.42, 0.65])
    ax_C.text(0.5, 0.5, 'C — Behavior traces\n(place manually)', transform=ax_C.transAxes,
              ha='center', va='center', fontsize=10, color='gray')
    ax_C.set_title('C', loc='left', fontweight='bold', fontsize=16)
    ax_C.axis('off')

    # ---- Middle column ----
    # Panel D: Training timeline schematic (programmatic)
    ax_D = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.32, 0.65], yspan=[0, 0.18])
    plot_figure_2D_schematic(ax=ax_D)
    ax_D.set_title('D', loc='left', fontweight='bold', fontsize=16)

    # Panel E: Lick rasters for Familiar / Novel / Novel+
    ax_E = utils.placeAxesOnGrid(fig, dim=[1, 3], xspan=[0.32, 0.65], yspan=[0.22, 0.65],
                                  wspace=0.3, sharey=True)
    trials_f = familiar_dataset.trials.copy()
    trials_f = trials_f[trials_f.go == True][:100]
    ppf.plot_lick_raster_for_trials(trials_f, title='Familiar', save_dir=None, suffix=None, ax=ax_E[0])
    ax_E[0].set_title('E  Familiar', loc='left', fontweight='bold', fontsize=13, color=colors[0])

    trials_n = novel_dataset.trials.copy()
    trials_n = trials_n[trials_n.go == True][:100]
    ppf.plot_lick_raster_for_trials(trials_n, title='Novel', save_dir=None, suffix=None, ax=ax_E[1])
    ax_E[1].set_title('Novel', color=colors[1], fontsize=13)
    ax_E[1].set_ylabel('')

    trials_np = novel_plus_dataset.trials.copy()
    trials_np = trials_np[trials_np.go == True][:100]
    ppf.plot_lick_raster_for_trials(trials_np, title='Novel +', save_dir=None, suffix=None, ax=ax_E[2])
    ax_E[2].set_title('Novel +', color=colors[2], fontsize=13)
    ax_E[2].set_ylabel('')

    # ---- Right column ----
    # Panel F: d-prime by experience level
    ax_F = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.70, 1.0], yspan=[0, 0.28])
    metric = 'mean_dprime_engaged'
    stats_F = behavior_stats_sdk.copy()
    stats_F = stats_F[stats_F[metric] > 0]
    stats_F = stats_F[stats_F.behavior_session_id.isin(platform_experiments.behavior_session_id.unique())]
    ppf.plot_behavior_metric_by_experience(stats_F, metric, title='',
                                           ylabel='d-prime', ylims=[-0.1, 3.3],
                                           best_image=False, show_mice=False, plot_stats=True,
                                           stripplot=True, show_ns=False,
                                           save_dir=None, folder=None, suffix=None, ax=ax_F)
    ax_F.set_title('F', loc='left', fontweight='bold', fontsize=16)

    # Panel G: Running speed population averages
    ax_G = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.70, 1.0], yspan=[0.32, 0.52])
    df_G = running_speed_mdf_changes.copy()
    df_G['expts'] = 'all'
    ppf.plot_population_averages_for_conditions(df_G, 'running_speed', 'changes',
                                                 'expts', 'experience_level', horizontal=True,
                                                 ylabel='Running speed (cm/s)',
                                                 xlim_seconds=[-1, 1.5], interval_sec=1,
                                                 xlabel='Time from image change (sec)',
                                                 palette=palette, ax=ax_G, title=None,
                                                 suptitle=None, save_dir=None, folder=None, suffix=None)
    ax_G.set_title('G', loc='left', fontweight='bold', fontsize=16)

    # Panel H: Pupil width population averages
    ax_H = utils.placeAxesOnGrid(fig, dim=[1, 1], xspan=[0.70, 1.0], yspan=[0.56, 0.76])
    df_H = pupil_mdf.copy()
    df_H['mean_trace'] = [mean_trace - 1 for mean_trace in df_H.mean_trace.values]
    df_H['expts'] = 'all'
    ppf.plot_population_averages_for_conditions(df_H, 'pupil_width', 'changes',
                                                 'expts', 'experience_level', horizontal=True,
                                                 xlim_seconds=[-1, 1.5], interval_sec=1,
                                                 palette=palette, ax=ax_H, title=None,
                                                 suptitle=None, ylabel='Pupil width (norm.)',
                                                 xlabel='Time from image change (sec)',
                                                 save_dir=None, folder=None, suffix=None)
    ax_H.set_title('H', loc='left', fontweight='bold', fontsize=16)

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'figure_2_composite', formats=['.pdf'])
    return fig


# Generate composite figure
colors = utils.get_experience_level_colors()
fig2 = plot_figure_2_composite(
    behavior_stats_sdk, platform_experiments, palette,
    running_speed_mdf_changes, pupil_mdf, lick_rate_mdf,
    familiar_dataset, novel_dataset, novel_plus_dataset,
    colors=colors, image_pre=image_pre, image_post=image_post,
    save_dir=save_dir, folder='composite_figures'
)
