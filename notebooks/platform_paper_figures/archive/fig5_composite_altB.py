# Run from figure_5_main.ipynb in place of the alternative composite cell:  %run -i fig5_composite_altB.py
# (-i runs it in the notebook namespace: needs the same setup cells as the main composite + data_dir, n_clusters)

# Composite Figure 5 main -- ALTERNATIVE: UMAP replaced by co-clustering matrix (B) + cluster dendrogram (C);
# the old C-G become D-H. Otherwise identical to the composite cell above except B/C are placed explicitly
# (so they are not snugged) and the output filename. Also needs data_dir and n_clusters (data-loading cells).
# Set B_EXP_STYLE in the B/C block to 'dots' or 'lines'; each style saves to its own file.
#   A  single-cell coding example (3 ROI masks F/N/N+  +  coding-score barplot)
#   B  co-clustering probability matrix
#   C  dendrogram of clusters (cluster-mean coding scores); IDs colored by preferred feature,
#      preferred experience as leaf dots or colored branches
#   C  cluster coding-score heatmap (one row per cluster)
#   D  per-cluster population CHANGE responses   (clusters x F/N/N+)
#   E  per-cluster population OMISSION responses (clusters x F/N/N+)
#   F  % cells per cluster per cell type (Exc/Sst/Vip), colored by dominant feature
#   G  example clusters per cell type with proportion pies
#
# figsize + REGIONS were derived from the Figure 5 panel geometry and
# tuned against a data-free mock until check_label_overlap() was clean. The panel
# functions are embedded LIVE (no saved PNGs): C/D/E/F take an axes list, G takes
# fig+bbox, A is assembled from sf.plot_cell_zoom + pse.plot_coding_scores_for_cell.
#
# UPSTREAM (run the data-loading cells above first so these stable vars exist):
#   cluster_meta, feature_matrix, clustering_dir, image_mdf, change_mdf,
#   omission_mdf, results_pivoted, platform_cells_table, platform_experiments.
# Panels B and F are self-contained (they recompute their inputs) so they do NOT
# depend on the exploratory `umap_df`/`colors`/`tmp` that later cells overwrite.
import os
import numpy as np
import visual_behavior.visualization.ophys.summary_figures as sf

figsize = (12.3, 20.0)
fig = plt.figure(figsize=figsize)
fig_setup(fig)

# REGIONS [x0, x1, y0, y1] in figure fraction, y TOP-DOWN.
# Derived from the measured panel geometry (letter-anchor -> content, normalized into
# the panels' bounding box), then gaps tuned against a data-free mock until
# check_label_overlap was clean. Layout: A,B top | C,D,E middle |
# F bottom-left (tall, % cells) | G bottom-right (3x4 example clusters).
REGIONS = {
    'A': [0.040, 0.436, 0.017, 0.199],   # top-left  : single-cell coding example (plots nudged up)
    'B': [0.390, 0.990, 0.015, 0.198],   # top-right : UMAP (moved left + wider)
    'D': [0.001, 0.370, 0.224, 0.624],   # mid-left  : cluster coding heatmap (thinner)
    'E': [0.390, 0.680, 0.224, 0.624],   # mid       : change traces (aligned top+bottom with E)
    'F': [0.700, 1.000, 0.224, 0.624],   # mid-right : omission traces (wider)
    'G': [0.000, 0.310, 0.655, 0.920],   # BOTTOM-LEFT (moved up a little)
    'H': [0.320, 0.995, 0.675, 0.925],   # BOTTOM-RIGHT (bottom raised; top unchanged)
}
F_GUTTER = (20, 28)   # smaller letter gutter for F so the plot can sit further left

# Per-panel LEFT decoration setback (pt). Smaller than the 80pt default pulls each
# panel's y-label up against its letter gutter (no empty gap). Tuned in the mock so
# no label clips into the gutter -- if a panel ends up with too much/little left
# margin on the real render, nudge its value here. A's ROI row needs almost none;
# A's coding-score bar legitimately needs ~80 (2-line "Coding score" + ticks).
# setbacks bumped vs the mock because the real render's labels are wider than the mock's.
DP = {'A': (16, 0), 'B': (64, 0), 'D': (58, 0), 'E': (58, 0), 'F': (58, 0), 'G': (44, 0)}
DP_A_BAR = (80, 0)

cluster_order = np.sort(cluster_meta.cluster_id.unique())
n_cl = len(cluster_order)

# ---- A: 3 ROI masks (top sub-row) + 1 full-width coding-score bar (bottom) ----
A_CELL = 1086620671
COLOR_CLUSTER_IDS_DEF = True   # color the cluster-ID labels in D-F by preferred feature (as in C)
EF_TITLE_DX = -0.005           # shift the E/F section titles left (figure fraction) from centered
CELL_GROUP_TERM = 'cell type'  # word used in the G/H titles, e.g. 'cell type' or 'subclass'
A_ROI_TOP = 0.21      # top of the ROI-image row within A (fraction of A's height); higher = lower + smaller images
TOP_LETTER_DROP = 0.01   # move the top-row panel letters (A, B, C) down by this figure fraction
Ax0, Ax1, Ay0, Ay1 = REGIONS['A']
A_top = [Ax0 + 0.03, Ax1 - 0.08, Ay0 + A_ROI_TOP * (Ay1 - Ay0), Ay0 + 0.58 * (Ay1 - Ay0)]   # ROI row: narrower + nudged right
# (A_ROI_TOP raised from 0.13 so the ROI row sits lower -- smaller images, room for A's title, which is
#  top-aligned with the B/C titles below)
A_bot = [Ax0 - 0.025, Ax1 - 0.05, Ay0 + 0.63 * (Ay1 - Ay0), Ay1]         # coding-score bar: left + narrower (right extent reduced)
# A/C/F left-spines are aligned to F's spine: gutter+decoration chosen so each panel's
# inner-axes left edge lands in the same 0.01 cell that F does (verified spine x0 == 0.08).
# 'A' letter at the top-left of the full A region (kept up top so it lines up with B's letter)
_axA_letter = panel(fig, [REGIONS['D'][0], Ax1], [REGIONS['B'][2], Ay1], 'A', dim=(1, 1)); _axA_letter.axis('off')  # letter aligned with B
axA_roi = np.atleast_1d(panel(fig, A_top[:2], A_top[2:], None, dim=(1, 3), wspace=0.25,
                              gutter_pt=(10, 28), decoration_pt=(4, 0)))
axA_bar = panel(fig, A_bot[:2], A_bot[2:], None, dim=(1, 1),
                gutter_pt=(0, 0), decoration_pt=(54, 0))

cell_meta_A = platform_cells_table[platform_cells_table.cell_specimen_id == A_CELL]
cell_drop_A = results_pivoted[results_pivoted.cell_specimen_id == A_CELL][
    processing.get_features_for_clustering() + ['experience_level']]
container_A = cell_meta_A.ophys_container_id.unique()[0]
exp_levels_A = ['Familiar', 'Novel', 'Novel +']
exp_colors_A = utils.get_experience_level_colors()
_dsets = globals().get('_fig5_A_datasets', {})   # cache so re-runs don't reload
for e, lvl in enumerate(exp_levels_A):
    oeid = platform_experiments[(platform_experiments.ophys_container_id == container_A) &
                                (platform_experiments.experience_level == lvl)].index.values[0]
    if oeid not in _dsets:
        print('loading dataset for', lvl, oeid, '...')
        _dsets[oeid] = loading.get_ophys_dataset(oeid, get_extended_stimulus_presentations=False)
    ds = _dsets[oeid]
    a = axA_roi[e]
    try:
        roi_id = ds.cell_specimen_table.loc[A_CELL].cell_roi_id
        sf.plot_cell_zoom(ds.roi_masks, ds.max_projection, roi_id,
                          spacex=50, spacey=50, show_mask=False, ax=a)
    except Exception:
        a.axis('off')
    a.set_title(lvl, color=exp_colors_A[e], fontsize=15)
globals()['_fig5_A_datasets'] = _dsets
pse.plot_coding_scores_for_cell(cell_drop_A, ax=axA_bar)
axA_bar.tick_params(axis='both', labelsize=15)                   # y-ticks up, x-ticks down, matched
_acols = [t.get_color() for t in axA_bar.get_xticklabels()]      # capitalize x labels, keep colors
axA_bar.set_xticklabels([t.get_text().capitalize() for t in axA_bar.get_xticklabels()])
for _t, _c in zip(axA_bar.get_xticklabels(), _acols):
    _t.set_color(_c); _t.set_fontsize(14)   # x-tick labels (set_xticklabels above resets the size)
_lh, _ll = axA_bar.get_legend_handles_labels()   # move the F/N/N+ legend inside, upper-right
if _lh:
    axA_bar.legend(_lh, _ll, loc='upper right', fontsize=11, frameon=False)
axA_bar.yaxis.label.set_fontsize(15)             # 'Coding score' y-label

# ---- B + C (alternative to the UMAP): B = co-clustering matrix, C = cluster dendrogram ----
# Left : co-clustering probability (100 spectral-clustering repeats), cells sorted by cluster ID.
# Right: dendrogram of clusters from cluster-mean coding scores (Euclidean, average linkage), horizontal.
# Cluster-ID labels are colored by dominant ("preferred") feature, computed with the same function panel F
# uses (so B and F agree). Dendrogram lines are colored by preferred experience, defined as FAMILIAR vs
# NOVEL-TYPE sessions: for each cluster's dominant feature, compare Familiar with the MEAN of Novel and
# Novel + (the two novel-image sessions):
#     novelty index = (mean(Novel, Novel +) - Familiar) / (mean(Novel, Novel +) + Familiar)
#     index >= +EXP_SEL_MIN  -> whichever of Novel / Novel + is larger
#     index <= -EXP_SEL_MIN  -> Familiar
#     otherwise, or peak coding < EXP_SCORE_MIN -> black (no clear preference)
# A merged branch is colored only if every cluster under it shares the same experience color.
# Self-contained: uses only stable vars.
from scipy.cluster.hierarchy import linkage, dendrogram
from matplotlib.lines import Line2D
EXP_SEL_MIN = 0.15      # 0.15 = stronger side >= ~1.35x the weaker; raise to color fewer clusters
EXP_SCORE_MIN = 0.1     # clusters with weaker peak coding (e.g. non-coding cluster 1) stay black
B_NUDGE_IDS = [7, 9]    # matrix y-labels shifted left so thin neighbouring clusters don't overlap
B_EXP_STYLE = 'dots'    # how preferred experience is shown on the dendrogram:
                        #   'dots'  = one dot per leaf (open circle = no preference), all branches black
                        #   'lines' = colored leaf stems / branches

_cmB = processing.get_coding_score_metrics_for_clusters(cluster_meta, results_pivoted)
_feat = plotting.get_feature_colors_with_gray()          # [gray] + features, same as F
_featdict = {c: _feat[1 + processing.get_features_for_clustering().index(_cmB.loc[c, 'dominant_feature'])]
             for c in _cmB.index}
_exp_levels, _exp_cols = utils.get_experience_levels(), utils.get_experience_level_colors()
_FAM, _NOV, _NOVP = _exp_levels                          # 'Familiar', 'Novel', 'Novel +'

# per-cluster mean coding score of the dominant feature in each session (same averaging as _cmB)
_rpB = (results_pivoted if 'cluster_id' in results_pivoted.columns
        else results_pivoted.merge(cluster_meta[['cluster_id']], on='cell_specimen_id'))
_lvl_means = _rpB.groupby(['cluster_id', 'experience_level'])[processing.get_features_for_clustering()].mean()
_rowsB = []
for _c in _cmB.index:
    _v = _lvl_means.loc[_c][_cmB.loc[_c, 'dominant_feature']]
    _novlvl = _NOV if _v[_NOV] >= _v[_NOVP] else _NOVP
    _fam, _nov = _v[_FAM], (_v[_NOV] + _v[_NOVP]) / 2
    _idx = (_nov - _fam) / (_nov + _fam) if (_nov + _fam) > 0 else 0.0
    _pref = _novlvl if _idx >= EXP_SEL_MIN else (_FAM if _idx <= -EXP_SEL_MIN else None)
    if max(_fam, _v[_NOV], _v[_NOVP]) < EXP_SCORE_MIN:
        _pref = None
    _rowsB.append(dict(cluster_id=_c, dominant_feature=_cmB.loc[_c, 'dominant_feature'], Familiar=_fam,
                       Novel=_v[_NOV], **{'Novel +': _v[_NOVP]}, novelty_index=_idx, preferred_experience=_pref))
_prefB = pd.DataFrame(_rowsB).set_index('cluster_id').sort_index()
_expdict = {c: (_exp_cols[_exp_levels.index(p)] if isinstance(p, str) else None)   # None -> NaN in pandas
            for c, p in _prefB.preferred_experience.items()}
print('Panel B dendrogram colors (coding score of the dominant feature per session):')
print(_prefB.round(2).to_string())
# non-coding clusters (peak coding < EXP_SCORE_MIN, i.e. cluster 1): gray ID label, as in panel F
_noncodingB = _prefB.index[_prefB[[_FAM, _NOV, _NOVP]].max(axis=1) < EXP_SCORE_MIN].tolist()
for _c in _noncodingB:
    _featdict[_c] = _feat[0]
print('non-coding clusters (gray):', _noncodingB)

_metaB = cluster_meta.sort_values('cluster_id', kind='stable')
_idsB, _clB = _metaB.index.values, _metaB.cluster_id.values
_NB = len(_idsB)
_cocl = pd.read_hdf(os.path.join(data_dir, f'coclustering_matrix_n_{n_clusters}_clusters.h5'),
                    key='df').loc[_idsB, _idsB].values
_centers = [np.flatnonzero(_clB == c).mean() for c in cluster_order]
_bounds = np.flatnonzero(np.diff(_clB)) + 0.5

# Layout: B (co-clustering matrix) and C (dendrogram + legends) share the top row. The matrix is square and as
# large as the row allows: its bottom lines up with the bottom of A's coding-score bar plot, its top with the
# top of the plotting area, unless C needs the width (then it shrinks). C starts right after the colorbar; the
# tops of B's and C's axes are aligned. panel() reserves the letters/gutters and gives the top edge; the axes
# are then placed explicitly. All widths are figure fractions -- nudge these to rebalance B / C.
B_CBAR_SPACE = 0.058    # matrix right edge -> C's left edge (colorbar + its label live here)
C_INSET = 0.035         # C's left edge (letter) -> dendrogram left edge
B_DENDRO_W = 0.065      # dendrogram width
B_LABEL_SPACE = 0.03    # dendrogram right edge -> legends (cluster-ID labels live here)
B_LEGEND_W = 0.107      # legend column (fits "No preference")
B_LEG_SHIFT = 0.1       # legends' left edge within the legend column (fraction of B_LEGEND_W); raise to move right
B_TITLE_ROOM_PT = 14    # room above the plots for the B/C titles; lower = taller (bigger) matrix
B_LEG_GAP = 0.10        # vertical gap between the two legends (fraction of the legend column height)
B_LEG_TOP = 0.9         # top of the first legend (fraction of the plot height; 1 = top of the dendrogram)
# Tick labels: all of B's tick labels (both panels + colorbar) use one size = B's matrix y-tick size (10 pt),
# raised to the journal minimum if 10 pt would print smaller. Neuron/Cell Press: figure text 6-8 pt at final
# size; this figure is assumed to print at double-column width (174 mm). If it will print narrower (e.g. scaled
# down to fit the page height), set FINAL_WIDTH_MM to the actual printed width and the minimum rises to match.
MIN_FONT_PT_FINAL = 6
FINAL_WIDTH_MM = 174
_min_fs = MIN_FONT_PT_FINAL * fig.get_size_inches()[0] / (FINAL_WIDTH_MM / 25.4)   # = 6 pt after scaling
B_TICK_FS = max(10, np.ceil(_min_fs * 2) / 2)                                       # round up to 0.5 pt
print(f'Panel B tick labels: {B_TICK_FS} pt in the figure = {B_TICK_FS * FINAL_WIDTH_MM / 25.4 / fig.get_size_inches()[0]:.1f} pt printed')
axB = np.atleast_1d(panel(fig, REGIONS['B'][:2], REGIONS['B'][2:], 'B', has_title=True,
                          dim=(1, 3), title_room_pt=B_TITLE_ROOM_PT, gutter_pt=(20, 28), decoration_pt=(46, 0)))
_figW, _figH = fig.get_size_inches()
_yb = axA_bar.get_position().y0                       # bottom of A's bar plot (figure fraction, bottom-up)
_yt = axB[0].get_position().y1                        # top from panel() (below the title room)
_xB0 = axB[0].get_position().x0
_C_W = C_INSET + B_DENDRO_W + B_LABEL_SPACE + B_LEGEND_W
_w_full = (_yt - _yb) * _figH / _figW                 # square matrix filling the full height
_w_room = REGIONS['B'][1] - _xB0 - B_CBAR_SPACE - _C_W   # width left once C fits
_wB = min(_w_full, _w_room)
print(f'Panel B matrix: {_wB * _figW:.2f} in square' +
      ('' if _wB == _w_full else f' (width-limited; full height would be {_w_full * _figW:.2f} in -- narrow C to grow it)'))
_hB = _wB * _figW / _figH
_ytop = _yb + _hB                                     # tops of B and C aligned
axB[0].set_position([_xB0, _yb, _wB, _hB])
_xC0 = _xB0 + _wB + B_CBAR_SPACE                       # C starts right after the colorbar
_xD0 = _xC0 + C_INSET
axB[1].set_position([_xD0, _ytop - _hB, B_DENDRO_W, _hB])
_xL0 = _xD0 + B_DENDRO_W + B_LABEL_SPACE
axB[2].set_position([_xL0, _ytop - _hB, B_LEGEND_W, _hB])
axB[2].axis('off')                                    # legend column
# C letter: at C's left edge, on the same line as B's letter
fig.delaxes(panel(fig, [_xC0, REGIONS['B'][1]], REGIONS['B'][2:], 'C', dim=(1, 1)))
# dendrogram + legends belong to panel C (for check_label_overlap); split the regions at C's edge
for _a in axB[1:]:
    _a._panel_grid_owner = 'C'
REGIONS['C'] = [_xC0, REGIONS['B'][1], REGIONS['B'][2], REGIONS['B'][3]]
REGIONS['B'] = [REGIONS['B'][0], _xC0, REGIONS['B'][2], REGIONS['B'][3]]

# -- left: co-clustering matrix (square) --
_imB = axB[0].imshow(_cocl, cmap='Greys', vmin=0, vmax=1, aspect='auto',
                     interpolation='antialiased', rasterized=True)
for _b in _bounds:
    axB[0].axhline(_b, color='0.6', lw=0.4); axB[0].axvline(_b, color='0.6', lw=0.4)
axB[0].set_yticks(_centers); axB[0].set_yticklabels(cluster_order)
axB[0].set_xticks([-0.5, _NB - 0.5]); axB[0].set_xticklabels(['0', str(_NB)])
axB[0].set_ylabel('Cells sorted by cluster ID'); axB[0].set_xlabel('Cells', labelpad=-8)
axB[0].set_title('Co-clustering probability')
for _sp in axB[0].spines.values():                      # box on all four sides
    _sp.set_visible(True); _sp.set_color('k'); _sp.set_linewidth(1)
_cax = axB[0].inset_axes([1.03, 0.0, 0.04, 0.35])     # colorbar to the right, bottom-aligned
_cb = plt.colorbar(_imB, cax=_cax, ticks=[0, 1])
_cb.set_label('Probability', fontsize=12, labelpad=2)

# -- right: dendrogram, distance on x, cluster IDs on the right --
_cent = feature_matrix.loc[_idsB].groupby(_clB).mean()
_labsB = list(_cent.index)
_Z = linkage(_cent.values, method='average', metric='euclidean')
_n = len(_labsB)
_leaves_under = {i: {_labsB[i]} for i in range(_n)}
for _k, (_a, _b) in enumerate(_Z[:, :2].astype(int)):
    _leaves_under[_n + _k] = _leaves_under[_a] | _leaves_under[_b]


def _link_color(k):      # dendrogram() needs a color STRING, so convert the RGB tuple
    from matplotlib.colors import to_hex
    cols = {_expdict.get(c) for c in _leaves_under[k]}
    return to_hex(cols.pop()) if (len(cols) == 1 and None not in cols) else 'k'


_d = dendrogram(_Z, labels=_labsB, ax=axB[1], orientation='left',
                link_color_func=(lambda k: _link_color(k)) if B_EXP_STYLE == 'lines' else (lambda k: 'k'))
_parent_h = {int(c): _Z[r, 2] for r in range(len(_Z)) for c in _Z[r, :2]}
for _j, _leaf in enumerate(_d['leaves']):
    _col = _expdict.get(_labsB[_leaf])
    if B_EXP_STYLE == 'lines':
        # color each cluster's own stem by its experience level (covers clusters that join a mixed branch)
        if _col is not None:
            axB[1].plot([0, _parent_h[_leaf]], [5 + 10 * _j] * 2, color=_col, lw=2, zorder=3, solid_capstyle='butt')
    else:
        # one dot at the leaf tip; open black circle = no preference
        axB[1].plot(0, 5 + 10 * _j, 'o', ms=8, mew=1.5, clip_on=False, zorder=4,
                    mfc=_col if _col is not None else 'white', mec=_col if _col is not None else 'k')
for _ln in axB[1].collections:                      # same width as the colored leaf stems
    _ln.set_linewidth(2)
axB[1].set_xlabel('Euclidean distance', loc='left')  # left-aligned so C's letter can sit close; no y-label needed
axB[1].set_title('Cluster similarity', loc='left', fontsize=16)   # left title isn't ax.title, so size set here
axB[1].set_xticks([0, 0.4, 0.8]); axB[1].set_xticklabels(['0', '0.4', '0.8'])
sns.despine(ax=axB[1], left=True, right=True)
axB[1].tick_params(axis='y', length=0, pad=10 if B_EXP_STYLE == 'dots' else 3.5)   # IDs clear of the dots

for _a in axB[:2]:
    _a.title.set_fontsize(16); _a.xaxis.label.set_fontsize(15); _a.yaxis.label.set_fontsize(15)   # match A / F
    _a.tick_params(labelsize=B_TICK_FS)
for _t in axB[1].get_yticklabels():                              # dendrogram IDs colored by dominant feature (matrix IDs stay black)
    _t.set_color(_featdict.get(int(_t.get_text()), _feat[0])); _t.set_fontweight('bold')
_cax.tick_params(labelsize=B_TICK_FS)
# thin clusters (8, 9) put matrix y-labels on top of each other: shift these IDs' labels left
for _tk, _c in zip(axB[0].yaxis.get_major_ticks(), cluster_order):
    if _c in B_NUDGE_IDS:
        _tk.set_pad(_tk.get_pad() + B_TICK_FS * 1.3)       # ~one label width

# legends stacked at the top of the legend column, the second directly under the first
_lgF = axB[2].legend([Line2D([], [], marker='s', ls='', ms=7, mfc=c, mec='none') for c in list(_feat[1:5]) + [_feat[0]]],
                     ['Images', 'Omissions', 'Task', 'Behavior', 'Non-coding'], title='Preferred\nfeature',
                     loc='upper left', bbox_to_anchor=(B_LEG_SHIFT, B_LEG_TOP), fontsize=11, title_fontsize=11,
                     frameon=False, handletextpad=0.1, labelspacing=0.12, borderpad=0, borderaxespad=0)
_lgF._legend_box.align = 'left'
axB[2].add_artist(_lgF)
fig.canvas.draw()
_lgF_bottom = axB[2].transAxes.inverted().transform(_lgF.get_window_extent())[0, 1]
_hE = ([Line2D([], [], color=c, lw=2) for c in list(_exp_cols[:3]) + ['k']] if B_EXP_STYLE == 'lines' else
       [Line2D([], [], marker='o', ls='', ms=8, mew=1.5, mfc=c, mec=c) for c in _exp_cols[:3]]
       + [Line2D([], [], marker='o', ls='', ms=8, mew=1.5, mfc='white', mec='k')])
_lgE = axB[2].legend(_hE,
                     list(_exp_levels) + ['No preference'],
                     title='Preferred\nexperience level', loc='upper left', bbox_to_anchor=(B_LEG_SHIFT, _lgF_bottom - B_LEG_GAP),
                     fontsize=11, title_fontsize=11, frameon=False, handlelength=1.2, handletextpad=0.3,
                     labelspacing=0.12, borderpad=0, borderaxespad=0)
_lgE._legend_box.align = 'left'

# ---- D: cluster coding-score heatmap (one row per cluster) ----
axC = list(np.atleast_1d(panel(fig, REGIONS['D'][:2], REGIONS['D'][2:], 'D',
                               dim=(n_cl, 1), hspace=0.0, has_title=True, title_room_pt=24,
                               gutter_pt=(20, 28), decoration_pt=(45, 0))))  # spine aligned to F; title room for headers
plotting.plot_coding_score_heatmap_matched(
    cluster_meta, feature_matrix, sort_by='cluster_id', cluster_order=cluster_order,
    session_colors=True, experience_index=None, title='',
    save_dir=None, folder=None, suffix='', ax=axC)
for _a in axC:                                   # smaller 'Cluster ID' y-label
    if _a.get_ylabel():
        _a.yaxis.label.set_fontsize(13)
    for _t in _a.texts:                          # rename the rotated 'Cluster ID' y-axis annotation
        if _t.get_text() == 'Cluster ID':
            _t.set_text('Cells sorted by cluster ID')
            _t.set_fontsize(16)                  # match panel C section title
            _t.set_va('center')                  # center on the plot (nudges it down)
            _t.set_y(_t.get_position()[1] + 0.15 * (_a.get_ylim()[1] - _a.get_ylim()[0]))  # nudge up a tiny bit

# ---- E: population CHANGE responses per cluster ----
axD_grid = np.array(panel(fig, REGIONS['E'][:2], REGIONS['E'][2:], 'E',
                          dim=(n_cl, 3), hspace=0.0, wspace=0.1, sharey='row', sharex=True,
                          has_title=True, title_room_pt=24, gutter_pt=(20, 28), decoration_pt=DP['E']))
axD_flat = [a for row in axD_grid for a in row]
plotting.plot_population_average_response_for_clusters_as_rows_split(
    change_mdf, 'changes', with_pre_change=True, cluster_order=cluster_order,
    ax=axD_flat, save_dir=None, cluster_id_fontsize=13, fill_alpha=0.25,
    cluster_id_xoffset=0.9, arrow_label_fontsize=12, apply_peak_threshold=True)

# ---- F: population OMISSION responses per cluster ----
axE_grid = np.array(panel(fig, REGIONS['F'][:2], REGIONS['F'][2:], 'F',
                          dim=(n_cl, 3), hspace=0.0, wspace=0.1, sharey='row', sharex=True,
                          has_title=True, title_room_pt=24, gutter_pt=(20, 28), decoration_pt=DP['F']))
axE_flat = [a for row in axE_grid for a in row]
plotting.plot_population_average_response_for_clusters_as_rows_split(
    omission_mdf, 'omissions', with_pre_change=True, cluster_order=cluster_order,
    ax=axE_flat, save_dir=None, cluster_id_fontsize=13, fill_alpha=0.25,
    cluster_id_xoffset=0.9, arrow_label_fontsize=12, apply_peak_threshold=True)

# D/E experience-level column labels (Familiar/Novel/Novel+): centered over each
# column (x=0.5), close to the plots (small pad), bigger (15pt)
for _grid in (axD_grid, axE_grid):
    for _a in _grid[0]:
        _t = _a.get_title()
        if _t:
            _a.set_title(_t, color=_a.title.get_color(), fontsize=15, x=0.5, pad=3)

# ---- G: % cells per cluster per cell type (colored by dominant feature) ----
# Self-contained: rebuild cluster_meta + per-cluster dominant_feature from stable
# inputs (cluster_meta, results_pivoted), not the notebook-state `tmp` (overwritten).
_cm_metrics = processing.get_coding_score_metrics_for_clusters(cluster_meta, results_pivoted)
_domF = _cm_metrics['dominant_feature']          # index = cluster_id
tmpF = cluster_meta.copy()
if 'cell_specimen_id' not in tmpF.columns:
    tmpF = tmpF.reset_index()
tmpF['dominant_feature_cluster'] = tmpF['cluster_id'].map(_domF)

axF = list(np.atleast_1d(panel(fig, REGIONS['G'][:2], REGIONS['G'][2:], 'G', has_title=True,
                               title_room_pt=34,   # extra top room so the section title sits above the Exc/Sst/Vip headers
                               dim=(1, 3), wspace=0.35, sharey=True,   # wider gaps so bar value labels show
                               gutter_pt=F_GUTTER, decoration_pt=DP['G'])))
plotting.plot_percent_cells_per_cluster_per_cre_dominant_feature(
    tmpF, col_to_group='cre_line', square=True, cluster_order=cluster_order,
    save_dir=None, ax=axF)
# sharey: only the leftmost column shows cluster-ID tick labels (axes already share y).
# y-tick + "Cluster ID" label keep their full autoscaled size; DP['G'] is sized so that
# (bigger) label sits snug against the letter gutter.
for _a in axF[1:]:
    _a.tick_params(labelleft=False)
for _a in axF:
    _a.tick_params(axis='x', labelsize=10)   # % Cells tick labels
axF[0].tick_params(axis='y', labelsize=14)   # cluster-ID ticks match panel C
axF[0].yaxis.label.set_fontsize(15)          # 'Cluster ID' y-label (matched with x-label)
for _a in axF:
    _a.xaxis.label.set_fontsize(15)          # '% Cells' x-label (matched with y-label)

# ---- H: example clusters per cell type (with proportion pies) ----
# reserve the letter + gutter with panel(), then embed the multi-axes grid via fig+bbox
axG_lbl = panel(fig, REGIONS['H'][:2], [0.655, REGIONS['H'][3]], 'H', dim=(1, 1))  # letter aligned with F
axG_lbl.axis('off')
# Insets tuned in the mock so the embedded grid's col-0 cell-type labels (drawn at
# ax.transAxes x=-0.55) and 2-line column headers clear the G letter gutter.
G_INSET_L, G_INSET_T = 0.105, 0.052   # left: grid near the G letter; top: lowered so the section title fits above
G_FONT = 1.4   # scale up all of panel G's fonts
G_bbox = [REGIONS['H'][0] + G_INSET_L, REGIONS['H'][2] + G_INSET_T,
          REGIONS['H'][1], REGIONS['H'][3]]
_, axG = plotting.plot_mean_cluster_response_profiles(
    image_mdf, change_mdf, omission_mdf, fig=fig, bbox=G_bbox, fontsize_scale=G_FONT,
    linewidth=1, fill_alpha=0.25, apply_peak_threshold=True)
# bump the per-axes fonts the function sets internally (titles, tick labels, the
# rotated cell-type row labels, and the pie % text live on the axes themselves)
for _a in np.array(axG, dtype=object).ravel():
    if _a.get_title():
        _a.title.set_fontsize(_a.title.get_fontsize() * G_FONT)
    _a.tick_params(labelsize=8 * G_FONT)
    for _t in _a.texts:
        _t.set_fontsize(_t.get_fontsize() * G_FONT)
for _a in np.array(axG, dtype=object).ravel():   # cell-type row labels 1 pt smaller
    for _t in _a.texts:
        if 'Excitatory' in _t.get_text() or 'Inhibitory' in _t.get_text():
            _t.set_fontsize(_t.get_fontsize() - 1)

# ---- snap inner-panel y-labels right up against their gutters (MEASURED, not guessed) ----
# B/D/E aren't leftmost, so their labels can't spill into the margin like A/C/F -- this
# rescales each about its right edge so the leftmost label lands at gutter+2pt. Pass each
# panel's own gutter (B uses 20pt, D/E use 32pt). G can't snap (its row labels would hit
# the letter); A/C/F are left aligned already, so leave them.
# (panels B and C are placed explicitly in their own block, so they are not snugged here)
snug_left_to_gutter(fig, REGIONS, only=['E', 'F'], gutter_pt=(20, 28))
# (snug_top intentionally NOT applied to C/D/E: leaving the has_title room in place drops each
# plot's TOP down a little -- bottoms unchanged -- so the section titles have clear room above the headers)
# snug_top_to_gutter(fig, REGIONS, only=['D', 'E', 'F'])

# nudge the D and E panel letters ever so slightly to the right
for _lax in fig.axes:
    if getattr(_lax, '_panel_grid_label_ax', False):
        _lp = _lax.get_position()
        if abs(_lp.x0 - REGIONS['E'][0]) < 0.008 or abs(_lp.x0 - REGIONS['F'][0]) < 0.008:
            _lax.set_position([_lp.x0 + 0.003, _lp.y0, _lp.width, _lp.height])

# ---- cluster-ID labels in D-F colored by preferred feature (same colors as C; cluster 1 gray) ----
if COLOR_CLUSTER_IDS_DEF:
    for _a in axC:                                             # D: each row's y-label is its cluster ID
        _lab = _a.get_ylabel()
        if _lab.isdigit() and int(_lab) in _featdict:
            _a.yaxis.label.set_color(_featdict[int(_lab)]); _a.yaxis.label.set_fontweight('bold')
    for _a in list(axD_grid.ravel()) + list(axE_grid.ravel()):  # E/F: cluster IDs are text annotations
        for _t in _a.texts:
            if _t.get_text().isdigit() and int(_t.get_text()) in _featdict:
                _t.set_color(_featdict[int(_t.get_text())]); _t.set_fontweight('bold')

# ---- panel section titles (bold headers on the letter line; nudge fontsize/dy_pt if needed) ----
_tA = panel_title(fig, (REGIONS['A'][0] - 0.04, REGIONS['A'][1]),
                  'Input to clustering: single cell\ncoding scores across experience levels',
                  y_frac=0.038, fontsize=16)
fig.canvas.draw()                                 # top-align A's title with B's title
_tA.set_va('top')
_tA.set_y(fig.transFigure.inverted().transform(axB[0].title.get_window_extent())[1, 1])
panel_title(fig, (0.08, REGIONS['D'][1]), 'Clustered coding scores',
            y_frac=0.245, fontsize=16)
# separate titles for D (image) and E (omission), each on its own panel's letter line
_axE_all = [a for a in axD_grid.ravel() if a.get_visible()]   # E/F titles centered over their plots
panel_title(fig, (min(a.get_position().x0 for a in _axE_all) + EF_TITLE_DX, max(a.get_position().x1 for a in _axE_all) + EF_TITLE_DX),
            'Average change responses', y_frac=0.245, fontsize=16)
_axF_all = [a for a in axE_grid.ravel() if a.get_visible()]
panel_title(fig, (min(a.get_position().x0 for a in _axF_all) + EF_TITLE_DX, max(a.get_position().x1 for a in _axF_all) + EF_TITLE_DX),
            'Average omission responses', y_frac=0.245, fontsize=16)
panel_title(fig, (0.050, REGIONS['G'][1]), f'Cluster sizes within each {CELL_GROUP_TERM}',
            y_frac=0.676, fontsize=16)
panel_title(fig, (0.390, REGIONS['H'][1]),
            f'Experience dependent responses for main clusters within each {CELL_GROUP_TERM}',
            y_frac=0.676, fontsize=16)

# move the top-row panel letters (A, B, C) down a little (less whitespace above the titles)
for _lax in fig.axes:
    if getattr(_lax, '_panel_grid_label_ax', False) and _lax.get_position().y1 > 1 - REGIONS['B'][3] / 2:
        _lp = _lax.get_position()
        _lax.set_position([_lp.x0, _lp.y0 - TOP_LETTER_DROP, _lp.width, _lp.height])

# ---- audit + save ----
# NOTE: A, C, F will each report a "left gutter" intrusion -- this is EXPECTED and intended:
# their spines are aligned to F (far left), so the rotated y-labels sit in the gutter strip,
# beside (not overlapping) the panel letters, which are at the top. Any OTHER warning is real.
check_label_overlap(fig, REGIONS, gutter_pt=F_GUTTER)
utils.save_figure(fig, figsize, save_dir, 'composite', 'figure_5_main_composite', formats=['.png', '.pdf'])
