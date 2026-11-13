"""
Cluster-colored max-projection tiles for cover art.

For every clustered cell: crop a fixed-size window of the max projection around its ROI,
invert it (bright soma -> dark), and map it through a white -> cluster-color colormap so the
cell appears in its cluster color on a white background.

Run from cover_art.ipynb (same folder), which loads cluster_meta and matched_cells_table.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, gaussian_filter
from matplotlib.colors import LinearSegmentedColormap

import visual_behavior.data_access.loading as loading
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache

_SDK_CACHE = None
MANIFEST = 'visual-behavior-ophys_project_manifest_v1.1.0.json'


def get_sdk_cache():
    """
    SDK cache on loading.get_sdk_cache_dir() (.../FinalDataAssetsforCodeOcean/visual_behavior_sdk_cache),
    created once and reused. Files already in the cache are read from disk; experiments missing from it
    are downloaded into it. Pinned to the v1.1.0 manifest so a newer release is never pulled in.
    (loading.get_ophys_dataset builds a new cache on every call, re-contacting S3 for each experiment.)
    """
    global _SDK_CACHE
    if _SDK_CACHE is None:
        _SDK_CACHE = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=loading.get_sdk_cache_dir())
        _SDK_CACHE.load_manifest(MANIFEST)
    return _SDK_CACHE

# Same cluster -> label -> color scheme as the graphical abstract (graphical_abstract.ipynb)
CLUSTER_LABELS = {1: 'non-coding / unselective', 2: 'novelty enhanced', 3: 'novel image coding',
                  4: 'familiar image coding', 5: 'novel image coding', 6: 'novelty suppressed',
                  7: 'non-coding / unselective', 8: 'familiar omission coding', 9: 'novel+ omission coding',
                  10: 'familiar task coding', 11: 'novel task coding', 12: 'behavior coding'}
LABEL_COLORS = {'novelty enhanced': '#aa1016', 'novelty suppressed': '#55a868',
                'novel image coding': '#572c92', 'familiar image coding': '#0b559f',
                'novel+ omission coding': '#8d89c0', 'familiar omission coding': '#539ecd',
                'novel task coding': '#e3bbcd', 'familiar task coding': '#ccb974',
                'behavior coding': '#dd8452', 'non-coding / unselective': '#BFBFBF'}


def cluster_cmap(cluster_id):
    """Colormap from cluster color (value 0) to white (value 1)."""
    color = LABEL_COLORS[CLUSTER_LABELS[cluster_id]]
    return LinearSegmentedColormap.from_list(f'cluster_{cluster_id}', [color, 'white'])


def crop_around_roi(max_projection, roi_mask, half_width=20):
    """Fixed-size square crop centered on the ROI centroid; edges padded with the image minimum."""
    y, x = np.where(roi_mask)
    cy, cx = int(round(y.mean())), int(round(x.mean()))
    padded = np.pad(max_projection, half_width, mode='constant', constant_values=max_projection.min())
    cy, cx = cy + half_width, cx + half_width
    return padded[cy - half_width:cy + half_width + 1, cx - half_width:cx + half_width + 1]


def colorize(crop, cluster_id, roi_crop, other_crop, low_pct=50, high_pct=95, gamma=1.0, fade_sigma=1.5,
             keep_crop=None, keep_sigma=2):
    """
    Normalize a crop to its target cell, fade out neighboring ROIs, invert, and color it.
    Returns an RGBA array (float, 0-1).

    Normalization is anchored on the target cell, so it reaches full color even when neighbors are brighter:
      white      = low_pct percentile of pixels outside all ROIs (background / neuropil)
      full color = high_pct percentile of the target ROI's pixels
    other_crop (dilated masks of neighboring ROIs) is faded to white with a soft edge (fade_sigma, px).
    The target's own processes outside its ROI are kept.
    keep_crop (optional, isolate mode): only pixels in this region (the dilated target ROI) are kept,
    with a soft edge (keep_sigma, px); everything else fades to white, including unsegmented neighbors.
    """
    vmin = np.percentile(crop[~roi_crop & ~other_crop], low_pct)
    vmax = np.percentile(crop[roi_crop], high_pct)
    norm = np.clip((crop - vmin) / (vmax - vmin), 0, 1) ** gamma
    fade = gaussian_filter(other_crop.astype(float), fade_sigma) if fade_sigma else other_crop.astype(float)
    fade[roi_crop] = 0                        # never fade the target cell itself
    norm = norm * (1 - fade)
    if keep_crop is not None:
        keep = gaussian_filter(keep_crop.astype(float), keep_sigma) if keep_sigma else keep_crop.astype(float)
        keep[roi_crop] = 1                    # target cell always fully kept
        norm = norm * keep
    inverted = 1 - norm                      # bright soma -> 0 -> full cluster color
    return cluster_cmap(cluster_id)(inverted)


def crispness(max_projection, roi_mask, ring=(1, 3)):
    """
    How crisp a cell looks in the max projection, from its ROI and the pixels just outside it.
      edge_sharpness = (ROI - ring) / (ROI - background), using medians.
          ring = annulus `ring[0]`-`ring[1]` px outside the ROI; background = everything farther out
          (within a 25 px window). ~1: brightness drops to background right at the cell edge (crisp);
          ~0: a bright halo continues past the edge (blurry / diffuse). Scale-free.
      snr = (ROI - background) / std(background): how bright the cell is relative to background noise.
      Cells no brighter than background (ROI <= background) get edge_sharpness = 0.
    """
    near = binary_dilation(roi_mask, iterations=ring[0])
    far = binary_dilation(roi_mask, iterations=ring[1])
    window = binary_dilation(roi_mask, iterations=25)
    roi_val = np.median(max_projection[roi_mask])
    ring_val = np.median(max_projection[far & ~near])
    bg = max_projection[window & ~far]
    bg_val = np.median(bg)
    contrast = roi_val - bg_val
    # a cell no brighter than its background can't look crisp; also avoids dividing by zero
    edge_sharpness = (roi_val - ring_val) / contrast if contrast > 0 else 0.0
    return dict(edge_sharpness=edge_sharpness, snr=contrast / bg.std())


def make_tile(max_projection, roi_masks, cell_roi_id, cluster_id, half_width=20,
              low_pct=50, high_pct=95, gamma=1.0, dilation=2, fade_sigma=1.5,
              isolate=True, isolate_dilation=4, isolate_sigma=2):
    """One colored tile for one cell. roi_masks: dataset.roi_masks (all ROIs in this field of view).
    Neighboring ROIs are dilated by `dilation` px before fading, to also cover their dim edges.
    isolate=True additionally keeps only the target ROI grown by `isolate_dilation` px (soft edge
    `isolate_sigma` px); larger isolate_dilation keeps more of the cell's processes."""
    roi_mask = get_roi_mask(roi_masks, cell_roi_id)
    assert roi_mask.shape == max_projection.shape, f'mask {roi_mask.shape} != image {max_projection.shape}'
    others = np.zeros_like(roi_mask)
    for m in roi_masks[roi_masks.cell_roi_id != cell_roi_id].roi_mask.values:
        others |= m.astype(bool)
    if dilation:
        others = binary_dilation(others, iterations=dilation)
    others &= ~roi_mask
    crop = crop_around_roi(max_projection, roi_mask, half_width)
    roi_crop = crop_around_roi(roi_mask, roi_mask, half_width)
    other_crop = crop_around_roi(others, roi_mask, half_width)
    keep_crop = None
    if isolate:
        keep_crop = crop_around_roi(binary_dilation(roi_mask, iterations=isolate_dilation), roi_mask, half_width)
    return colorize(crop, cluster_id, roi_crop, other_crop, low_pct, high_pct, gamma, fade_sigma,
                    keep_crop, isolate_sigma)


# Session to take each cell type's max projection from: Novel for Vip/Exc (ROIs brighter),
# Familiar for Sst (dominant session for Sst)
SESSION_BY_CELL_TYPE = {'Excitatory': 'Novel', 'Vip Inhibitory': 'Novel', 'Sst Inhibitory': 'Familiar'}


def get_roi_mask(roi_masks, cell_roi_id):
    """Full-frame ROI mask, looked up the same way as summary_figures.plot_cell_zoom."""
    return roi_masks[roi_masks.cell_roi_id == cell_roi_id].roi_mask.values[0].astype(bool)


def make_all_tiles(cluster_meta, matched_cells_table, save_dir, session_by_cell_type=SESSION_BY_CELL_TYPE,
                   half_width=20, low_pct=50, high_pct=95, gamma=1.0, dilation=2, fade_sigma=1.5,
                   isolate=True, isolate_dilation=4, isolate_sigma=2):
    """
    Save one PNG tile per clustered cell, using its max projection from the session
    given for its cell type in `session_by_cell_type`.

    cluster_meta: indexed by cell_specimen_id, columns cluster_id and cell_type.
    matched_cells_table: rows per cell_specimen_id x experience_level with ophys_experiment_id.
    Tile settings (low_pct, high_pct, gamma, dilation, fade_sigma, isolate*) are described in make_tile / colorize.
    Datasets are loaded once per experiment. Returns a manifest DataFrame (also saved as CSV).
    """
    os.makedirs(save_dir, exist_ok=True)
    cells = matched_cells_table[['cell_specimen_id', 'experience_level', 'ophys_experiment_id']]
    cells = cells.merge(cluster_meta[['cluster_id', 'cell_type']], left_on='cell_specimen_id', right_index=True)
    cells = cells[cells.experience_level == cells.cell_type.map(session_by_cell_type)]
    cells = cells.drop_duplicates(subset='cell_specimen_id').set_index('cell_specimen_id')
    print(f'{len(cells)} of {len(cluster_meta)} clustered cells found in their chosen session')

    cache = get_sdk_cache()  # create before the loop so a cache problem stops the run instead of skipping every experiment
    rows = []
    for i, (ophys_experiment_id, expt_cells) in enumerate(cells.groupby('ophys_experiment_id')):
        print(f'{i + 1}/{cells.ophys_experiment_id.nunique()} experiment {ophys_experiment_id}: {len(expt_cells)} cells')
        try:
            dataset = cache.get_behavior_ophys_experiment(ophys_experiment_id)
        except Exception as e:  # some experiments fail to load; skip them rather than stop the run
            print('  could not load:', e)
            continue
        max_projection = np.asarray(getattr(dataset.max_projection, 'data', dataset.max_projection), dtype=float)
        ct = dataset.cell_specimen_table
        roi_masks = dataset.roi_masks  # same full-frame masks plot_cell_zoom uses

        for csid, row in expt_cells.iterrows():
            cell_roi_id = ct.loc[csid].cell_roi_id
            metrics = crispness(max_projection, get_roi_mask(roi_masks, cell_roi_id))
            rgba = make_tile(max_projection, roi_masks, cell_roi_id, row.cluster_id, half_width,
                             low_pct, high_pct, gamma, dilation, fade_sigma,
                             isolate, isolate_dilation, isolate_sigma)
            # sharpness x100 first (clipped to 000-999) so files sort by crispness within the folder
            sharp = int(np.clip(round(metrics['edge_sharpness'] * 100), 0, 999))
            cell_type_short = row.cell_type.split(' ')[0][:3]  # Exc / Sst / Vip
            filename = f'{sharp:03d}_{cell_type_short}_cluster{row.cluster_id:02d}_csid{csid}_expt{ophys_experiment_id}.png'
            plt.imsave(os.path.join(save_dir, filename), rgba)
            rows.append(dict(cell_specimen_id=csid, ophys_experiment_id=ophys_experiment_id,
                             cluster_id=row.cluster_id, cluster_label=CLUSTER_LABELS[row.cluster_id],
                             cell_type=row.cell_type, filename=filename, **metrics))

    manifest = pd.DataFrame(rows)
    manifest.to_csv(os.path.join(save_dir, 'tile_manifest.csv'), index=False)
    return manifest


def preview_tiles(manifest, save_dir, n_per_cluster=8, cell_type=None):
    """Grid of example tiles: one row per cluster, for checking crop size / contrast before tiling the cover."""
    m = manifest if cell_type is None else manifest[manifest.cell_type == cell_type]
    cluster_ids = sorted(m.cluster_id.unique())
    fig, axes = plt.subplots(len(cluster_ids), n_per_cluster,
                             figsize=(n_per_cluster * 0.9, len(cluster_ids) * 0.9), squeeze=False)
    for r, cid in enumerate(cluster_ids):
        files = m[m.cluster_id == cid].filename.values[:n_per_cluster]
        for c, ax in enumerate(axes[r]):
            ax.axis('off')
            if c < len(files):
                ax.imshow(plt.imread(os.path.join(save_dir, files[c])))
        axes[r, 0].set_title(f'{cid}: {CLUSTER_LABELS[cid]}', fontsize=7, loc='left')
    fig.tight_layout()
    return fig


def preview_by_metric(manifest, save_dir, metric='edge_sharpness', cell_type=None, n_bins=5, n_per_bin=10, seed=0):
    """Random example tiles from each quantile bin of `metric` (lowest bin on top), labeled with the bin range.
    Use it to pick a cutoff: find the row where tiles start looking crisp, then filter manifest[metric] > value."""
    m = manifest if cell_type is None else manifest[manifest.cell_type == cell_type]
    n_rows = len(m)
    m = m.dropna(subset=[metric])
    if len(m) == 0:
        raise ValueError(f'no {metric} values for cell_type={cell_type!r}: {n_rows} rows in manifest, all NaN. '
                         f'Manifest cell types: {sorted(manifest.cell_type.unique())}')
    m = m.assign(bin=pd.qcut(m[metric], min(n_bins, len(m)), duplicates='drop'))  # fewer bins for small samples
    n_bins = m.bin.nunique()
    fig, axes = plt.subplots(n_bins, n_per_bin, figsize=(n_per_bin * 0.9, n_bins * 1.0), squeeze=False)
    for r, (interval, group) in enumerate(m.groupby('bin', observed=True)):
        files = group.sample(min(n_per_bin, len(group)), random_state=seed).filename.values
        for c, ax in enumerate(axes[r]):
            ax.axis('off')
            if c < len(files):
                ax.imshow(plt.imread(os.path.join(save_dir, files[c])))
        axes[r, 0].set_title(f'{metric} {interval.left:.2f} to {interval.right:.2f}', fontsize=7, loc='left')
    fig.suptitle(cell_type or 'all cells', y=1.0)
    fig.tight_layout()
    return fig
