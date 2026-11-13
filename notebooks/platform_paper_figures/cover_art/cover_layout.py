"""Iris cover layouts. Each returns a DataFrame with one row per neuron:
cell_specimen_id, filename, cell_type, cluster_id, X, Y (canvas px, y down), size (px), color (hex to paint the tile).

Shared geometry: concentric bands by cell type -- Vip at the pupil edge, Excitatory in the middle, Sst at the rim.
Each band's width is set from its number of cells so every band has the same tile density.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# Neuron asks for ~9 x 11.5 in (final trim 8 3/8 x 10 7/8 in) at >= 300 dpi
W, H = 2700, 3450
TOP = int(0.17 * H)          # top band kept clear for the journal masthead
CX, CY = W / 2, TOP + (H - TOP) / 2
BAND_ORDER = ['Vip Inhibitory', 'Excitatory', 'Sst Inhibitory']   # pupil -> rim
# interneuron tiles drawn larger so the thin Vip and Sst bands read (every cell still appears exactly once)
TYPE_SCALE = {'Vip Inhibitory': 1.6, 'Excitatory': 1.0, 'Sst Inhibitory': 1.7}
R_PUPIL, R_MAX = 330, 1130   # pupil radius and outer radius (px); R_MAX keeps the iris inside Neuron's trim

# colours the tiles were generated with (cell_tiles.LABEL_COLORS via CLUSTER_LABELS); needed to recover
# each tile's intensity before repainting it
TILE_COLORS = {1: '#BFBFBF', 2: '#aa1016', 3: '#572c92', 4: '#0b559f', 5: '#572c92', 6: '#55a868', 7: '#BFBFBF',
               8: '#539ecd', 9: '#8d89c0', 10: '#ccb974', 11: '#e3bbcd', 12: '#dd8452'}

# UMAP iris palette = graphical-abstract cluster palette, except cluster 6 (green) -> brown, because Cell Press
# asks that red and green not be used together on covers
UMAP_COLORS = {**TILE_COLORS, 6: '#8c564b'}

# pie iris palette = preferred experience level (utils.get_experience_level_colors), gray = no preference
EXPERIENCE_COLORS = {'Familiar': '#0b559f', 'Novel': '#aa1016', 'Novel +': '#572c92', None: '#a6a6a6'}


def load_cells(cluster_csv, manifest_csv, drop_clusters=()):
    """Clustered cells that have a tile. cluster_csv: index cell_specimen_id, columns cluster_id, cell_type
    (+ x, y for the UMAP design)."""
    c = pd.read_csv(cluster_csv, index_col=0)
    m = pd.read_csv(manifest_csv).set_index('cell_specimen_id')
    cells = c.join(m[['filename', 'edge_sharpness', 'snr']], how='inner')
    return cells[~cells.cluster_id.isin(drop_clusters)].reset_index()


def _bands_for(counts, r_pupil, base, grow, pack, gap):
    bands, r = {}, r_pupil
    for ct in BAND_ORDER:
        w = 0
        for _ in range(20):                       # tile size grows with radius, so iterate
            size = base * TYPE_SCALE[ct] * (1 + grow * (r + w / 2 - r_pupil) / 900)
            w = np.sqrt(r ** 2 + counts[ct] * (pack * size) ** 2 / np.pi) - r
        bands[ct] = (r, r + w)
        r += w + gap
    return bands


def band_radii(counts, r_pupil, r_max, grow, pack, gap):
    """Inner/outer radius of each cell-type band and the base tile size. Each band is sized so each tile gets
    ~(pack*size)^2 of area (pack < 1 lets tiles overlap; only the middle of a tile holds the cell), and the base
    tile size is chosen so the outermost band ends at r_max."""
    base = 40.0
    for _ in range(30):
        bands = _bands_for(counts, r_pupil, base, grow, pack, gap)
        outer = bands[BAND_ORDER[-1]][1]
        base *= ((r_max - r_pupil) / (outer - r_pupil)) ** 0.5
    return _bands_for(counts, r_pupil, base, grow, pack, gap), base


def _place(c, theta, r0, r1, r_pupil, base, grow, rng):
    rr = np.sqrt(rng.uniform(r0 ** 2, r1 ** 2, len(c)))      # uniform over the band's area
    scale = c.cell_type.map(TYPE_SCALE).values
    return c.assign(X=CX + rr * np.cos(theta), Y=CY + rr * np.sin(theta),
                    size=base * scale * (1 + grow * (rr - r_pupil) / 900))


def umap_iris(cells, r_pupil=R_PUPIL, r_max=R_MAX, grow=0.5, pack=0.62, gap=6, seed=0, hole=(9.6, 7.3)):
    """Angle = each cell's angle around the hole in the middle of the UMAP ring (rank-equalised within each
    cell type, so the band is evenly filled); colour = functional cluster."""
    rng = np.random.default_rng(seed)
    cells = cells.assign(theta_umap=np.arctan2(-(cells.y - hole[1]), cells.x - hole[0]))
    bands, base = band_radii(cells.cell_type.value_counts(), r_pupil, r_max, grow, pack, gap)
    rows = []
    for ct, (r0, r1) in bands.items():
        c = cells[cells.cell_type == ct].sort_values('theta_umap')
        n = len(c)
        theta = -np.pi / 2 + 2 * np.pi * (np.arange(n) + rng.uniform(-0.3, 0.3, n)) / n
        rows.append(_place(c, theta, r0, r1, r_pupil, base, grow, rng))
    out = pd.concat(rows)
    out['color'] = out.cluster_id.map(UMAP_COLORS)
    return finish(out, r_pupil, pack)


def pie_iris(cells, fractions, order, preferred_experience, r_pupil=R_PUPIL, r_max=R_MAX, grow=0.5, pack=0.62, gap=6,
             sector_gap_deg=0.8, seed=0):
    """Each cell-type band is that cell type's pie chart: clusters are angular sectors in `order`, clockwise from
    12 o'clock, each sector's width = the cluster's fraction of that cell type (`fractions`: cell_type x cluster).
    Cells of a cluster fill its sector; colour = the cluster's preferred experience level."""
    rng = np.random.default_rng(seed)
    bands, base = band_radii(cells.cell_type.value_counts(), r_pupil, r_max, grow, pack, gap)
    gap_rad = np.deg2rad(sector_gap_deg)
    rows = []
    for ct, (r0, r1) in bands.items():
        f = fractions.loc[ct, order]
        start = -np.pi / 2                                           # 12 o'clock, clockwise (y points down)
        for cid, frac in f.items():
            c = cells[(cells.cell_type == ct) & (cells.cluster_id == cid)]
            span = 2 * np.pi * frac
            if len(c) and span > 0:
                lo, hi = start + gap_rad / 2, start + max(span - gap_rad / 2, gap_rad)
                theta = rng.uniform(lo, hi, len(c))
                rows.append(_place(c.assign(sector_start=start, sector_end=start + span), theta,
                                   r0, r1, r_pupil, base, grow, rng))
            start += span
    out = pd.concat(rows)
    out['preferred_experience'] = out.cluster_id.map(preferred_experience)
    out['color'] = [EXPERIENCE_COLORS[p if isinstance(p, str) else None] for p in out.preferred_experience]
    return finish(out, r_pupil, pack, keep_sectors=True)


def finish(df, r_pupil, pack, keep_sectors=False):
    """Relax overlaps, then keep tiles out of the pupil (and inside their sector for the pie design)."""
    df = relax(df, pack)
    dx, dy = df.X - CX, df.Y - CY
    rr = np.hypot(dx, dy)
    k = np.maximum(rr, r_pupil + df['size'] * 0.3) / rr
    df = df.assign(X=CX + dx * k, Y=CY + dy * k)
    if keep_sectors:                       # relaxing may nudge a tile across a sector edge: clamp its angle back
        rr = np.hypot(df.X - CX, df.Y - CY)
        th = np.arctan2(df.Y - CY, df.X - CX)
        th = df.sector_start + np.mod(th - df.sector_start, 2 * np.pi)
        th = np.clip(th, df.sector_start, df.sector_end)
        df = df.assign(X=CX + rr * np.cos(th), Y=CY + rr * np.sin(th))
    return df


def relax(df, pack, iters=60, step=0.5):
    """Push overlapping tiles apart a little (keeps the overall arrangement)."""
    P = df[['X', 'Y']].values.astype(float)
    s = df['size'].values * pack
    for _ in range(iters):
        pairs = cKDTree(P).query_pairs(s.max())
        if not pairs:
            break
        pairs = np.array(list(pairs))
        d = P[pairs[:, 1]] - P[pairs[:, 0]]
        dist = np.linalg.norm(d, axis=1) + 1e-6
        over = (s[pairs[:, 0]] + s[pairs[:, 1]]) / 2 - dist
        m = over > 0
        if not m.any():
            break
        push = (d[m] / dist[m, None]) * (over[m, None] * step / 2)
        np.add.at(P, pairs[m, 1], push)
        np.add.at(P, pairs[m, 0], -push)
    return df.assign(X=P[:, 0], Y=P[:, 1])
