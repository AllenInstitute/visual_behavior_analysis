"""Composite the cell tiles onto the cover canvas.

Each tile is a real max-projection crop, inverted and painted in its cluster colour on white:
tile = (1 - v) * white + v * tile_colour, where v is the cell's normalised brightness. The renderer recovers v,
repaints the cell in the layout's `color`, and either
  light: multiplies onto white (white = transparent; overlapping cells darken like ink), or
  dark:  adds light onto black (screen blend), like fluorescence.
Only colour and size change; each tile's pixel structure is the cell's real image.
"""
import os
import numpy as np
from PIL import Image
from matplotlib.colors import to_rgb
from cover_layout import W, H, TILE_COLORS


def _intensity(t, tile_color):
    c = np.asarray(to_rgb(tile_color), np.float32)
    return np.clip((1 - t).max(axis=2) / max((1 - c).max(), 1e-3), 0, 1)[..., None]


def render(df, tile_dir, path, mode='dark', pupil_r=None, gamma_dark=1.4):
    """pupil_r: radius of the dark pupil disk drawn in the light version (the dark version is black already)."""
    dark = mode == 'dark'
    canvas = np.zeros((H, W, 3), np.float32) if dark else np.ones((H, W, 3), np.float32)
    cache = {}
    for r in df.itertuples():
        s = int(round(r.size))
        if s < 4:
            continue
        key = (r.filename, s, r.color)
        if key not in cache:
            im = Image.open(os.path.join(tile_dir, r.filename)).convert('RGB').resize((s, s), Image.LANCZOS)
            v = _intensity(np.asarray(im, np.float32) / 255, TILE_COLORS[r.cluster_id])
            c = np.asarray(to_rgb(r.color), np.float32)
            if dark:   # same hue, lifted so it reads on black; grays stay dim so they recede
                lit = c * 0.45 if c.max() - c.min() < 0.05 else c / max(c.max(), 1e-3) * 0.8 + 0.12
                cache[key] = v ** gamma_dark * lit
            else:
                cache[key] = 1 - v * (1 - c)
        t = cache[key]
        x0, y0 = int(round(r.X - s / 2)), int(round(r.Y - s / 2))
        xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + s, W), min(y0 + s, H)
        if xa >= xb or ya >= yb:
            continue
        patch = t[ya - y0:yb - y0, xa - x0:xb - x0]
        if dark:
            canvas[ya:yb, xa:xb] = 1 - (1 - canvas[ya:yb, xa:xb]) * (1 - patch)
        else:
            canvas[ya:yb, xa:xb] *= patch
    if pupil_r is not None and not dark:          # soft-edged dark pupil for the light version
        from cover_layout import CX, CY
        yy, xx = np.ogrid[:H, :W]
        a = np.clip((pupil_r - np.hypot(xx - CX, yy - CY)) / 12, 0, 1)[..., None]
        canvas = canvas * (1 - a) + np.array([0.06, 0.06, 0.08], np.float32) * a
    img = Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8))
    if path.lower().endswith(('.tif', '.tiff')):
        img.save(path, dpi=(300, 300), compression='tiff_lzw')
    else:
        img.save(path, dpi=(300, 300))
    return path
