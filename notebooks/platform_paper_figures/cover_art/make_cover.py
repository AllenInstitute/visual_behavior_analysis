"""Render an iris cover layout to files (used by cover_designs.ipynb).

For each design writes, in out_dir:
  <name>_dark.tif / <name>_light.tif  full size (9 x 11.5 in, 300 dpi, LZW) -- submission files
  <name>_dark_preview.jpg / ..._light_preview.jpg  small previews
  <name>_layout.csv  one row per neuron: which cell is where, its cluster and colour
"""
import os
from PIL import Image
import cover_render as R

PREVIEW_W = 900


def make_design(name, layout, tile_dir, out_dir, pupil_r, modes=('dark', 'light')):
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for mode in modes:
        tif = os.path.join(out_dir, f'{name}_{mode}.tif')
        R.render(layout, tile_dir, tif, mode=mode, pupil_r=pupil_r)
        im = Image.open(tif)
        prev = os.path.join(out_dir, f'{name}_{mode}_preview.jpg')
        im.resize((PREVIEW_W, round(PREVIEW_W * im.height / im.width)), Image.LANCZOS).save(prev, quality=92)
        paths[mode] = tif
    keep = [c for c in ['cell_specimen_id', 'cell_type', 'cluster_id', 'preferred_experience', 'color',
                        'X', 'Y', 'size', 'filename'] if c in layout.columns]
    layout[keep].to_csv(os.path.join(out_dir, f'{name}_layout.csv'), index=False)
    return paths
