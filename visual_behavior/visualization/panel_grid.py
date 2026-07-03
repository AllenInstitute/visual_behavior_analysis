"""
panel_grid.py -- gutter-aware composite figure layout.

THE DESIGN
----------
Every panel has an OUTER region [x0, x1, y0, y1] in figure fraction (y TOP-DOWN
to match placeAxesOnGrid).  Within the outer region a LABEL GUTTER is reserved
at the TOP and LEFT for the panel letter (and the panel's own y-axis tick
labels, which naturally extend left of the spine).  The axes spine + title +
data sit BELOW the top gutter and RIGHT of the left gutter.

  ┌─────── outer region (panel X) ───────┐
  │ ▓▓▓▓▓ TOP GUTTER (label letter) ▓▓▓▓ │
  │ ▓▓  ┌──── axes spine ────────┐       │   <- title (if has_title=True)
  │ ▓▓  │   data area            │       │      sits in extra title room
  │ ▓▓  │   (ticks, labels go    │       │      below the gutter
  │ ▓▓  │    into the gutter)    │       │
  │ ▓▓  └────────────────────────┘       │
  └──────────────────────────────────────┘

Because every panel owns its own gutter, panels can touch edge-to-edge and
labels still can't collide with neighbor content.

USAGE
-----
    from panel_grid import fig_setup, panel, show_label_gutters, check_label_overlap

    fig = plt.figure(figsize=(18, 18))
    fig_setup(fig)                              # makes xspan == figure fraction

    REGIONS = {
        'A': [0.03, 0.30, 0.02, 0.98],
        'B': [0.32, 0.55, 0.02, 0.27],
        ...
    }

    # Optional: red-box overlay BEFORE plotting to plan placement.
    # show_label_gutters(fig, REGIONS)

    ax_A = panel(fig, REGIONS['A'][:2], REGIONS['A'][2:], 'A')
    ax_B = panel(fig, REGIONS['B'][:2], REGIONS['B'][2:], 'B', has_title=True)
    ...

    # After plotting, audit:
    panels = {lbl: {'region': r, 'axes': locals()['ax_'+lbl]} for lbl, r in REGIONS.items()}
    check_label_overlap(fig, panels)            # prints + returns list of violations

KEY DETAIL
----------
`fig_setup(fig)` calls `fig.subplots_adjust(left=0, right=1, top=1, bottom=0)`
so the GridSpec inside placeAxesOnGrid maps xspan/yspan directly to figure
fraction.  Without this, GridSpec's default margins shift everything ~12% in
from each edge and the gutter math is wrong.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from visual_behavior.visualization.utils import placeAxesOnGrid


# Two-zone model (in POINTS so it scales with fontsize, not figsize):
#
#   ┌──────────────────────── outer region ─────────────────────────┐
#   │ ░░░ LABEL GUTTER ░░░ │                                        │   <- red, letter only
#   │ ░ │                                                           │
#   │ ░ │   DECORATION                                               │   <- invisible, holds
#   │ ░ │   SETBACK     ┌── title room ─────────────────────────┐    │      y-tick labels,
#   │ ░ │     ↓         │           (matplotlib title)          │    │      y-axis label,
#   │ ░ │  (y-tick     ┌─ axes spine ────────────────────────┐  │    │      and leftmost
#   │ ░ │   labels,    │                                     │  │    │      x-tick label
#   │ ░ │   y-axis     │   data                              │  │    │      overflow
#   │ ░ │   label)     │                                     │  │    │
#   │ ░ │              └─────────────────────────────────────┘  │    │
#   │ ░ │                                                       │    │
#   └────────────────────────────────────────────────────────────────┘
#
# show_label_gutters() draws ONLY the small red label gutter.  Plot decoration
# (y-axis labels/tick labels, leftmost x-tick label that overhangs the spine)
# lives in the larger setback zone, which is invisible and never touches the
# red.
DEFAULT_LABEL_GUTTER_PT = (20, 28)        # (left_pt, top_pt)  -- visible red, letter only (20pt: fits a 24pt bold letter and keeps plots tight to the letter)
DEFAULT_DECORATION_PT   = (80, 0)         # (left_pt, top_pt)  -- invisible setback (initial placement)
                                          #   Wide enough so the placement step never INTRUDES into
                                          #   the gutter at seaborn font_scale=1.5 (worst case).
                                          #   Run auto_fit_left() afterwards to SHRINK the setback
                                          #   to exactly what each panel needs -- the placement
                                          #   default is intentionally conservative; the fit-step
                                          #   makes it adaptive.
DEFAULT_TITLE_ROOM_PT   = 28              # extra top room when has_title=True; scales with title_lines


def fig_setup(fig):
    """Set figure margins so placeAxesOnGrid xspan/yspan == figure fraction.

    Call this ONCE per figure, before any panel(), so that the figure-fraction
    arithmetic this module does for gutters matches the actual rendered axes
    positions.
    """
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)


def _zone_frac(fig, gutter_pt, decoration_pt, title_room_pt, has_title, title_lines=1):
    """Return (gx, gy, dx, dy, title) all in figure fraction.
    gx, gy   : visible label gutter (letter only)
    dx, dy   : invisible decoration setback (axis labels/ticks live here)
    title    : extra room below the gutter for an axes title; scales with title_lines
    """
    w, h = fig.get_size_inches()
    gx = gutter_pt[0] / 72.0 / w
    gy = gutter_pt[1] / 72.0 / h
    dx = decoration_pt[0] / 72.0 / w
    dy = decoration_pt[1] / 72.0 / h
    title = (title_room_pt * title_lines / 72.0 / h) if has_title else 0.0
    return gx, gy, dx, dy, title


def panel(fig, xspan, yspan, label=None, has_title=False, title_lines=1,
          dim=(1, 1), wspace=None, hspace=None, sharex=False, sharey=False,
          width_ratios=None, height_ratios=None,
          gutter_pt=DEFAULT_LABEL_GUTTER_PT, decoration_pt=DEFAULT_DECORATION_PT,
          title_room_pt=DEFAULT_TITLE_ROOM_PT,
          label_fontsize=24, label_fontweight='bold'):
    """Place a labeled panel; axes sits below the top gutter and right of the left gutter.

    xspan, yspan : outer region in figure fraction (y TOP-DOWN, like placeAxesOnGrid)
    label        : panel letter (or None to skip the label)
    has_title    : if True, reserve extra top room so a matplotlib title sits below the gutter
    dim, wspace, hspace, sharex, sharey, width_ratios, height_ratios : passed to placeAxesOnGrid

    Returns the axes (or list/array of axes if dim>(1,1)).
    """
    gx, gy, dx, dy, title_room = _zone_frac(fig, gutter_pt, decoration_pt,
                                            title_room_pt, has_title, title_lines)
    x0, x1 = xspan
    y0, y1 = yspan       # TOP-DOWN figure fraction

    label_ax = None
    if label is not None:
        # Place the panel letter inside the outer region's top-left.  Using
        # placeAxesOnGrid for the label ensures its coords match the main axes.
        label_ax = placeAxesOnGrid(fig, dim=[1, 1], xspan=[x0, x0 + gx], yspan=[y0, y0 + gy])
        label_ax.axis('off')
        label_ax.text(0.15, 0.55, label, fontsize=label_fontsize, fontweight=label_fontweight,
                      ha='left', va='center', transform=label_ax.transAxes)
        label_ax._panel_grid_label_ax = True   # mark so check_label_overlap can skip

    # Axes sits PAST the label gutter AND the decoration setback.  Decoration
    # (y-tick labels, y-axis label, leftmost x-tick label overflow) lives in the
    # invisible setback zone; the visible red gutter stays clean.
    #
    # placeAxesOnGrid uses GridSpec(100, 100) with int() (floor) when mapping
    # xspan/yspan to row/col indices.  That floor would push the inner axes UP
    # into the gutter when (y0 + gy) doesn't land cleanly on a 0.01 boundary --
    # ceil() to the next 0.01 cell so the axes lands BELOW the gutter.
    inner_x_left  = np.ceil((x0 + gx + dx)              * 100) / 100
    inner_y_top   = np.ceil((y0 + gy + dy + title_room) * 100) / 100
    inner_xspan = [inner_x_left, x1]
    inner_yspan = [inner_y_top,  y1]
    ax = placeAxesOnGrid(fig, dim=list(dim),
                         xspan=inner_xspan, yspan=inner_yspan,
                         wspace=wspace, hspace=hspace, sharex=sharex, sharey=sharey,
                         width_ratios=width_ratios, height_ratios=height_ratios)
    # Tag every leaf-axes in the returned object with the panel label so
    # check_label_overlap can identify ownership without manual tracking.
    def _tag(obj):
        if obj is None: return
        if hasattr(obj, 'spines'):
            obj._panel_grid_owner = label
        elif isinstance(obj, (list, tuple)):
            for o in obj: _tag(o)
        else:
            try:
                for o in obj: _tag(o)
            except TypeError:
                pass
    _tag(ax)
    return ax


def panel_title(fig, x_span, text, y_top=None, y_frac=None,
                gutter_pt=DEFAULT_LABEL_GUTTER_PT, fontsize=16,
                fontweight='normal', dy_pt=0.0, ha='center', **kw):
    """Section title centered horizontally over x_span (figure fraction).

    Draws a descriptive title on the panel-letter line so it reads as a section
    header beside the letter, WITHOUT displacing the plot -- panel() already
    seats the axes below the top label gutter, so the title occupies that gutter
    line next to the letter.

    x_span   : (x0, x1) figure-fraction horizontal extent to center over. Pass a
               single panel's [x0, x1], or (left_panel_x0, right_panel_x1) to
               span several panels with one title.
    y_top    : the panel region's TOP edge (top-down fraction). The title is
               centered vertically in that panel's top gutter (same line as the
               letter). Ignored if y_frac is given.
    y_frac   : an explicit top-down fraction for the title's vertical center --
               use to float a spanning title in the gap ABOVE a row of panels
               (e.g. one title over two side-by-side panels whose letters would
               otherwise sit under a centered header).
    dy_pt    : nudge down (+) / up (-) in points.

    Returns the matplotlib Text artist. (fig.text is not tracked by
    check_label_overlap, which only inspects axes -- placement is by geometry.)
    """
    x0, x1 = x_span
    w, h = fig.get_size_inches()
    if y_frac is not None:
        y_td = y_frac
    else:
        y_td = y_top + (gutter_pt[1] / 72.0 / h) / 2.0
    y_td += dy_pt / 72.0 / h
    _x = {'left': x0, 'right': x1}.get(ha, (x0 + x1) / 2.0)
    return fig.text(_x, 1.0 - y_td, text, ha=ha, va='center',
                    fontsize=fontsize, fontweight=fontweight, **kw)


def show_label_gutters(fig, regions, gutter_pt=DEFAULT_LABEL_GUTTER_PT,
                       color='red', alpha=0.18, edgecolor='red', lw=1.5):
    """Overlay translucent red rectangles for every region's LABEL GUTTER
    (letter-only zone).  The decoration setback is intentionally invisible --
    axis labels/tick labels live there and shouldn't trigger a red overlay.

    Call BEFORE plotting to plan placement, or after rendering to eyeball overlaps.
    """
    gx, gy, _, _, _ = _zone_frac(fig, gutter_pt, (0, 0), 0, False)
    for _, (x0, x1, y0, y1) in regions.items():
        top = mpatches.Rectangle((x0, 1 - y0 - gy), x1 - x0, gy,
                                 facecolor=color, alpha=alpha, edgecolor=edgecolor,
                                 linewidth=lw, transform=fig.transFigure, zorder=200)
        left = mpatches.Rectangle((x0, 1 - y1), gx, (y1 - y0) - gy,
                                  facecolor=color, alpha=alpha, edgecolor=edgecolor,
                                  linewidth=lw, transform=fig.transFigure, zorder=200)
        fig.patches.extend([top, left])


def show_decoration_zones(fig, regions, gutter_pt=DEFAULT_LABEL_GUTTER_PT,
                          decoration_pt=DEFAULT_DECORATION_PT,
                          color='gold', alpha=0.10):
    """OPTIONAL diagnostic: overlay the invisible decoration setback in gold so
    you can confirm axis decorations land there (not in the red gutter).
    """
    gx, gy, dx, dy, _ = _zone_frac(fig, gutter_pt, decoration_pt, 0, False)
    for _, (x0, x1, y0, y1) in regions.items():
        if dx > 0:
            dec_left = mpatches.Rectangle((x0 + gx, 1 - y1), dx, (y1 - y0) - gy,
                                          facecolor=color, alpha=alpha, edgecolor='none',
                                          transform=fig.transFigure, zorder=199)
            fig.patches.append(dec_left)
        if dy > 0:
            dec_top = mpatches.Rectangle((x0 + gx, 1 - y0 - gy - dy),
                                         (x1 - x0) - gx, dy,
                                         facecolor=color, alpha=alpha, edgecolor='none',
                                         transform=fig.transFigure, zorder=199)
            fig.patches.append(dec_top)


def auto_fit_left(fig, regions, gutter_pt=DEFAULT_LABEL_GUTTER_PT,
                  pad_pt=3.0, verbose=False, skip=()):
    """After plotting, expand each panel's axes so the y-axis decoration sits
    SNUGLY just past the label gutter -- adaptive setback, no wasted whitespace.

    For each panel:
      1. Measure the leftmost extent of the rendered tight bbox.
      2. Compute the shift needed for that leftmost to land at gutter_right + pad.
      3. Shift the axes (or every subplot in a multi-axes row) by that amount,
         growing each axes' width on the left (right edges stay fixed).
      4. For multi-subplot rows the shift is capped at the inter-subplot gap so
         neighboring subplots do not overlap.

    Run as the last step of a composite cell, after all plotting is done.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    w, h = fig.get_size_inches()
    gx = gutter_pt[0] / 72.0 / w
    pad = pad_pt / 72.0 / w

    by_owner = {}
    for ax in fig.axes:
        if getattr(ax, '_panel_grid_label_ax', False):
            continue
        owner = getattr(ax, '_panel_grid_owner', None)
        if owner is not None:
            by_owner.setdefault(owner, []).append(ax)

    for owner, axes in by_owner.items():
        if owner not in regions:
            continue
        if owner in skip:
            if verbose:
                print(f'  auto_fit_left skip panel "{owner}" (user requested)')
            continue
        x0 = regions[owner][0]
        gutter_right = x0 + gx
        try:
            bboxes = [ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
                      for ax in axes]
        except Exception:
            continue
        leftmost = min(bb.x0 for bb in bboxes)
        shift = (gutter_right + pad) - leftmost
        if abs(shift) < 0.001:
            continue

        # Cap negative shift (axes expanding leftward) at the inter-subplot
        # gap so multi-subplot rows don't overlap each other.
        if len(axes) > 1 and shift < 0:
            sorted_axes = sorted(axes, key=lambda a: a.get_position().x0)
            gaps = [sorted_axes[i+1].get_position().x0 - sorted_axes[i].get_position().x1
                    for i in range(len(sorted_axes) - 1)]
            min_gap = min(gaps) if gaps else 0
            shift = max(shift, -(min_gap * 0.9))     # leave 10% safety

        for ax in axes:
            pos = ax.get_position()
            new_x0 = pos.x0 + shift
            new_w = pos.width - shift   # shift<0 -> width grows; right edge fixed
            if new_w > 0.01 and new_x0 >= 0:
                ax.set_position([new_x0, pos.y0, new_w, pos.height])
        if verbose:
            print(f'  auto_fit_left shifted panel "{owner}" by {shift*72*w:+.1f} pt')


def snug_left_to_gutter(fig, regions, only=None, gutter_pt=DEFAULT_LABEL_GUTTER_PT,
                        pad_pt=2.0, passes=3, verbose=False):
    """After plotting, RESCALE each panel's x-extent so its leftmost rendered decoration
    (y-axis label / tick labels / left-side annotations) sits exactly at gutter_right + pad,
    with the panel's RIGHT edge held fixed.

    Why this and not auto_fit_left: auto_fit_left *grows* each subplot leftward and caps the
    shift at the inter-subplot gap, so it cannot snug a multi-column or vertically-stacked
    panel. This instead scales every sub-axes proportionally about the panel's right edge --
    columns/gaps are preserved, nothing overlaps -- and it MEASURES the rendered label, so
    there is never a leftover gap or a clipped label regardless of font/label width.

    Annotations pinned to data coords (e.g. cluster-id labels at a fixed data offset) move
    when the axes is rescaled, so `passes` (default 3) iterates to convergence.

    only : iterable of panel labels to adjust (None = every owner-tagged panel).
    gutter_pt : the gutter of the targeted panels (pass the same value used in panel()).
    Panels whose axes aren't tagged with _panel_grid_owner (bbox-embedded grids) are skipped.
    """
    w, h = fig.get_size_inches()
    gx = gutter_pt[0] / 72.0 / w
    pad = pad_pt / 72.0 / w
    for _ in range(max(1, passes)):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        by_owner = {}
        for ax in fig.axes:
            if getattr(ax, '_panel_grid_label_ax', False):
                continue
            owner = getattr(ax, '_panel_grid_owner', None)
            if owner is not None:
                by_owner.setdefault(owner, []).append(ax)
        for owner, axes in by_owner.items():
            if owner not in regions or (only is not None and owner not in only):
                continue
            x0, x1 = regions[owner][0], regions[owner][1]
            gutter_right = x0 + gx
            try:
                bboxes = [ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
                          for ax in axes]
            except Exception:
                continue
            leftmost = min(bb.x0 for bb in bboxes)
            spine_left = min(ax.get_position().x0 for ax in axes)
            decoration_w = spine_left - leftmost           # decoration extends this far left of the spine
            new_spine_left = gutter_right + pad + decoration_w
            old_span = x1 - spine_left
            new_span = x1 - new_spine_left
            if old_span <= 0 or new_span <= 0 or abs(new_span - old_span) < 1e-4:
                continue
            scale = new_span / old_span                    # scale about the fixed right edge x1
            for ax in axes:
                pos = ax.get_position()
                ax.set_position([new_spine_left + (pos.x0 - spine_left) * scale,
                                 pos.y0, pos.width * scale, pos.height])
            if verbose:
                print(f'  snug_left "{owner}": spine {spine_left:.4f} -> {new_spine_left:.4f}')


def snug_top_to_gutter(fig, regions, only=None, gutter_pt=DEFAULT_LABEL_GUTTER_PT,
                       pad_pt=2.0, passes=3, verbose=False):
    """Vertical analog of `snug_left_to_gutter`. Rescales each named panel about its BOTTOM
    edge so its topmost rendered decoration (axes title / column header) sits just below the
    top gutter (gutter_bottom - pad). This removes the empty space left when `title_room` /
    `has_title` reserves more room than the rendered header actually needs, so the content
    sits right under the panel letter. The bottom edge (x-tick labels / scale bars) stays put.

    Use on `has_title` panels (column headers, titles). One pass converges for axis-relative
    titles (`ax.set_title`); data-anchored headers (text at a fixed data y, like a heatmap's
    `ax.text(y=ymax, ...)`) move when the axes is rescaled, so `passes` iterates to converge.

    only / gutter_pt : as in snug_left_to_gutter. Owner-untagged (bbox-embedded) panels skipped.
    """
    w, h = fig.get_size_inches()
    gy = gutter_pt[1] / 72.0 / h
    pad = pad_pt / 72.0 / h
    for _ in range(max(1, passes)):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        by_owner = {}
        for ax in fig.axes:
            if getattr(ax, '_panel_grid_label_ax', False):
                continue
            owner = getattr(ax, '_panel_grid_owner', None)
            if owner is not None:
                by_owner.setdefault(owner, []).append(ax)
        for owner, axes in by_owner.items():
            if owner not in regions or (only is not None and owner not in only):
                continue
            gutter_bottom = 1.0 - regions[owner][2] - gy        # lower edge of the top gutter (bottom-up y)
            try:
                bboxes = [ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
                          for ax in axes]
            except Exception:
                continue
            topmost = max(bb.y1 for bb in bboxes)               # topmost rendered point (title/header)
            axes_top = max(a.get_position().y1 for a in axes)   # spine top of the top row
            axes_bottom = min(a.get_position().y0 for a in axes)
            decoration_h = topmost - axes_top                   # how far the title extends above the spine
            new_axes_top = (gutter_bottom - pad) - decoration_h
            old_span = axes_top - axes_bottom
            new_span = new_axes_top - axes_bottom
            if old_span <= 0 or new_span <= 0 or abs(new_span - old_span) < 1e-4:
                continue
            scale = new_span / old_span                         # scale about the fixed bottom edge
            for ax in axes:
                pos = ax.get_position()
                ax.set_position([pos.x0, axes_bottom + (pos.y0 - axes_bottom) * scale,
                                 pos.width, pos.height * scale])
            if verbose:
                print(f'  snug_top "{owner}": axes_top {axes_top:.4f} -> {new_axes_top:.4f}')


def auto_fit_bottom(fig, regions, pad_pt=3.0, verbose=False):
    """After plotting, shrink each panel's axes from the bottom so the x-tick
    labels and x-axis label stay INSIDE the panel's outer region (not spilling
    into the next row's top gutter).

    For each panel:
      1. Measure the lowest extent of the rendered tight bbox.
      2. If it falls below the outer region's bottom edge, shift the axes
         bottom UP by the deficit (the top edge stays fixed; height shrinks).
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    w, h = fig.get_size_inches()
    pad = pad_pt / 72.0 / h

    by_owner = {}
    for ax in fig.axes:
        if getattr(ax, '_panel_grid_label_ax', False):
            continue
        owner = getattr(ax, '_panel_grid_owner', None)
        if owner is not None:
            by_owner.setdefault(owner, []).append(ax)

    for owner, axes in by_owner.items():
        if owner not in regions:
            continue
        region = regions[owner]
        outer_bottom_bu = 1 - region[3]   # bottom-up fraction
        try:
            bboxes = [ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
                      for ax in axes]
        except Exception:
            continue
        lowest = min(bb.y0 for bb in bboxes)
        target = outer_bottom_bu + pad
        if lowest >= target - 0.001:
            continue
        shift_up = target - lowest
        for ax in axes:
            pos = ax.get_position()
            new_y0 = pos.y0 + shift_up
            new_h  = pos.height - shift_up
            if new_h > 0.01:
                ax.set_position([pos.x0, new_y0, pos.width, new_h])
        if verbose:
            print(f'  auto_fit_bottom shrank panel "{owner}" by {shift_up*72*h:.1f} pt')


def check_label_overlap(fig, regions, gutter_pt=DEFAULT_LABEL_GUTTER_PT,
                        tol_pt=2.5, verbose=True):
    """After rendering, report any axes whose tight bbox intrudes into ANOTHER panel's gutter.

    regions : dict[label] = [x0, x1, y0, y1]  (top-down y, figure fraction).
              Ownership is inferred from the `_panel_grid_owner` attribute that
              panel() stamps on each axes; the small invisible label sub-axes
              (marked with `_panel_grid_label_ax`) are skipped automatically.

    tol_pt : tolerance in points; intrusions smaller than this are ignored (covers outward
             tick marks and the matplotlib default tick-pad slop, which never overlap text
             content).

    Returns list of violation dicts.  Empty list = clean layout.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    gx, gy, _, _, _ = _zone_frac(fig, gutter_pt, (0, 0), 0, False)
    w, h = fig.get_size_inches()
    tol_x = tol_pt / 72.0 / w
    tol_y = tol_pt / 72.0 / h

    gutters = {}
    for lbl, region in regions.items():
        x0, x1, y0, y1 = region
        gutters[lbl] = [
            ('top',  x0,      1 - y0 - gy, x1,       1 - y0),
            ('left', x0,      1 - y1,      x0 + gx,  1 - y0 - gy),
        ]

    def _infer_owner_from_position(ax):
        # When an embedded function creates its own sub-axes inside a panel,
        # those axes don't carry _panel_grid_owner.  Infer the owning panel
        # from the axes' SPINE center (not tight bbox -- decorations can extend
        # past the panel's outer region and would falsely match a neighbor).
        pos = ax.get_position()
        cx = (pos.x0 + pos.x1) / 2
        cy_bu = (pos.y0 + pos.y1) / 2
        cy_td = 1 - cy_bu                              # convert to TOP-DOWN
        for lbl, (rx0, rx1, ry0, ry1) in regions.items():
            if rx0 <= cx <= rx1 and ry0 <= cy_td <= ry1:
                return lbl
        return None

    violations = []
    for ax in fig.axes:
        # Skip the tiny invisible label sub-axes created by panel()
        if getattr(ax, '_panel_grid_label_ax', False):
            continue
        owner = getattr(ax, '_panel_grid_owner', None)
        if owner is None:
            owner = _infer_owner_from_position(ax)
        try:
            bb = ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
        except Exception:
            continue
        ax_x0, ax_y0, ax_x1, ax_y1 = bb.x0, bb.y0, bb.x1, bb.y1
        for g_owner, rects in gutters.items():
            # NB: we DO check against the axes' own panel too.  The visible label
            # gutter is letter-only -- y-axis labels / leftmost x-tick labels are
            # supposed to sit in the (invisible) decoration setback to the right
            # of the gutter, not in the gutter itself.  An intrusion of the
            # axes into its own gutter is still a layout bug.
            for side, gx0, gy0, gx1, gy1 in rects:
                if not (ax_x1 <= gx0 + tol_x or ax_x0 >= gx1 - tol_x or
                        ax_y1 <= gy0 + tol_y or ax_y0 >= gy1 - tol_y):
                    v = dict(gutter_panel=g_owner, intruding_panel=owner or '<unowned>',
                             side=side,
                             ax_bbox=(ax_x0, ax_y0, ax_x1, ax_y1),
                             gutter_bbox=(gx0, gy0, gx1, gy1))
                    violations.append(v)
                    if verbose:
                        print(f'OVERLAP: axes of panel "{v["intruding_panel"]}" intrudes into panel '
                              f'"{g_owner}" ({side} gutter)')
    if verbose and not violations:
        print('check_label_overlap: clean.')
    return violations
