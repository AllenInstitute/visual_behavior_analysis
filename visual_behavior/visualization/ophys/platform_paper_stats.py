"""
Statistics for the platform paper. Two entry points:

- ``test_significant_metric_averages``: legacy one-way ANOVA + Tukey HSD path.
  Returns ``(anova, tukey_table)`` for backward compatibility with existing
  inline callers.

- ``test_significant_metric_averages_mlm``: hierarchical mixed linear model
  with a random intercept for ``mouse_id``, accounting for non-independence
  of cells from the same mouse. Falls back automatically to ANOVA / Welch's
  t-test when the data is too sparse for MLM. Returns a dict with a uniform
  ``pairwise`` DataFrame schema across all three paths.

- ``compute_stats``: unified wrapper around the two above. Returns a single
  ``stats_table`` DataFrame whose columns are consistent enough for the
  ``add_stats_to_plot*`` family to consume without branching on the path.
  Used by the plotters in ``platform_paper_figures.py``.
"""
import warnings
import os
import hashlib

import numpy as np
import pandas as pd

from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as mc
from statsmodels.stats.multitest import multipletests


def compute_icc(data, metric, group_column):
    """
    Compute the intraclass correlation coefficient (ICC) using a one-way random effects model.

    ICC = between-group variance / (between-group variance + within-group variance)

    This tells you what proportion of total variance in the metric is attributable to
    differences between groups (e.g., mice). A low ICC means most variance is within
    groups (between cells), and the grouping structure has little impact.

    Parameters
    ----------
    data : pd.DataFrame
    metric : str
    group_column : str

    Returns
    -------
    dict with keys: icc, mouse_variance, residual_variance
    """
    # Fit an intercept-only mixed model to decompose variance.
    # Patsy parses hyphens / dots / spaces in column refs as operators, so
    # rename the metric to a safe placeholder before building the formula
    # (covers metrics like 'all-images').
    safe_metric = '__metric__'
    fit_data = data.rename(columns={metric: safe_metric})
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            null_model = smf.mixedlm(f'{safe_metric} ~ 1', data=fit_data, groups=fit_data[group_column])
            null_result = null_model.fit(reml=True)
    except Exception:
        return {
            'icc': np.nan,
            'mouse_variance': np.nan,
            'residual_variance': np.nan
        }

    # Extract variance components
    # result.cov_re is the random effects covariance matrix (group-level variance)
    # result.scale is the residual variance (within-group)
    mouse_variance = float(null_result.cov_re.iloc[0, 0])
    residual_variance = float(null_result.scale)
    total_variance = mouse_variance + residual_variance

    if total_variance > 0:
        icc = mouse_variance / total_variance
    else:
        icc = 0.0

    return {
        'icc': icc,
        'mouse_variance': mouse_variance,
        'residual_variance': residual_variance
    }


def _summary_stats(values):
    """Return mean, SEM, std, 95% CI, and n for a 1-D array of values."""
    vals = np.asarray(values)
    vals = vals[~pd.isnull(vals)]
    n = len(vals)
    if n == 0:
        return {'mean': np.nan, 'sem': np.nan, 'std': np.nan,
                'ci_low': np.nan, 'ci_high': np.nan, 'n': 0}
    m = float(vals.mean())
    if n > 1:
        sd = float(vals.std(ddof=1))
        sem = sd / np.sqrt(n)
        tcrit = stats.t.ppf(0.975, df=n - 1)
        ci_low = m - tcrit * sem
        ci_high = m + tcrit * sem
    else:
        sd = np.nan
        sem = np.nan
        ci_low = ci_high = np.nan
    return {'mean': m, 'sem': sem, 'std': sd,
            'ci_low': ci_low, 'ci_high': ci_high, 'n': n}


_NAN_DESCRIPTIVE = {
    'mean': np.nan, 'sem': np.nan, 'std': np.nan,
    'ci_low': np.nan, 'ci_high': np.nan, 'n': 0,
}


def _compute_group_descriptives(data, metric, column_to_compare, group_column=None):
    """
    Compute mean / SEM / std / 95% CI / n for each level of column_to_compare,
    at both the cell (observation) level and (optionally) the mouse level.

    Cell-level stats are computed across all rows in `data`. These are good for
    showing the spread of the raw observations, but for hierarchical input they
    underestimate uncertainty because cells within a mouse are not independent.

    Mouse-level stats are computed by first averaging per (mouse, condition) and
    then taking mean / SEM / std / 95% CI across those mouse means. This matches
    the unit of analysis used by the MLM and is the appropriate error bar to
    plot alongside MLM-derived p-values. Returned only when `group_column` is
    provided and is a column of `data`; otherwise the mouse-level fields are
    NaN with n=0.

    If the input is *already* aggregated to one row per (mouse, condition), the
    cell-level and mouse-level descriptives would be identical. To avoid that
    duplication, the cell-level fields are filled with NaN in that case.

    Returns
    -------
    dict mapping group label -> dict with keys:
        mean_cells, sem_cells, std_cells, ci_low_cells, ci_high_cells, n_cells
        mean_mouse, sem_mouse, std_mouse, ci_low_mouse, ci_high_mouse, n_mouse
    """
    has_groups = group_column is not None and group_column in data.columns

    if has_groups:
        cells_per_unit = data.groupby([group_column, column_to_compare]).size()
        already_mouse_level = cells_per_unit.max() <= 1
        mouse_means = (data
                       .groupby([group_column, column_to_compare])[metric]
                       .mean()
                       .reset_index())
    else:
        already_mouse_level = False
        mouse_means = None

    out = {}
    for g, sub in data.groupby(column_to_compare):
        if already_mouse_level:
            # Each row is already a per-mouse value; reporting cell-level stats
            # here would just duplicate the mouse-level numbers.
            cell_stats = dict(_NAN_DESCRIPTIVE)
        else:
            cell_stats = _summary_stats(sub[metric].values)
        entry = {
            'mean_cells': cell_stats['mean'],
            'sem_cells': cell_stats['sem'],
            'std_cells': cell_stats['std'],
            'ci_low_cells': cell_stats['ci_low'],
            'ci_high_cells': cell_stats['ci_high'],
            'n_cells': cell_stats['n'],
        }
        if has_groups:
            mm = mouse_means.loc[mouse_means[column_to_compare] == g, metric].values
            ms = _summary_stats(mm)
        else:
            ms = dict(_NAN_DESCRIPTIVE)
        entry.update({
            'mean_mouse': ms['mean'],
            'sem_mouse': ms['sem'],
            'std_mouse': ms['std'],
            'ci_low_mouse': ms['ci_low'],
            'ci_high_mouse': ms['ci_high'],
            'n_mouse': ms['n'],
        })
        out[g] = entry
    return out


# Keys used to populate the pairwise descriptive columns. Order is preserved
# so the resulting DataFrame columns are stable. All keys are of the form
# `<stat>_<level>` where level is 'cells' or 'mouse'.
_DESCRIPTIVE_KEYS = (
    'mean_cells', 'sem_cells', 'std_cells', 'ci_low_cells', 'ci_high_cells', 'n_cells',
    'mean_mouse', 'sem_mouse', 'std_mouse', 'ci_low_mouse', 'ci_high_mouse', 'n_mouse',
)


def _pairwise_descriptive_labels():
    """
    Return ordered (label1, label2) pairs corresponding to _DESCRIPTIVE_KEYS.

    Naming rules:
      - 'ci_low_<level>' / 'ci_high_<level>'
            -> 'ci1_low_<level>' / 'ci2_low_<level>' (and high)
      - 'stat_<level>'  ->  'stat1_<level>' / 'stat2_<level>'
    """
    out = []
    for key in _DESCRIPTIVE_KEYS:
        if key.startswith('ci_'):
            tail = key[len('ci_'):]
            out.append((f'ci1_{tail}', f'ci2_{tail}'))
        else:
            head, tail = key.split('_', 1)
            out.append((f'{head}1_{tail}', f'{head}2_{tail}'))
    return out


def _pairwise_descriptives(group_stats, g1, g2):
    """Return descriptive-stat columns for the two groups in a pairwise comparison."""
    s1 = group_stats.get(g1, {})
    s2 = group_stats.get(g2, {})
    out = {}
    for key, (label1, label2) in zip(_DESCRIPTIVE_KEYS, _pairwise_descriptive_labels()):
        default = 0 if key.startswith('n_') else np.nan
        out[label1] = s1.get(key, default)
        out[label2] = s2.get(key, default)
    return out


# Column ordering: metadata first, then comparison fields, then model/sample-size
# context, then cell-level descriptives, then mouse-level descriptives.
# `cell_type` sits alongside `event_type` because nearly every analysis is
# stratified by cell type before the within-cell-type comparison; recording it
# at construction time keeps the column in canonical position without callers
# needing to append it post hoc.
_LEADING_COLUMNS = ['metric', 'event_type', 'cell_type', 'column_to_compare', 'groups',
                    'group1', 'group2', 'x1', 'x2',
                    'diff', 'pvalue_raw', 'pvalue_adj', 'pvalue', 'reject',
                    'model_type', 'omnibus_pvalue', 'icc',
                    'mouse_variance', 'residual_variance',
                    'n_observations', 'n_groups', 'n_per_group']


def _empty_pairwise_columns():
    """Column list for the empty-pairwise edge case (must match populated output)."""
    desc = []
    for label1, label2 in _pairwise_descriptive_labels():
        desc += [label1, label2]
    return _LEADING_COLUMNS + desc


def test_significant_metric_averages_mlm(data, metric,
                                          column_to_compare='experience_level',
                                          group_column='mouse_id',
                                          event_type='Not specified',
                                          cell_type='Not specified',
                                          infer_event_type_from_metric=False):
    """
    Test for significant differences in a metric across conditions using a mixed linear
    model with a random intercept for the grouping variable (e.g., mouse_id).

    This replaces the ANOVA + Tukey approach to account for the nested/hierarchical
    structure of the data (multiple cells per mouse).

    Parameters
    ----------
    data : pd.DataFrame
        Each row is one observation (typically one cell in one experiment).
        Must contain columns for: metric, column_to_compare, and group_column.
    metric : str
        Column name for the metric to test.
    column_to_compare : str
        Categorical column to compare across (e.g., 'experience_level', 'cell_type').
    group_column : str
        Column identifying the grouping/nesting variable (e.g., 'mouse_id').
    event_type : str, default ``'Not specified'``
        Optional label for the stimulus/event type the metric was computed over
        (e.g., 'changes', 'images', 'omissions'). Recorded as a column in the
        results table for downstream filtering/reporting; not used by the model.

    Returns
    -------
    results : dict with keys:
        'omnibus_pvalue' : float
            P-value from a Wald test of the overall effect of column_to_compare.
        'pairwise' : pd.DataFrame
            All pairwise comparisons. Columns (in order):
            - metric, event_type, column_to_compare, groups: metadata identifying the analysis
            - group1, group2: compared groups
            - x1, x2: integer indices for plotting
            - diff: mean difference (effect size)
            - pvalue: p-value for pairwise comparison
            - reject: bool, whether p < 0.05
            - model_type: 'mlm' or 'anova'
            - omnibus_pvalue: overall test p-value
            - icc: intraclass correlation (proportion of variance from mouse)
            - mouse_variance, residual_variance: variance components
            - n_observations: total number of observations
            - n_groups: number of mice/groups
            - n_per_group: mean observations per group
            - mean1_cells, sem1_cells, std1_cells, ci1_low_cells, ci1_high_cells,
              n1_cells: cell-level descriptives for group 1 (NaN if input is
              already mouse-aggregated)
            - mean2_cells, ... n2_cells: cell-level descriptives for group 2
            - mean1_mouse, sem1_mouse, std1_mouse, ci1_low_mouse, ci1_high_mouse,
              n1_mouse: mouse-level descriptives for group 1 (per-mouse means
              first, then summary stats across mice -- this is the appropriate
              error bar for plots reporting MLM stats)
            - mean2_mouse, ... n2_mouse: mouse-level descriptives for group 2
        'model_type' : str
            'mlm' if mixed model was used, 'anova' if fell back to standard ANOVA.
        'model_result' : fitted model object
        'icc' : float
            Intraclass correlation coefficient.
    """
    # Clean data
    data = data[~data[metric].isnull()].copy()

    # Infer event type from metric name only when explicitly requested.
    if infer_event_type_from_metric:
        metric_lower = str(metric).lower()
        if 'changes' in metric_lower:
            event_type = 'changes'
        elif 'image' in metric_lower:
            event_type = 'images'
        elif 'omissions' in metric_lower:
            event_type = 'omissions'
    groups = sorted(data[column_to_compare].unique())

    # Build group index mapper (for plot positioning)
    mapper = {str(g): i for i, g in enumerate(groups)}

    # Handle edge cases
    if len(groups) < 2:
        pairwise = pd.DataFrame(columns=_empty_pairwise_columns())
        return {
            'omnibus_pvalue': np.nan,
            'pairwise': pairwise,
            'model_type': 'none',
            'model_result': None,
            'icc': np.nan
        }

    # Decide whether MLM is appropriate
    repeated_measurements = False
    if group_column in data.columns:
        cells_per_unit = data.groupby([group_column, column_to_compare]).size()
        repeated_measurements = cells_per_unit.max() > 1

    use_mlm = (
        group_column in data.columns
        and data[group_column].nunique() >= 3
        and repeated_measurements
    )

    if use_mlm:
        try:
            return _run_mlm(data, metric, column_to_compare, group_column, groups, mapper,
                            event_type, cell_type)
        except Exception as exc:
            warnings.warn(f'MLM failed ({exc}); falling back to ANOVA/Tukey.', RuntimeWarning)
            return _run_anova_tukey(data, metric, column_to_compare, group_column, groups, mapper,
                                    event_type, cell_type)
    else:
        return _run_anova_tukey(data, metric, column_to_compare, group_column, groups, mapper,
                                event_type, cell_type)


def _build_pairwise_row(metric, column_to_compare, groups, g1, g2, mapper,
                         diff, pval, group_stats, event_type, cell_type):
    """
    Build a pairwise row using the canonical column ordering. Trailing fields
    (model_type, omnibus_pvalue, etc.) are left empty here and filled in by the
    caller -- but the keys are pre-inserted so DataFrame column order is stable.
    """
    row = {
        'metric': metric,
        'event_type': event_type,
        'cell_type': cell_type,
        'column_to_compare': column_to_compare,
        'groups': tuple(groups),
        'group1': g1,
        'group2': g2,
        'x1': mapper[str(g1)],
        'x2': mapper[str(g2)],
        'diff': diff,
        'pvalue_raw': pval,
        'pvalue_adj': pval,
        'pvalue': pval,
        'reject': (pval < 0.05) if pd.notnull(pval) else False,
        # placeholders -- callers overwrite with real values
        'model_type': None,
        'omnibus_pvalue': np.nan,
        'icc': np.nan,
        'mouse_variance': np.nan,
        'residual_variance': np.nan,
        'n_observations': 0,
        'n_groups': 0,
        'n_per_group': 0.0,
    }
    row.update(_pairwise_descriptives(group_stats, g1, g2))
    return row


def _run_mlm(data, metric, column_to_compare, group_column, groups, mapper,
             event_type='Not specified', cell_type='Not specified'):
    """Run mixed linear model with random intercept and extract pairwise comparisons."""

    # --- Compute ICC from intercept-only model ---
    icc_results = compute_icc(data, metric, group_column)

    # --- Sample size info ---
    n_observations = len(data)
    n_groups = data[group_column].nunique()
    n_per_group = n_observations / n_groups

    # --- Group-level descriptive statistics for reporting ---
    # Includes both cell-level and mouse-level mean/SEM/std/95% CI.
    group_stats = _compute_group_descriptives(data, metric, column_to_compare, group_column)

    # --- Fit the full model ---
    # Patsy parses hyphens / dots / spaces in column refs as operators, so
    # rename the metric to a safe placeholder before building the formula
    # (covers metrics like 'all-images').
    safe_metric = '__metric__'
    fit_data = data.rename(columns={metric: safe_metric})
    formula = f'{safe_metric} ~ C({column_to_compare})'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(formula=formula, data=fit_data, groups=fit_data[group_column])
        result = model.fit(reml=True)

    # --- Omnibus test (Wald test for overall effect of the categorical variable) ---
    # Match params by the patsy prefix `C(column)[T.` to avoid false substring matches
    param_prefix = f'C({column_to_compare})[T.'
    cat_params = [p for p in result.params.index if p.startswith(param_prefix)]
    if len(cat_params) > 0:
        r_matrix = np.zeros((len(cat_params), len(result.params)))
        for i, param in enumerate(cat_params):
            j = list(result.params.index).index(param)
            r_matrix[i, j] = 1
        wald = result.wald_test(r_matrix, scalar=True)
        omnibus_pvalue = float(wald.pvalue)
    else:
        omnibus_pvalue = np.nan

    # --- Pairwise comparisons ---
    pairwise_rows = []
    raw_pvals = []
    ref_group = groups[0]

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1, g2 = groups[i], groups[j]

            if g1 == ref_group:
                param_name = f'C({column_to_compare})[T.{g2}]'
                if param_name in result.params.index:
                    diff = result.params[param_name]
                    pval = result.pvalues[param_name]
                else:
                    diff = np.nan
                    pval = np.nan
            else:
                param1 = f'C({column_to_compare})[T.{g1}]'
                param2 = f'C({column_to_compare})[T.{g2}]'

                if param1 in result.params.index and param2 in result.params.index:
                    # diff is reported as (group2 - group1) to match group1/group2
                    # column order and the ANOVA-fallback convention. For the
                    # reference-group case above, diff = result.params[T.g2] is
                    # already (g2 - ref) = (g2 - g1). For the non-reference case,
                    # diff = (g2 - ref) - (g1 - ref) = (g2 - g1).
                    diff = result.params[param2] - result.params[param1]
                    # Wald contrast tests the same difference (sign-invariant for
                    # the 2-sided p, so contrast direction here doesn't matter).
                    contrast = np.zeros(len(result.params))
                    contrast[list(result.params.index).index(param2)] = 1
                    contrast[list(result.params.index).index(param1)] = -1
                    wald = result.wald_test(contrast, scalar=True)
                    pval = float(wald.pvalue)
                else:
                    diff = np.nan
                    pval = np.nan

            row = _build_pairwise_row(metric, column_to_compare, groups,
                                       g1, g2, mapper, diff, pval, group_stats,
                                       event_type, cell_type)
            row['model_type'] = 'mlm'
            row['omnibus_pvalue'] = omnibus_pvalue
            row['icc'] = icc_results['icc']
            row['mouse_variance'] = icc_results['mouse_variance']
            row['residual_variance'] = icc_results['residual_variance']
            row['n_observations'] = n_observations
            row['n_groups'] = n_groups
            row['n_per_group'] = round(n_per_group, 1)
            pairwise_rows.append(row)
            raw_pvals.append(pval)

    pairwise = pd.DataFrame(pairwise_rows)

    # Holm correction for family-wise error across all pairwise comparisons.
    if len(pairwise) > 0:
        pvals = np.asarray(raw_pvals, dtype=float)
        valid = np.isfinite(pvals)
        p_adj = np.full_like(pvals, np.nan, dtype=float)
        if valid.sum() > 0:
            _, corr_pvals, _, _ = multipletests(pvals[valid], alpha=0.05, method='holm')
            p_adj[valid] = corr_pvals
        pairwise['pvalue_raw'] = pvals
        pairwise['pvalue_adj'] = p_adj
        pairwise['pvalue'] = pairwise['pvalue_adj']
        pairwise['reject'] = pairwise['pvalue_adj'] < 0.05

    return {
        'omnibus_pvalue': omnibus_pvalue,
        'pairwise': pairwise,
        'model_type': 'mlm',
        'model_result': result,
        'icc': icc_results['icc']
    }


def _run_anova_tukey(data, metric, column_to_compare, group_column, groups, mapper,
                     event_type='Not specified', cell_type='Not specified'):
    """Fallback: standard ANOVA + Tukey HSD for non-nested data."""

    group_data = [data[data[column_to_compare] == g][metric] for g in groups]
    model_result = None
    if len(groups) == 2:
        # Keep omnibus and pairwise aligned for two-group tests.
        ttest = stats.ttest_ind(group_data[0], group_data[1], equal_var=False, nan_policy='omit')
        omnibus_pvalue = float(ttest.pvalue)
        model_result = ttest
    else:
        anova = stats.f_oneway(*group_data)
        omnibus_pvalue = anova.pvalue
        model_result = anova

    # Sample size info. When group_column isn't in data, we don't know how many
    # mice are represented -- report NaN rather than misleadingly setting it equal
    # to n_observations (which would silently imply 1 observation per mouse).
    n_observations = len(data)
    if group_column in data.columns:
        n_groups = data[group_column].nunique()
        n_per_group = n_observations / max(n_groups, 1)
    else:
        n_groups = np.nan
        n_per_group = np.nan

    # Group-level descriptive statistics for reporting (cell-level + mouse-level if available)
    group_stats = _compute_group_descriptives(data, metric, column_to_compare, group_column)

    pairwise_rows = []

    if len(groups) > 2:
        comp = mc.MultiComparison(data[metric], data[column_to_compare])
        post_hoc_res = comp.tukeyhsd()
        tukey_table = pd.read_html(post_hoc_res.summary().as_html(), header=0, index_col=0)[0]
        tukey_table = tukey_table.reset_index()
        # statsmodels labels the adjusted p column 'p-adj' (some versions: 'padj')
        pval_col = next((c for c in ['p-adj', 'padj', 'pvalue'] if c in tukey_table.columns), None)

        for _, row in tukey_table.iterrows():
            pval = row[pval_col] if pval_col is not None else np.nan
            g1, g2 = row['group1'], row['group2']
            new_row = _build_pairwise_row(metric, column_to_compare, groups,
                                           g1, g2, mapper,
                                           row.get('meandiff', np.nan), pval, group_stats,
                                           event_type, cell_type)
            # The Tukey table's reject is multiple-comparison-corrected, so
            # prefer it over the naive pval < 0.05 check from _build_pairwise_row.
            if 'reject' in row:
                new_row['reject'] = bool(row['reject'])
            new_row['pvalue_raw'] = pval
            new_row['pvalue_adj'] = pval
            new_row['pvalue'] = pval
            pairwise_rows.append(new_row)
    elif len(groups) == 2:
        # Two-group case: pairwise p-value equals omnibus p-value.
        g1, g2 = groups[0], groups[1]
        new_row = _build_pairwise_row(metric, column_to_compare, groups,
                                       g1, g2, mapper,
                                       group_data[1].mean() - group_data[0].mean(),
                                       omnibus_pvalue, group_stats,
                                       event_type, cell_type)
        new_row['pvalue_raw'] = omnibus_pvalue
        new_row['pvalue_adj'] = omnibus_pvalue
        new_row['pvalue'] = omnibus_pvalue
        pairwise_rows.append(new_row)

    # Add shared model/sample-size fields
    for row in pairwise_rows:
        row['model_type'] = 'anova'
        row['omnibus_pvalue'] = omnibus_pvalue
        row['icc'] = np.nan
        row['mouse_variance'] = np.nan
        row['residual_variance'] = np.nan
        row['n_observations'] = n_observations
        row['n_groups'] = n_groups
        row['n_per_group'] = round(n_per_group, 1)

    pairwise = pd.DataFrame(pairwise_rows)

    return {
        'omnibus_pvalue': omnibus_pvalue,
        'pairwise': pairwise,
        'model_type': 'anova',
        'model_result': model_result,
        'icc': np.nan
    }


def test_significant_metric_averages(data, metric, column_to_compare='experience_level'):
    """
    Legacy one-way ANOVA + Tukey HSD across the levels of ``column_to_compare``.

    Treats every row in ``data`` as independent (no nesting accounted for).
    Kept for backward compatibility and for inline use in heatmap functions
    that only need an omnibus p-value.

    Returns
    -------
    anova : scipy.stats.F_onewayResult-like
        Object with ``.statistic`` and ``.pvalue`` (a namedtuple is returned
        in the <2-groups edge case so the interface is uniform).
    tukey_table : pd.DataFrame
        Pairwise table with columns: group1, group2, x1, x2, reject,
        one_way_anova_p_val, column_to_compare, groups. For >2 groups the
        Tukey HSD meandiff/lower/upper columns are also included.
    """
    # remove null values
    data = data[~data[metric].isnull()].copy()
    # get conditions to compare
    groups = data[column_to_compare].unique()
    # run anova across groups depending on how many conditions there are
    if len(groups) < 2:
        from collections import namedtuple
        AnovaResult = namedtuple('AnovaResult', ['statistic', 'pvalue'])
        anova = AnovaResult(statistic=np.nan, pvalue=1.0)
        tukey_table = pd.DataFrame(columns=['group1', 'group2', 'x1', 'x2', 'reject', 'one_way_anova_p_val'])
        tukey_table['column_to_compare'] = column_to_compare
        return anova, tukey_table
    else:
        group_data = [data[data[column_to_compare] == g][metric] for g in groups]
        anova = stats.f_oneway(*group_data)
    # get group index mapper
    mapper = {}
    for i, group in enumerate(groups):
        mapper[str(group)] = i

    if len(data[column_to_compare].unique()) > 2:
        # create tukey table for multiple comparisons across all pairs that can be compared
        comp = mc.MultiComparison(data[metric], data[column_to_compare])
        post_hoc_res = comp.tukeyhsd()
        tukey_table = pd.read_html(post_hoc_res.summary().as_html(), header=0, index_col=0)[0]
        tukey_table = tukey_table.reset_index()
        tukey_table['x1'] = [mapper[str(x)] for x in tukey_table['group1']]
        tukey_table['x2'] = [mapper[str(x)] for x in tukey_table['group2']]
    elif len(data[column_to_compare].unique()) == 2:
        tukey_table = pd.DataFrame()
        tukey_table['group1'] = [groups[0]]
        tukey_table['group2'] = [groups[1]]
        tukey_table['x1'] = [0]
        tukey_table['x2'] = [1]

    tukey_table['column_to_compare'] = column_to_compare
    # Store compared groups as one value per tukey row to avoid length mismatch.
    tukey_table['groups'] = [tuple(groups)] * len(tukey_table)
    tukey_table['one_way_anova_p_val'] = anova[1]
    return anova, tukey_table


# Optional on-disk cache for compute_stats results. When set to a directory path, compute_stats
# loads a previously-saved result instead of recomputing (the hierarchical MLM is expensive). The
# cache key is a hash of the actual inputs (metric/comparison/grouping values + test settings), so
# a result is only reused for an identical computation. Set back to None to disable.
STATS_CACHE_DIR = None


def _stats_cache_path(data, metric, column_to_compare, use_mlm, group_column,
                      event_type, cell_type):
    if STATS_CACHE_DIR is None:
        return None
    cols = [c for c in [metric, column_to_compare, group_column] if c in data.columns]
    try:
        sub = data.loc[~data[metric].isnull(), cols] if metric in data.columns else data[cols]
        content = hashlib.md5(
            pd.util.hash_pandas_object(sub, index=False).values.tobytes()).hexdigest()
    except Exception:
        content = str(len(data))
    key = '|'.join(map(str, [metric, column_to_compare, use_mlm, group_column,
                             event_type, cell_type, content]))
    digest = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(STATS_CACHE_DIR, 'compute_stats_' + digest + '.pkl')


def compute_stats(data, metric, column_to_compare='experience_level', *,
                  use_mlm=True, group_column='mouse_id',
                  event_type='Not specified',
                  cell_type='Not specified'):
    """
    Unified stats entry point for the ``add_stats_to_plot*`` family.

    Returns a single ``stats_table`` DataFrame whose columns are consistent
    enough for the plotters to consume without branching on the underlying
    test. Both paths produce these columns:

    - ``x1``, ``x2``, ``group1``, ``group2``, ``reject``
    - ``omnibus_pvalue`` (same value on every row -- used for the omnibus gate)
    - ``one_way_anova_p_val`` (alias of pvalue_adj in the MLM path, of the
      ANOVA p in the legacy path; kept for the 2-group branch in legacy
      plotter code)

    MLM-specific columns (``icc``, ``mouse_variance``, ``model_type``,
    ``n_observations``, mouse-level descriptives, etc.) are present only when
    ``use_mlm=True``.

    Parameters
    ----------
    data, metric, column_to_compare : passed through to the underlying test.
    use_mlm : bool
        If True (default), run the hierarchical MLM (with auto-fallback to
        ANOVA / Welch t-test when the data is too sparse). If False, run the
        legacy one-way ANOVA + Tukey HSD.
    group_column : str
        Nesting variable for MLM (e.g., ``'mouse_id'``). Ignored when
        ``use_mlm=False``.
    event_type : str
        Optional label recorded as a column in the saved stats table.
    """
    # Warn if any level of column_to_compare will be silently dropped due to
    # all-NaN metric values. Both underlying paths do `data[~data[metric].isnull()]`
    # which removes whole levels without complaint -- a user expecting a 3-way
    # comparison can end up with a 2-way comparison and not notice.
    if column_to_compare in data.columns and metric in data.columns:
        all_levels = set(data[column_to_compare].dropna().unique())
        remaining = set(
            data.loc[~data[metric].isnull(), column_to_compare].dropna().unique()
        )
        dropped = all_levels - remaining
        if dropped:
            warnings.warn(
                f"compute_stats: dropping {len(dropped)} level(s) of "
                f"'{column_to_compare}' with all-NaN '{metric}' values: "
                f"{sorted(map(str, dropped))}. Test will run on remaining levels only.",
                RuntimeWarning,
            )

    # on-disk cache: only generate the (slow) stats if an identical result isn't already saved
    _cache_path = _stats_cache_path(data, metric, column_to_compare, use_mlm,
                                    group_column, event_type, cell_type)
    if _cache_path is not None and os.path.exists(_cache_path):
        return pd.read_pickle(_cache_path)

    if use_mlm:
        results = test_significant_metric_averages_mlm(
            data, metric, column_to_compare,
            group_column=group_column, event_type=event_type, cell_type=cell_type,
        )
        stats_table = results['pairwise'].copy()
        # legacy 2-group plotter branch reads `one_way_anova_p_val`; in MLM the
        # adjusted pairwise p is the right analog (== omnibus for 2 groups).
        if 'pvalue_adj' in stats_table.columns:
            stats_table['one_way_anova_p_val'] = stats_table['pvalue_adj']
        else:
            stats_table['one_way_anova_p_val'] = np.nan
        # omnibus_pvalue already a column (populated per-row in _run_mlm / _run_anova_tukey)
    else:
        anova, tukey = test_significant_metric_averages(data, metric, column_to_compare)
        stats_table = tukey.copy()
        # broadcast the omnibus p to a column so the plotter can read it uniformly
        stats_table['omnibus_pvalue'] = float(anova.pvalue) if hasattr(anova, 'pvalue') else float(anova[1])
        # Inject canonical leading metadata so the legacy path produces a table
        # with the same first few columns as the MLM path (metric, event_type,
        # cell_type, then column_to_compare, ...). Without this, the two paths
        # save CSVs with different column orderings.
        for col, val in (('cell_type', cell_type),
                         ('event_type', event_type),
                         ('metric', metric)):
            if col in stats_table.columns:
                stats_table[col] = val  # overwrite if already present
                continue
            stats_table.insert(0, col, val)

    if _cache_path is not None:
        os.makedirs(os.path.dirname(_cache_path), exist_ok=True)
        stats_table.to_pickle(_cache_path)
    return stats_table


# ---------------------------------------------------------------------------
# Post-hoc metadata insertion (used by the figures-file callers)
# ---------------------------------------------------------------------------

# Columns that compute_stats already emits. Passing these as kwargs to
# insert_stats_metadata is a no-op + warning -- the call site should stop
# adding them.
_RESERVED_METADATA = ('metric', 'event_type', 'cell_type', 'column_to_compare')


def insert_stats_metadata(stats_table, *, after='cell_type', **metadata):
    """
    Insert caller-supplied metadata columns into ``stats_table`` immediately
    after the ``after`` column (default 'cell_type'), so they appear up front
    in saved CSVs but after the canonical leading columns
    (metric, event_type, cell_type) that compute_stats emits.

    Use this for caller-known context that compute_stats doesn't have access to
    -- typically ``condition`` (the outer loop variable) and ``cohort`` /
    ``project_code``. ``cell_type`` and ``event_type`` should be passed to
    compute_stats / add_stats_to_plot* directly, not through here.

    Behavior:
      - None values are skipped (lets you pass through optional context cleanly).
      - Kwargs whose names duplicate columns that compute_stats already emits
        (metric, event_type, cell_type, column_to_compare) are dropped with a
        RuntimeWarning. Fix the call site instead of relying on this.
      - For non-reserved kwargs that already exist as columns, the value is
        replaced in place (preserving column position).
      - Remaining kwargs are inserted immediately after ``after``. If ``after``
        isn't in the table, they're inserted at the front.
      - Returns a new DataFrame; does not mutate the input.
    """
    if stats_table is None or len(stats_table) == 0:
        return stats_table

    out = stats_table.copy()

    # Drop None values up front.
    metadata = {k: v for k, v in metadata.items() if v is not None}

    # Filter reserved keys.
    cleaned = {}
    for k, v in metadata.items():
        if k in _RESERVED_METADATA:
            warnings.warn(
                f"insert_stats_metadata: dropping '{k}'={v!r} -- compute_stats "
                f"already emits this column. Pass it to compute_stats / "
                f"add_stats_to_plot* directly instead.",
                RuntimeWarning)
            continue
        cleaned[k] = v

    # In-place replace for keys already in the table.
    in_place_keys = [k for k in cleaned if k in out.columns]
    for k in in_place_keys:
        out[k] = cleaned[k]
        del cleaned[k]

    if not cleaned:
        return out

    # Insert remaining keys after `after`.
    cols = list(out.columns)
    try:
        anchor = cols.index(after) + 1
    except ValueError:
        anchor = 0
    for k, v in cleaned.items():
        out[k] = v
    new_keys = list(cleaned.keys())
    rest = [c for c in cols[anchor:] if c not in new_keys]
    return out[cols[:anchor] + new_keys + rest]
