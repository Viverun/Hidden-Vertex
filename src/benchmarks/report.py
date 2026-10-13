"""Model-agnostic benchmark: any per-event score + labels -> AUC, CI, within-bin AUC, gate.

`scores` is one 1-D array or a list/tuple of arrays (one per training seed). With a list,
every AUC is the mean over the list and the bootstrap interval is for that seed-averaged AUC.
All scores follow "higher = more anomalous".
"""
import json

import numpy as np
from sklearn.metrics import roc_auc_score


def _as_list(scores):
    if isinstance(scores, (list, tuple)):
        return [np.asarray(s, dtype=np.float64) for s in scores]
    return [np.asarray(scores, dtype=np.float64)]


def mean_auc(y, scores):
    y = np.asarray(y)
    return float(np.mean([roc_auc_score(y, s) for s in _as_list(scores)]))


def auc_ci(y, scores, level=0.95, n_boot=1000, seed=0):
    """Percentile bootstrap CI over events (same resample for every seed in a list)."""
    y = np.asarray(y)
    arrays = _as_list(scores)
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        aucs.append(np.mean([roc_auc_score(yb, s[idx]) for s in arrays]))
    if not aucs:
        return float('nan'), float('nan')
    alpha = (1 - level) / 2
    return float(np.quantile(aucs, alpha)), float(np.quantile(aucs, 1 - alpha))


def binned_auc(y, scores, covariate, n_bins=10, min_per_class=20):
    """AUC inside quantile bins of `covariate` (bins of the whole test set, both classes).

    Bins with fewer than `min_per_class` events of either class are skipped.
    """
    y = np.asarray(y)
    cov = np.asarray(covariate, dtype=np.float64)
    arrays = _as_list(scores)
    edges = np.unique(np.quantile(cov, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return []
    bin_id = np.clip(np.searchsorted(edges, cov, side='right') - 1, 0, len(edges) - 2)
    out = []
    for b in range(len(edges) - 1):
        sel = bin_id == b
        pos = int(y[sel].sum())
        neg = int(sel.sum()) - pos
        if pos < min_per_class or neg < min_per_class:
            continue
        out.append({
            'lo': float(edges[b]), 'hi': float(edges[b + 1]), 'n': int(sel.sum()),
            'auc': float(np.mean([roc_auc_score(y[sel], s[sel]) for s in arrays])),
        })
    return out


def benchmark_row(name, y, scores, covariates, group, level, n_boot, n_bins=10, notes=''):
    lo, hi = auc_ci(y, scores, level=level, n_boot=n_boot)
    return {
        'name': name, 'group': group, 'auc': mean_auc(y, scores),
        'ci_lo': lo, 'ci_hi': hi, 'level': level, 'notes': notes,
        'within_bin': {c: binned_auc(y, scores, v, n_bins=n_bins) for c, v in covariates.items()},
    }


def within_bin_min(row):
    vals = [b['auc'] for bins in row['within_bin'].values() for b in bins]
    return min(vals) if vals else float('nan')


def evaluate_gate(rows, min_within_bin=0.6):
    """Spec Section 1: a model score passes if the lower CI bound of its (seed-averaged) AUC is above
    the best *particle-level* baseline AUC and its minimum within-bin AUC is >= `min_within_bin`.
    The headline model rows have group 'model'; jet-level and oracle baselines are not the bar."""
    bar_rows = [r for r in rows if r['group'] == 'particle']
    model_rows = [r for r in rows if r['group'] == 'model']
    if not bar_rows or not model_rows:
        raise ValueError('evaluate_gate needs at least one particle baseline and one model row')
    bar = max(bar_rows, key=lambda r: r['auc'])
    models = []
    for r in model_rows:
        wb = within_bin_min(r)
        beats = bool(r['ci_lo'] > bar['auc'])
        ok = bool(wb >= min_within_bin)
        models.append({'name': r['name'], 'auc': r['auc'], 'ci_lo': r['ci_lo'], 'beats_bar': beats,
                       'within_bin_min': wb, 'within_bin_ok': ok, 'passes': beats and ok})
    return {'bar_name': bar['name'], 'bar_auc': bar['auc'], 'min_within_bin': min_within_bin,
            'models': models, 'passed': any(m['passes'] for m in models)}


def check_reference(rows, expected):
    """Stage 0 check: each expected AUC must lie inside the harness's own CI for that row."""
    by_name = {r['name']: r for r in rows}
    out = {}
    for name, value in expected.items():
        row = by_name.get(name)
        if row is None:
            out[name] = {'expected': value, 'missing': True, 'ok': False}
        else:
            out[name] = {'expected': value, 'auc': row['auc'], 'ci_lo': row['ci_lo'], 'ci_hi': row['ci_hi'],
                         'missing': False, 'ok': bool(row['ci_lo'] <= value <= row['ci_hi'])}
    return out


def render_markdown(rows, gate=None, reference=None, title='Benchmark'):
    lines = [f'# {title}', '',
             '| Score | Group | AUC | CI | CI level | min within-bin AUC | Notes |',
             '|---|---|---|---|---|---|---|']
    for r in sorted(rows, key=lambda r: -r['auc']):
        lines.append(f"| {r['name']} | {r['group']} | {r['auc']:.3f} | {r['ci_lo']:.3f}–{r['ci_hi']:.3f} | "
                     f"{r['level']:.1%} | {within_bin_min(r):.3f} | {r['notes']} |")
    if reference:
        lines += ['', '## Stage 0 reference check (expected value must lie inside the harness CI)', '',
                  '| Score | Expected | Measured | CI | OK |', '|---|---|---|---|---|']
        for name, ref in reference.items():
            if ref['missing']:
                lines.append(f"| {name} | {ref['expected']:.3f} | missing | | NO |")
            else:
                lines.append(f"| {name} | {ref['expected']:.3f} | {ref['auc']:.3f} | "
                             f"{ref['ci_lo']:.3f}–{ref['ci_hi']:.3f} | {'yes' if ref['ok'] else 'NO'} |")
    if gate:
        lines += ['', '## Gate', '',
                  f"Bar: {gate['bar_name']} (AUC {gate['bar_auc']:.3f}). "
                  f"Result: **{'PASSED' if gate['passed'] else 'NOT PASSED'}**", '',
                  '| Model score | AUC | CI lower | beats bar | min within-bin | within-bin ok | passes |',
                  '|---|---|---|---|---|---|---|']
        for m in gate['models']:
            lines.append(f"| {m['name']} | {m['auc']:.3f} | {m['ci_lo']:.3f} | {m['beats_bar']} | "
                         f"{m['within_bin_min']:.3f} | {m['within_bin_ok']} | {m['passes']} |")
    return '\n'.join(lines) + '\n'


def write_report(rows, gate, reference, json_path, md_path, title, meta=None):
    payload = {'rows': rows, 'gate': gate, 'reference': reference, 'meta': meta or {}}
    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
    with open(md_path, 'w', encoding='utf-8') as handle:
        handle.write(render_markdown(rows, gate, reference, title))
