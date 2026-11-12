"""Part 3C (C1): evaluation-side complexity matching. No training.

For each model score (seed-averaged AUC) and three particle baselines, on the frozen Part 2 test set:
  plain AUC; AUC with background weighted so its multiplicity distribution matches the signal's (20 quantile
  bins of multiplicity over the test set); AUC with joint (multiplicity, HT) matching (8x8 quantile bins);
  95% percentile-bootstrap CI (1000 resamples of events; the weights are recomputed in each resample with
  the bin edges of the full test set). Background weights are clipped at their 99th percentile.

    python scripts/part3c_matched_eval.py                       # run tag main -> results/part3/c1_matched.{md,json}
    python scripts/part3c_matched_eval.py --tag multmatch --name c1_matched_multmatch

Fixed prediction (C1): if complexity alone explains the kNN/GMM inversion, their matched AUC is within
0.5 +/- 0.03; if it stays below 0.47, complexity (as measured by multiplicity and HT) does not fully explain it.
"""
import argparse
import glob
import json
import os

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORE_NAMES = ('chamfer', 'knn', 'gmm')
BASELINES = ('mean pT per particle', 'HT (sum pT)', '-multiplicity')
CLIP_PERCENTILE = 99.0
N_BINS_1D = 20
N_BINS_2D = 8
VARIANTS = ('plain', 'mult', 'mult_ht')


def quantile_bins(x, n_bins):
    """Bin id of each value in quantile bins of x (edges from the whole array; ties merge bins)."""
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    n = max(len(edges) - 1, 1)
    return np.clip(np.searchsorted(edges, x, side='right') - 1, 0, n - 1), n


def cell_weights(y, cell, n_cells, clip_percentile=CLIP_PERCENTILE):
    """Signal weight 1; background weight p_sig(cell) / p_bg(cell), clipped at the background-weight
    percentile. Background in cells without signal gets weight 0."""
    y = np.asarray(y).astype(int)
    n_sig_c = np.bincount(cell[y == 1], minlength=n_cells)
    n_bg_c = np.bincount(cell[y == 0], minlength=n_cells)
    p_sig = n_sig_c / max(n_sig_c.sum(), 1)
    p_bg = n_bg_c / max(n_bg_c.sum(), 1)
    ratio = np.divide(p_sig, p_bg, out=np.zeros(n_cells), where=n_bg_c > 0)
    w = np.ones(len(y))
    w_bg = ratio[cell[y == 0]]
    clip = float(np.percentile(w_bg, clip_percentile))
    w[y == 0] = np.minimum(w_bg, clip)
    unmatched_sig = float(n_sig_c[n_bg_c == 0].sum() / max(n_sig_c.sum(), 1))
    return w, {'clip_value': clip, 'signal_fraction_in_cells_without_background': unmatched_sig}


def all_weights(y, cells):
    """{variant: weights or None} for the given resample; cells = {variant: (cell ids, n_cells)}."""
    out = {'plain': None}
    for v in ('mult', 'mult_ht'):
        out[v] = cell_weights(y, *cells[v])[0]
    return out


def score_aucs(y, scores, weights):
    """{name: {variant: seed-averaged AUC}}; scores = {name: [arrays]}."""
    return {name: {v: float(np.mean([roc_auc_score(y, s, sample_weight=w) for s in arrays]))
                   for v, w in weights.items()}
            for name, arrays in scores.items()}


def _boot_once(seed_seq, y, scores, cells):
    rng = np.random.default_rng(seed_seq)
    idx = rng.integers(0, len(y), len(y))
    yb = y[idx]
    if yb.min() == yb.max():
        return None
    cb = {v: (c[idx], n) for v, (c, n) in cells.items()}
    return score_aucs(yb, {k: [s[idx] for s in arr] for k, arr in scores.items()}, all_weights(yb, cb))


def load_inputs(output_dir, bank_path, tag):
    with np.load(bank_path) as loaded:
        bank = {k: loaded[k] for k in loaded.files}
    with open(bank_path.replace('.npz', '.json'), encoding='utf-8') as handle:
        meta = json.load(handle)
    y = bank['y'].astype(int)
    names = [m['name'] for m in meta]
    scores = {}
    files = sorted(glob.glob(os.path.join(output_dir, 'runs', tag, 'scores_s*.npz')))
    if not files:
        raise FileNotFoundError(f'no scores for run tag {tag!r}')
    runs = []
    for f in files:
        with np.load(f) as loaded:
            runs.append({k: loaded[k] for k in loaded.files})
        if not np.array_equal(runs[-1]['labels'].astype(int), y):
            raise ValueError(f'{f}: labels differ from the bank')
    rows = [r['rows'] for r in runs if 'rows' in r]
    if rows and not all(np.array_equal(rows[0], r) for r in rows):
        raise ValueError(f'run {tag}: test rows differ between seeds')
    for score in SCORE_NAMES:
        scores[f'EventAE {score}' + ('' if tag == 'main' else f' [{tag}]')] = [r[score].astype(np.float64)
                                                                             for r in runs]
    for b in BASELINES:
        scores[b] = [bank[f's{names.index(b)}'].astype(np.float64)]
    info = {'tag': tag, 'seed_files': [os.path.relpath(f, REPO) for f in files],
            'rows_in_score_files': bool(rows), 'n_test': int(len(y)), 'n_sig': int(y.sum())}
    return y, bank['cov_multiplicity'], bank['cov_ht'], scores, info


def weighted_mean(x, w):
    return float(np.sum(x * w) / np.sum(w))


def run(output_dir, bank_path, tag, n_boot=1000, n_jobs=-1, seed=0):
    y, mult, ht, scores, info = load_inputs(output_dir, bank_path, tag)
    b_m, n_m = quantile_bins(mult, N_BINS_1D)
    b_m8, n_m8 = quantile_bins(mult, N_BINS_2D)
    b_h8, n_h8 = quantile_bins(ht, N_BINS_2D)
    cells = {'mult': (b_m, n_m), 'mult_ht': (b_m8 * n_h8 + b_h8, n_m8 * n_h8)}
    weights = all_weights(y, cells)
    point = score_aucs(y, scores, weights)

    children = np.random.SeedSequence(seed).spawn(n_boot)
    boots = Parallel(n_jobs=n_jobs)(delayed(_boot_once)(c, y, scores, cells) for c in children)
    boots = [b for b in boots if b is not None]
    results = {}
    for name in scores:
        results[name] = {'n_seeds': len(scores[name])}
        for v in VARIANTS:
            vals = np.array([b[name][v] for b in boots])
            results[name][v] = {'auc': point[name][v], 'ci_lo': float(np.quantile(vals, 0.025)),
                                'ci_hi': float(np.quantile(vals, 0.975))}

    diag = {}
    bg = y == 0
    for v in ('mult', 'mult_ht'):
        w, d = cell_weights(y, *cells[v])
        wb = w[bg]
        diag[v] = {**d, 'n_cells': int(cells[v][1]), 'bg_effective_sample_size': float(wb.sum() ** 2 / np.sum(wb ** 2)),
                   'bg_fraction_zero_weight': float(np.mean(wb == 0)),
                   'bg_mean_multiplicity_weighted': weighted_mean(mult[bg], wb),
                   'bg_mean_ht_weighted': weighted_mean(ht[bg], wb)}
    diag['unweighted'] = {'bg_mean_multiplicity': float(mult[bg].mean()), 'sig_mean_multiplicity': float(mult[~bg].mean()),
                          'bg_mean_ht': float(ht[bg].mean()), 'sig_mean_ht': float(ht[~bg].mean())}
    settings = {'n_bins_mult': N_BINS_1D, 'n_bins_2d': N_BINS_2D, 'clip_percentile': CLIP_PERCENTILE,
                'n_boot': n_boot, 'n_boot_valid': len(boots), 'ci_level': 0.95, 'bootstrap_seed': seed,
                'weights_recomputed_per_resample': True}
    return {'info': info, 'settings': settings, 'diagnostics': diag, 'results': results}


def verdict(auc):
    if abs(auc - 0.5) <= 0.03:
        return 'within 0.5 +/- 0.03'
    if auc < 0.47:
        return 'below 0.47'
    return 'above 0.53'


def render(out):
    r, d, s = out['results'], out['diagnostics'], out['settings']
    lines = [f"# Part 3C / C1: complexity-matched AUC (run tag {out['info']['tag']})", '',
             f"Test set: {out['info']['n_test']} events ({out['info']['n_sig']} signal). Background weighted to the "
             f"signal distribution: (b) {s['n_bins_mult']} quantile bins of multiplicity; (c) {s['n_bins_2d']}x"
             f"{s['n_bins_2d']} quantile bins of (multiplicity, HT). Weights clipped at the {s['clip_percentile']:g}th "
             f"percentile. 95% CI: percentile bootstrap, {s['n_boot_valid']} resamples of events, weights recomputed "
             'in each resample. Model rows: AUC averaged over seeds.', '',
             '| Score | seeds | (a) plain AUC [95% CI] | (b) mult-matched [95% CI] | (c) (mult, HT)-matched [95% CI] |',
             '|---|---|---|---|---|']
    for name, row in r.items():
        cells = ' | '.join(f"{row[v]['auc']:.3f} [{row[v]['ci_lo']:.3f}, {row[v]['ci_hi']:.3f}]" for v in VARIANTS)
        lines.append(f"| {name} | {row['n_seeds']} | {cells} |")
    lines += ['', '## Weight diagnostics', '',
              '| Matching | cells | clip value | bg ESS (of 5000) | bg with weight 0 | signal in cells without bg | '
              'weighted bg mean mult | weighted bg mean HT |', '|---|---|---|---|---|---|---|---|']
    for v in ('mult', 'mult_ht'):
        x = d[v]
        lines.append(f"| {v} | {x['n_cells']} | {x['clip_value']:.3f} | {x['bg_effective_sample_size']:.0f} | "
                     f"{x['bg_fraction_zero_weight']:.1%} | {x['signal_fraction_in_cells_without_background']:.1%} | "
                     f"{x['bg_mean_multiplicity_weighted']:.1f} | {x['bg_mean_ht_weighted']:.0f} |")
    u = d['unweighted']
    lines += ['', f"Unweighted means: background multiplicity {u['bg_mean_multiplicity']:.1f}, signal "
              f"{u['sig_mean_multiplicity']:.1f}; background HT {u['bg_mean_ht']:.0f}, signal {u['sig_mean_ht']:.0f}.",
              '', '## Fixed prediction (kNN and GMM)', '',
              'If complexity alone explains the kNN/GMM inversion, their matched AUC is within 0.5 +/- 0.03; if it '
              'stays below 0.47, complexity (as measured by multiplicity and HT) does not fully explain it.', '',
              '| Score | (b) mult-matched | (c) (mult, HT)-matched |', '|---|---|---|']
    for name, row in r.items():
        if 'knn' in name or 'gmm' in name:
            lines.append(f"| {name} | {row['mult']['auc']:.3f}: {verdict(row['mult']['auc'])} | "
                         f"{row['mult_ht']['auc']:.3f}: {verdict(row['mult_ht']['auc'])} |")
    if not out['info']['rows_in_score_files']:
        lines += ['', 'Note: the score files hold no `rows` key; alignment with the bank is checked through the '
                  'labels only (same frozen test set, same order).']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='results/part2', help='Part 2 output dir (runs/, bank.npz)')
    parser.add_argument('--tag', default='main')
    parser.add_argument('--name', default='c1_matched')
    parser.add_argument('--out-dir', default='results/part3')
    parser.add_argument('--n-boot', type=int, default=1000)
    args = parser.parse_args()
    output_dir = os.path.join(REPO, args.output_dir)
    out = run(output_dir, os.path.join(output_dir, 'bank.npz'), args.tag, n_boot=args.n_boot)
    out_dir = os.path.join(REPO, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f'{args.name}.json'), 'w', encoding='utf-8') as handle:
        json.dump(out, handle, indent=2)
    md = render(out)
    with open(os.path.join(out_dir, f'{args.name}.md'), 'w', encoding='utf-8') as handle:
        handle.write(md)
    print(md)


if __name__ == '__main__':
    main()
