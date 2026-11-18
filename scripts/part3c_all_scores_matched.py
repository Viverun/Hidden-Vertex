"""Part 3C: all scores under complexity matching (point estimates, no CIs). No training.

Uses the weighting of scripts/part3c_matched_eval.py (background reweighted to the signal's multiplicity
distribution, 20 quantile bins; and to its (multiplicity, HT) distribution, 8x8 quantile bins; weights clipped
at the 99th percentile) on the saved score files: the Part 2 bank (baselines, Part 1 AE), the Part 2 event-level
scores (run tag main), and the Part 3 jet-feature density, WNAE pilot and WNAE long scores. Model rows are
seed-averaged.

    python scripts/part3c_all_scores_matched.py   # -> results/part3/matched_all_scores.{md,json}
"""
import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))
from part3c_matched_eval import N_BINS_1D, N_BINS_2D, VARIANTS, cell_weights, quantile_bins  # noqa: E402

BANK_ROWS = (('-|mjj-3500| (oracle)', '-|mjj-3500 GeV| (oracle)'), ('-tau21 (leading jet)', '-tau21 (leading jet)'))
MODEL_ROWS = (('density', 'knn_jet4', 'kNN, jet4'), ('density', 'gmm_jet4', 'GMM, jet4'),
              ('wnae', 'wnae_jet5', 'WNAE pilot, jet5'), ('density', 'knn_jet5', 'kNN, jet5'),
              ('density', 'gmm_jet5', 'GMM, jet5'), ('wnae_long', 'wnae_jet5', 'WNAE long, jet5'))
BANK_ROWS_2 = (('Part 1 AE (mean)', 'Node-level GAE (Part 1 AE, mean)'), ('mjj', 'mjj'),
               ('mean pT per particle', 'mean pT per particle'), ('HT (sum pT)', 'HT (sum pT)'),
               ('-multiplicity', '-multiplicity'))
AE_ROWS = (('density', 'ae_jet4', 'MLP autoencoder, jet4'), ('density', 'ae_jet5', 'MLP autoencoder, jet5'))
EVENT_ROWS = (('chamfer', 'Event-level GAE: Chamfer'), ('knn', 'Event-level GAE: kNN'),
              ('gmm', 'Event-level GAE: GMM'))


def _load_runs(pattern, y):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    runs = []
    for f in files:
        with np.load(f) as loaded:
            r = {k: loaded[k] for k in loaded.files}
        if not np.array_equal(r['labels'].astype(int), y):
            raise ValueError(f'{f}: labels differ from the bank')
        runs.append(r)
    return runs, [os.path.relpath(f, REPO) for f in files]


def collect_scores(part2_dir, part3_dir):
    """Returns y, multiplicity, HT, ordered {row label: [score arrays]}, {row label: source files}."""
    with np.load(os.path.join(part2_dir, 'bank.npz')) as loaded:
        bank = {k: loaded[k] for k in loaded.files}
    with open(os.path.join(part2_dir, 'bank.json'), encoding='utf-8') as handle:
        names = [m['name'] for m in json.load(handle)]
    y = bank['y'].astype(int)
    scores, sources = {}, {}

    def add_bank(rows):
        for key, label in rows:
            scores[label] = [bank[f's{names.index(key)}'].astype(np.float64)]
            sources[label] = [f'{os.path.relpath(part2_dir, REPO)}/bank.npz:{key}']

    def add_models(rows):
        for sub, key, label in rows:
            runs, files = _load_runs(os.path.join(part3_dir, sub, 'scores_s*.npz'), y)
            scores[label] = [r[key].astype(np.float64) for r in runs]
            sources[label] = [f'{f}:{key}' for f in files]

    add_bank(BANK_ROWS)
    add_models(MODEL_ROWS)
    add_bank(BANK_ROWS_2)
    add_models(AE_ROWS)
    runs, files = _load_runs(os.path.join(part2_dir, 'runs', 'main', 'scores_s*.npz'), y)
    for key, label in EVENT_ROWS:
        scores[label] = [r[key].astype(np.float64) for r in runs]
        sources[label] = [f'{f}:{key}' for f in files]
    return y, bank['cov_multiplicity'], bank['cov_ht'], scores, sources


def run(part2_dir, part3_dir):
    y, mult, ht, scores, sources = collect_scores(part2_dir, part3_dir)
    b_m, n_m = quantile_bins(mult, N_BINS_1D)
    b_m8, n_m8 = quantile_bins(mult, N_BINS_2D)
    b_h8, n_h8 = quantile_bins(ht, N_BINS_2D)
    cells = {'mult': (b_m, n_m), 'mult_ht': (b_m8 * n_h8 + b_h8, n_m8 * n_h8)}
    weights = {'plain': None}
    diag = {}
    for v, (c, n) in cells.items():
        w, d = cell_weights(y, c, n)
        weights[v] = w
        wb = w[y == 0]
        diag[v] = {**d, 'n_cells': int(n), 'bg_effective_sample_size': float(wb.sum() ** 2 / np.sum(wb ** 2))}
    results = {name: {'n_seeds': len(arr), 'sources': sources[name],
                      **{v: float(np.mean([roc_auc_score(y, s, sample_weight=weights[v]) for s in arr]))
                         for v in VARIANTS}}
               for name, arr in scores.items()}
    return {'n_test': int(len(y)), 'n_sig': int(y.sum()),
            'settings': {'n_bins_mult': N_BINS_1D, 'n_bins_2d': N_BINS_2D, 'clip_percentile': 99.0,
                         'confidence_intervals': None},
            'diagnostics': diag, 'results': results}


def render(out):
    d = out['diagnostics']
    lines = ['# Part 3C: all scores under complexity matching', '',
             f"Test set: {out['n_test']} events ({out['n_sig']} signal). Background weighted to the signal "
             f"distribution: multiplicity ({d['mult']['n_cells']} quantile bins) or (multiplicity, HT) "
             f"({out['settings']['n_bins_2d']}x{out['settings']['n_bins_2d']} quantile bins). Weights clipped at "
             'the 99th percentile. Point estimates only, no confidence intervals. Model rows: AUC averaged over '
             'seeds. The oracle row uses the known signal mass. Matched columns use signal information and are '
             'diagnostics.', '',
             f"Background effective sample size (of 5000): mult-matched {d['mult']['bg_effective_sample_size']:.0f}, "
             f"(mult, HT)-matched {d['mult_ht']['bg_effective_sample_size']:.0f}.", '',
             '| Score | seeds | plain | mult-matched | (mult, HT)-matched |', '|---|---|---|---|---|']
    for name, r in out['results'].items():
        lines.append(f"| {name} | {r['n_seeds']} | {r['plain']:.3f} | {r['mult']:.3f} | {r['mult_ht']:.3f} |")
    lines += ['', 'Command: `python scripts/part3c_all_scores_matched.py`']
    return '\n'.join(lines) + '\n'


def main():
    out = run(os.path.join(REPO, 'results', 'part2'), os.path.join(REPO, 'results', 'part3'))
    out_dir = os.path.join(REPO, 'results', 'part3')
    with open(os.path.join(out_dir, 'matched_all_scores.json'), 'w', encoding='utf-8') as handle:
        json.dump(out, handle, indent=2)
    md = render(out)
    with open(os.path.join(out_dir, 'matched_all_scores.md'), 'w', encoding='utf-8') as handle:
        handle.write(md)
    print(md)


if __name__ == '__main__':
    main()
