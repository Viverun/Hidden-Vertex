"""Part 3d: CWoLa-style bump hunt (weak supervision). The protocol is fixed in configs/part3_cwola.yaml.

Labels are used only (a) to choose which signal rows are injected into the unlabeled pool and (b) to evaluate
the frozen test set in `make_report`. Every training target is signal-region (SR) vs sideband (SB) membership in
mjj. Windows, merges and quantile references use only the mjj and features of pool events.

Interpretations of the config (fixed before any CWoLa test score was computed):
  * windows: for a grid with edges e0..eN there are N windows. Window 0 has SR (-inf, e1) and one sideband
    [e1, e1 + 0.4); window N-1 has SR [e(N-1), +inf) and one sideband [e(N-1) - 0.4, e(N-1)); every other window
    has SR [e_i, e_i+1) and sidebands of `sideband` width on each side. All intervals are half-open [lo, hi).
  * merge: if the SR or the SB of a window has fewer than `min_events_per_class` events in the TRAINING part of
    the pool, the window is merged with its inner neighbour (towards the middle of the grid). The merged SR is the
    union; its sidebands are recomputed just outside the merged SR (one 0.4 sideband if the SR is open-ended).
    Repeat until every window passes.
  * quantile = (number of reference SR events of the window with a strictly lower ensemble output) / (number of
    reference SR events of the window).
  * ensemble member k of run seed s uses random_state = 1000 * s + k (this seeds the early-stopping split).
  * test-set metrics use the Part 2 bank (results/part2/bank.npz) for the HT / multiplicity / leading-pT
    covariates and the particle-level bar row, 10 quantile bins and a 95% percentile bootstrap CI (as the Part 3B
    WNAE rows), and the Part 3 SIC definition (max over FPR >= 1e-3).
"""
import json
import os
import time

import numpy as np
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from src.data.jet_features import feature_matrix, load_features, mjj

GEV_PER_TEV = 1000.0
OPEN_SIDEBAND = 0.4                        # sideband width of an open-ended outer window (TeV)
BANK_PATH = 'results/part2/bank.npz'
COVARIATES = ('ht', 'multiplicity', 'max_pt')
N_BINS = 10
CI_LEVEL = 0.95
SIC_MIN_FPR = 1e-3
SIC_FPRS = (0.01, 0.1)


def load_config(path='configs/part3_cwola.yaml'):
    with open(path, 'r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def frac_tag(fraction):
    return f'{float(fraction):g}'


# ----------------------------------------------------------------------------------------------- windows
def sidebands(lo, hi, sideband, open_sideband=OPEN_SIDEBAND):
    """Sidebands just outside SR [lo, hi). An open-ended SR gets one sideband of `open_sideband` on its inner side."""
    if np.isinf(lo) and np.isinf(hi):
        raise ValueError('a signal region open on both sides has no sideband')
    if np.isinf(lo):
        return [(hi, hi + open_sideband)]
    if np.isinf(hi):
        return [(lo - open_sideband, lo)]
    return [(lo - sideband, lo), (hi, hi + sideband)]


def make_window(lo, hi, sideband, open_sideband=OPEN_SIDEBAND):
    return {'sr': (float(lo), float(hi)), 'sb': [tuple(map(float, s)) for s in sidebands(lo, hi, sideband,
                                                                                          open_sideband)]}


def grid_windows(edges, width, sideband, open_sideband=OPEN_SIDEBAND):
    """Windows of one grid before merging (see the module docstring)."""
    edges = [float(e) for e in edges]
    if not np.allclose(np.diff(edges), width):
        raise ValueError(f'grid edges must be spaced by the window width {width}: {edges}')
    n = len(edges) - 1
    out = []
    for i in range(n):
        lo = -np.inf if i == 0 else edges[i]
        hi = np.inf if i == n - 1 else edges[i + 1]
        out.append(make_window(lo, hi, sideband, open_sideband))
    return out


def in_intervals(m, intervals):
    m = np.asarray(m, dtype=np.float64)
    mask = np.zeros(len(m), bool)
    for lo, hi in intervals:
        mask |= (m >= lo) & (m < hi)
    return mask


def sr_mask(m, window):
    return in_intervals(m, [window['sr']])


def sb_mask(m, window):
    return in_intervals(m, window['sb'])


def window_counts(m, window):
    return int(sr_mask(m, window).sum()), int(sb_mask(m, window).sum())


def merge_windows(windows, m_train, min_events, sideband, open_sideband=OPEN_SIDEBAND):
    """Merge windows whose SR or SB has < min_events training events with their inner neighbour (label-free)."""
    windows = [dict(w) for w in windows]
    log = []
    while len(windows) > 1:
        bad = [i for i, w in enumerate(windows) if min(window_counts(m_train, w)) < min_events]
        if not bad:
            break
        i = bad[0]
        j = i + 1 if i < len(windows) / 2 - 0.5 else i - 1     # inner neighbour (towards the middle)
        a, b = sorted((i, j))
        lo, hi = windows[a]['sr'][0], windows[b]['sr'][1]
        log.append({'merged': [list(windows[a]['sr']), list(windows[b]['sr'])], 'counts': list(window_counts(
            m_train, windows[i])), 'into': [lo, hi]})
        windows[a:b + 1] = [make_window(lo, hi, sideband, open_sideband)]
    return windows, log


def assign_window(m, windows):
    """Index of the window whose SR holds each mjj value; every value must be in exactly one SR."""
    masks = np.stack([sr_mask(m, w) for w in windows])
    if not np.all(masks.sum(axis=0) == 1):
        raise AssertionError('signal regions must cover the mjj axis exactly once')
    return masks.argmax(axis=0)


# ----------------------------------------------------------------------------------------------- scoring
def quantile_scores(ref_out, test_out):
    """Fraction of reference outputs strictly lower than each test output (in [0, 1])."""
    ref = np.sort(np.asarray(ref_out, dtype=np.float64))
    if len(ref) == 0:
        raise ValueError('empty reference sample')
    return np.searchsorted(ref, np.asarray(test_out, dtype=np.float64), side='left') / len(ref)


def train_ensemble(x, target, ccfg, seed):
    if ccfg['kind'] != 'HistGradientBoostingClassifier':
        raise ValueError(f"unsupported classifier {ccfg['kind']!r}")
    models = []
    for k in range(int(ccfg['ensemble'])):
        clf = HistGradientBoostingClassifier(
            max_iter=ccfg['max_iter'], learning_rate=ccfg['learning_rate'], max_leaf_nodes=ccfg['max_leaf_nodes'],
            early_stopping=ccfg['early_stopping'], validation_fraction=ccfg['validation_fraction'],
            n_iter_no_change=ccfg['n_iter_no_change'], class_weight=ccfg['class_weight'],
            random_state=1000 * int(seed) + k)
        models.append(clf.fit(x, target))
    return models


def ensemble_predict(models, x):
    if len(x) == 0:
        return np.zeros(0)
    return np.mean([mdl.predict_proba(x)[:, 1] for mdl in models], axis=0)


def _window_classifier(x_train, m_train, window, ccfg, seed):
    sr, sb = sr_mask(m_train, window), sb_mask(m_train, window)
    if (sr & sb).any():
        raise AssertionError('SR and SB overlap')
    sel = sr | sb
    models = train_ensemble(x_train[sel], sr[sel].astype(int), ccfg, seed)
    return models, {'n_sr_train': int(sr.sum()), 'n_sb_train': int(sb.sum()),
                    'n_iter': [int(mdl.n_iter_) for mdl in models]}


def cwola_scores(x_train, m_train, x_ref, m_ref, x_test, m_test, cfg, seed, log=print):
    """Core CWoLa procedure on arrays (mjj in TeV). Returns test scores and label-free run info."""
    wcfg, ccfg = cfg['windows'], cfg['classifier']
    grids, info = {}, {'grids': {}}
    for name in ('a', 'b'):
        windows = grid_windows(wcfg[f'grid_{name}_edges'], wcfg['width'], wcfg['sideband'])
        windows, merges = merge_windows(windows, m_train, wcfg['min_events_per_class'], wcfg['sideband'])
        w_test, w_ref = assign_window(m_test, windows), assign_window(m_ref, windows)
        q = np.full(len(m_test), np.nan)
        winfo = []
        for i, w in enumerate(windows):
            started = time.time()
            models, meta = _window_classifier(x_train, m_train, w, ccfg, seed)
            ref_out = ensemble_predict(models, x_ref[w_ref == i])
            sel = w_test == i
            q[sel] = quantile_scores(ref_out, ensemble_predict(models, x_test[sel]))
            meta.update({'sr': list(w['sr']), 'sb': [list(s) for s in w['sb']], 'n_ref_sr': int(len(ref_out)),
                         'n_test_sr': int(sel.sum()), 'seconds': round(time.time() - started, 1)})
            winfo.append(meta)
            log(f"  grid {name} SR [{w['sr'][0]:.1f}, {w['sr'][1]:.1f}): train SR/SB {meta['n_sr_train']}/"
                f"{meta['n_sb_train']}, iters {min(meta['n_iter'])}-{max(meta['n_iter'])}, {meta['seconds']} s")
        if np.isnan(q).any():
            raise AssertionError('a test event got no quantile')
        grids[name] = q
        info['grids'][name] = {'windows': winfo, 'merges': merges}
    o = cfg['secondary']['oracle_window']
    oracle = {'sr': tuple(o['sr']), 'sb': [tuple(s) for s in o['sb']]}
    started = time.time()
    models, meta = _window_classifier(x_train, m_train, oracle, ccfg, seed)
    oracle_score = ensemble_predict(models, x_test)
    meta.update({'sr': list(oracle['sr']), 'sb': [list(s) for s in oracle['sb']],
                 'seconds': round(time.time() - started, 1)})
    info['oracle'] = meta
    return {'score': 0.5 * (grids['a'] + grids['b']), 'score_a': grids['a'], 'score_b': grids['b'],
            'oracle_score': oracle_score}, info


# ----------------------------------------------------------------------------------------------- rows
def _rng_range(r):
    return np.arange(int(r[0]), int(r[1]))


def test_rows(cfg):
    d = cfg['data']
    return np.r_[_rng_range(d['test_bg_rows']), _rng_range(d['test_sig_rows'])]


def injected_rows(cfg, fraction, seed):
    """Signal rows injected for (fraction, seed). n = round(fraction * n_pool_bg). Nested across fractions."""
    d = cfg['data']
    n_bg = int(d['pool_bg_rows'][1]) - int(d['pool_bg_rows'][0])
    n_sig = int(round(float(fraction) * n_bg))
    cand = _rng_range(d['inject_sig_rows'])
    if n_sig > len(cand):
        raise ValueError('not enough signal rows to inject')
    perm = np.random.default_rng([int(seed), 1]).permutation(len(cand))
    return np.sort(cand[perm[:n_sig]])


def pool_rows(cfg, fraction, seed):
    rows = np.r_[_rng_range(cfg['data']['pool_bg_rows']), injected_rows(cfg, fraction, seed)]
    if np.intersect1d(rows, test_rows(cfg)).size:
        raise AssertionError('test rows must never enter the pool')
    return rows


def split_pool(n, train_fraction, seed):
    """Boolean mask: True = training part, False = reference sample."""
    perm = np.random.default_rng([int(seed), 2]).permutation(n)
    mask = np.zeros(n, bool)
    mask[perm[:int(round(train_fraction * n))]] = True
    return mask


def check_labels(labels, cfg):
    """The file must be sorted: background rows have label 0 and signal rows label 1."""
    d = cfg['data']
    for key, want in (('test_bg_rows', 0), ('pool_bg_rows', 0), ('test_sig_rows', 1), ('inject_sig_rows', 1)):
        lo, hi = d[key]
        if not np.all(labels[lo:hi] == want):
            raise AssertionError(f'{key} rows do not all have label {want}')


# ----------------------------------------------------------------------------------------------- run
def load_arrays(cfg):
    df = load_features(cfg['data']['features_file'])
    feats = tuple(cfg['data']['features'])
    if feats != ('mj1', 'mj2', 'tau21_j1', 'tau21_j2'):
        raise ValueError(f'config features {feats} are not jet4')
    x = feature_matrix(df, 'jet4')
    m = mjj(df) / GEV_PER_TEV
    labels = df['label'].to_numpy().astype(int)
    check_labels(labels, cfg)
    return x, m, labels


def out_dir(cfg):
    return cfg['output']['dir']


def run_one(cfg, fraction, seed, arrays=None, threads=None, log=print):
    if threads:
        from threadpoolctl import threadpool_limits
        threadpool_limits(int(threads))
    x, m, labels = arrays if arrays is not None else load_arrays(cfg)
    started = time.time()
    t_rows, p_rows = test_rows(cfg), pool_rows(cfg, fraction, seed)
    train = split_pool(len(p_rows), cfg['split']['train_fraction'], seed)
    tr, ref = p_rows[train], p_rows[~train]
    for part in (tr, ref):
        assert not np.intersect1d(part, t_rows).size
    log(f'fraction {frac_tag(fraction)} seed {seed}: pool {len(p_rows)}, train {len(tr)}, ref {len(ref)}')
    scores, info = cwola_scores(x[tr], m[tr], x[ref], m[ref], x[t_rows], m[t_rows], cfg, seed, log=log)
    inj = injected_rows(cfg, fraction, seed)
    # For information only (uses labels of the pool): S and B in the signal regions near 3.5 TeV.
    near = {}
    for name, (lo, hi) in {'grid_a_3.2-3.6': (3.2, 3.6), 'grid_b_3.4-3.8': (3.4, 3.8),
                           'oracle_3.3-3.7': (3.3, 3.7)}.items():
        sel = (m[p_rows] >= lo) & (m[p_rows] < hi)
        s, b = int(labels[p_rows][sel].sum()), int((labels[p_rows][sel] == 0).sum())
        near[name] = {'S': s, 'B': b, 'S/B': s / b if b else float('nan'),
                      'S/sqrt(B)': s / np.sqrt(b) if b else float('nan')}
    info.update({'fraction': float(fraction), 'seed': int(seed), 'n_pool': int(len(p_rows)),
                 'n_pool_bg': int(len(p_rows) - len(inj)), 'n_injected': int(len(inj)),
                 'n_train': int(len(tr)), 'n_ref': int(len(ref)), 'n_injected_train': int(np.isin(inj, tr).sum()),
                 'window_counts_info': near, 'seconds_total': round(time.time() - started, 1)})
    d = out_dir(cfg)
    os.makedirs(d, exist_ok=True)
    tag = f'f{frac_tag(fraction)}_s{seed}'
    tmp = os.path.join(d, f'scores_{tag}.tmp.npz')
    np.savez(tmp, labels=labels[t_rows], rows=t_rows, mjj=m[t_rows], **scores)
    os.replace(tmp, os.path.join(d, f'scores_{tag}.npz'))
    with open(os.path.join(d, f'metrics_{tag}.json'), 'w', encoding='utf-8') as handle:
        json.dump(info, handle, indent=2)
    log(f'fraction {frac_tag(fraction)} seed {seed}: done in {info["seconds_total"] / 60:.1f} min')
    return info


def _worker(args):
    cfg, fraction, seed, threads = args
    from threadpoolctl import threadpool_limits
    with threadpool_limits(int(threads)):
        x, m, labels = load_arrays(cfg)
        return run_one(cfg, fraction, seed, arrays=(x, m, labels),
                       log=lambda s: print(f'[f{frac_tag(fraction)} s{seed}] {s}', flush=True))


def run_all(cfg, workers=6, threads=2, skip_existing=True, log=print):
    from concurrent.futures import ProcessPoolExecutor
    jobs = []
    for f in cfg['injection']['fractions']:
        for s in cfg['injection']['seeds']:
            path = os.path.join(out_dir(cfg), f'scores_f{frac_tag(f)}_s{s}.npz')
            if skip_existing and os.path.exists(path):
                log(f'f{frac_tag(f)} s{s}: exists, skipped')
                continue
            jobs.append((cfg, f, s, threads))
    # Largest fractions first is not needed; run order does not change any result (every run is seeded).
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for info in ex.map(_worker, jobs):
            log(f"finished f{frac_tag(info['fraction'])} s{info['seed']} ({info['seconds_total'] / 60:.1f} min)")


# ----------------------------------------------------------------------------------------------- report
def load_bank(path=BANK_PATH):
    with np.load(path) as loaded:
        bank = {k: loaded[k] for k in loaded.files}
    with open(path.replace('.npz', '.json'), encoding='utf-8') as handle:
        meta = json.load(handle)
    return bank, meta


def predictions_verdicts(seed_means, fractions, primary_gate):
    """The four pre-registered predictions of configs/part3_cwola.yaml."""
    out = []
    a0 = seed_means[0.0]
    out.append({'text': 'fraction 0.0: seed-mean AUC inside [0.47, 0.53]', 'value': a0,
                'passed': bool(0.47 <= a0 <= 0.53)})
    fs = sorted(fractions)
    drops = [(fs[i], fs[i + 1], seed_means[fs[i + 1]] - seed_means[fs[i]]) for i in range(len(fs) - 1)]
    worst = min(drops, key=lambda d: d[2])
    out.append({'text': 'seed-mean AUC does not decrease as the fraction increases (tolerance 0.01)',
                'value': {'worst_step': [worst[0], worst[1]], 'change': worst[2]},
                'passed': bool(all(d[2] >= -0.01 for d in drops))})
    out.append({'text': 'fraction 0.001 (primary): gate passed (CI lower bound > 0.811 and min within-bin >= 0.6)',
                'value': {'ci_lo': primary_gate['ci_lo'], 'within_bin_min': primary_gate['within_bin_min']},
                'passed': bool(primary_gate['passes'])})
    big = {f: seed_means[f] for f in fs if f >= 0.005}
    out.append({'text': 'fraction >= 0.005: seed-mean AUC > 0.90', 'value': big,
                'passed': bool(all(v > 0.90 for v in big.values()))})
    return out


def make_report(cfg, bank_path=BANK_PATH, log=print):
    from src.benchmarks.report import benchmark_row, evaluate_gate, within_bin_min
    from src.pipeline.part3 import sic_summary
    ecfg = cfg['evaluation']
    bank, meta = load_bank(bank_path)
    y = bank['y'].astype(int)
    cov = {c: bank[f'cov_{c}'] for c in COVARIATES}
    names = [mm['name'] for mm in meta]
    bar_rows = [benchmark_row(mm['name'], y, bank[f's{i}'], cov, mm['group'], CI_LEVEL, ecfg['bootstrap'], N_BINS,
                              mm['notes']) for i, mm in enumerate(meta) if mm['group'] == 'particle']
    d = out_dir(cfg)
    fractions = [float(f) for f in cfg['injection']['fractions']]
    seeds = list(cfg['injection']['seeds'])
    t_rows = test_rows(cfg)
    results, missing = {}, []
    for f in fractions:
        runs, infos = {}, {}
        for s in seeds:
            tag = f'f{frac_tag(f)}_s{s}'
            p = os.path.join(d, f'scores_{tag}.npz')
            if not os.path.exists(p):
                missing.append(tag)
                continue
            with np.load(p) as loaded:
                runs[s] = {k: loaded[k] for k in loaded.files}
            with open(os.path.join(d, f'metrics_{tag}.json'), encoding='utf-8') as handle:
                infos[s] = json.load(handle)
            r = runs[s]
            if not (np.array_equal(r['labels'].astype(int), y) and np.array_equal(r['rows'], t_rows)):
                raise AssertionError(f'{tag}: test rows/labels differ from the bank')
            if not np.allclose(r['mjj'] * GEV_PER_TEV, bank[f"s{names.index('mjj')}"]):
                raise AssertionError(f'{tag}: test mjj differs from the bank mjj column: rows misaligned')
        if runs:
            results[f] = (runs, infos)
    if missing:
        raise FileNotFoundError(f'missing runs: {missing}')

    table, seed_means = [], {}
    for f, (runs, infos) in results.items():
        entry = {'fraction': f, 'n_injected': {s: infos[s]['n_injected'] for s in seeds}}
        for key, label in (('score', 'CWoLa'), ('oracle_score', 'CWoLa oracle window (uses known mass)')):
            arrays = [runs[s][key] for s in seeds]
            row = benchmark_row(f'{label} f={frac_tag(f)}', y, arrays, cov, 'model', CI_LEVEL, ecfg['bootstrap'],
                                N_BINS, f'{len(seeds)} seed(s)')
            row['sic'] = sic_summary(y, arrays, SIC_MIN_FPR, SIC_FPRS)
            gate = evaluate_gate([*bar_rows, row], ecfg['min_within_bin_auc'])
            if abs(gate['bar_auc'] - ecfg['gate_bar_auc']) > 5e-4:
                raise AssertionError(f"bar AUC {gate['bar_auc']:.4f} differs from the config {ecfg['gate_bar_auc']}")
            entry[key] = {'row': row, 'per_seed_auc': {s: float(roc_auc_score(y, runs[s][key])) for s in seeds},
                          'within_bin_min': within_bin_min(row), 'gate': gate['models'][0], 'bar_auc': gate['bar_auc'],
                          'within_bin_ranges': {c: ([min(b['auc'] for b in bins), max(b['auc'] for b in bins)]
                                                    if bins else None) for c, bins in row['within_bin'].items()}}
        entry['grid_auc'] = {g: float(np.mean([roc_auc_score(y, runs[s][f'score_{g}']) for s in seeds]))
                             for g in ('a', 'b')}
        entry['near_3p5'] = {s: infos[s]['window_counts_info'] for s in seeds}
        entry['windows'] = {g: [{'sr': w['sr'], 'sb': w['sb'], 'n_sr_train': w['n_sr_train'],
                                 'n_sb_train': w['n_sb_train']} for w in infos[seeds[0]]['grids'][g]['windows']]
                            for g in ('a', 'b')}
        entry['merges'] = {s: {g: infos[s]['grids'][g]['merges'] for g in ('a', 'b')} for s in seeds}
        entry['minutes'] = {s: infos[s]['seconds_total'] / 60 for s in seeds}
        seed_means[f] = entry['score']['row']['auc']
        table.append(entry)
    primary = next(e for e in table if e['fraction'] == float(cfg['injection']['primary_fraction']))
    preds = predictions_verdicts(seed_means, fractions, primary['score']['gate'])
    payload = {'config': cfg, 'bar_auc': primary['score']['bar_auc'], 'ci_level': CI_LEVEL, 'n_bins': N_BINS,
               'table': table, 'predictions': preds}
    with open(cfg['output']['json'], 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    with open(cfg['output']['report'], 'w', encoding='utf-8') as handle:
        handle.write(render_markdown(cfg, table, preds, primary['score']['bar_auc'], seeds))
    log(f"cwola report written: {cfg['output']['report']}")
    return payload


def _fmt_iv(iv):
    lo, hi = iv
    a = '-inf' if lo == -np.inf else f'{lo:.1f}'
    b = '+inf' if hi == np.inf else f'{hi:.1f}'
    return f'[{a}, {b})' if lo != -np.inf else f'({a}, {b})'


def render_markdown(cfg, table, preds, bar_auc, seeds):
    L = ['# Part 3d: CWoLa bump hunt (weak supervision)', '',
         'This study uses the protocol in `configs/part3_cwola.yaml`. The protocol was fixed before we computed a '
         'test score. Training uses no labels. Each classifier learns to separate events in an mjj signal region '
         'from events in the sidebands. Labels choose the injected signal events and evaluate the frozen test set '
         '(5,000 background + 5,000 signal). Test events never enter the pool, the training data or the reference '
         'sample.', '',
         f'Score: the mean of two quantiles (grid A and grid B). Higher = more anomalous. Bar: mean pT per particle '
         f'(AUC {bar_auc:.3f}). CI: {CI_LEVEL:.0%} percentile bootstrap of the seed-mean AUC '
         f"({cfg['evaluation']['bootstrap']} resamples). Within-bin: {N_BINS} quantile bins of HT, multiplicity and "
         f"leading pT. Gate: CI lower bound > {cfg['evaluation']['gate_bar_auc']} and min within-bin AUC >= "
         f"{cfg['evaluation']['min_within_bin_auc']}. The gate is judged at fraction "
         f"{cfg['injection']['primary_fraction']}.", '',
         '## Main result (model-independent windows)', '',
         '| Fraction S/B | injected S (per seed) | per-seed AUC | seed-mean AUC | 95% CI | min within-bin AUC | '
         'grid A / grid B AUC | max SIC | gate |', '|---|---|---|---|---|---|---|---|---|']
    for e in table:
        r, g = e['score']['row'], e['score']['gate']
        ps = ' / '.join(f'{e["score"]["per_seed_auc"][s]:.3f}' for s in seeds)
        ninj = ' / '.join(str(e['n_injected'][s]) for s in seeds)
        L.append(f"| {frac_tag(e['fraction'])} | {ninj} | {ps} | {r['auc']:.3f} | {r['ci_lo']:.3f}–{r['ci_hi']:.3f} | "
                 f"{e['score']['within_bin_min']:.3f} | {e['grid_auc']['a']:.3f} / {e['grid_auc']['b']:.3f} | "
                 f"{r['sic']['mean']['max_sic']:.2f} | {'PASSED' if g['passes'] else 'NOT PASSED'} |")
    L += ['', '## Oracle window (uses the known signal mass: SR 3.3–3.7 TeV; not model independent)', '',
          'One classifier per run. It scores all test events. The score is the raw ensemble output.', '',
          '| Fraction S/B | per-seed AUC | seed-mean AUC | 95% CI | min within-bin AUC | max SIC | gate (info only) |',
          '|---|---|---|---|---|---|---|']
    for e in table:
        r, g = e['oracle_score']['row'], e['oracle_score']['gate']
        ps = ' / '.join(f'{e["oracle_score"]["per_seed_auc"][s]:.3f}' for s in seeds)
        L.append(f"| {frac_tag(e['fraction'])} | {ps} | {r['auc']:.3f} | {r['ci_lo']:.3f}–{r['ci_hi']:.3f} | "
                 f"{e['oracle_score']['within_bin_min']:.3f} | {r['sic']['mean']['max_sic']:.2f} | "
                 f"{'passed' if g['passes'] else 'not passed'} |")
    L += ['', '## Within-bin AUC ranges (main score)', '',
          '| Fraction S/B | HT bins | multiplicity bins | leading-pT bins |', '|---|---|---|---|']
    for e in table:
        rng = [e['score']['within_bin_ranges'].get(c) for c in COVARIATES]
        cells = [f'{x[0]:.3f}–{x[1]:.3f}' if x else 'n/a' for x in rng]
        L.append(f"| {frac_tag(e['fraction'])} | {' | '.join(cells)} |")
    L += ['', '## Signal and background in the pool near 3.5 TeV (for information; uses pool labels)', '',
          'Counts are for the whole pool (training part + reference sample), seed-mean.', '',
          '| Fraction S/B | region (TeV) | S | B | S/B | S/sqrt(B) |', '|---|---|---|---|---|---|']
    for e in table:
        for reg in e['near_3p5'][seeds[0]]:
            vals = [e['near_3p5'][s][reg] for s in seeds]
            S = np.mean([v['S'] for v in vals])
            B = np.mean([v['B'] for v in vals])
            L.append(f"| {frac_tag(e['fraction'])} | {reg} | {S:.0f} | {B:.0f} | {S / B:.4f} | {S / np.sqrt(B):.1f} |")
    e0 = table[0]
    L += ['', '## Windows after the merge rule (training counts of fraction '
          f"{frac_tag(e0['fraction'])}, seed {seeds[0]})", '']
    for g in ('a', 'b'):
        L += [f'Grid {g.upper()}:', '', '| SR (TeV) | sidebands (TeV) | train SR events | train SB events |',
              '|---|---|---|---|']
        for w in e0['windows'][g]:
            L.append(f"| {_fmt_iv(w['sr'])} | {' + '.join(_fmt_iv(s) for s in w['sb'])} | {w['n_sr_train']} | "
                     f"{w['n_sb_train']} |")
        L.append('')
    all_merges = [m for e in table for s in seeds for g in ('a', 'b') for m in e['merges'][s][g]]
    L.append(f'Merges in all runs: {len(all_merges)}.' + ('' if not all_merges else
                                                         ' See cwola.json for the list.'))
    L += ['', '## Pre-registered predictions', '']
    for i, p in enumerate(preds, 1):
        L.append(f"{i}. {p['text']}: **{'PASS' if p['passed'] else 'FAIL'}** (value: {_fmt_val(p['value'])})")
    return '\n'.join(L) + '\n'


def _fmt_val(v):
    if isinstance(v, float):
        return f'{v:.3f}'
    if isinstance(v, dict):
        return ', '.join(f'{k}: {_fmt_val(x)}' for k, x in v.items())
    if isinstance(v, list):
        return '–'.join(_fmt_val(x) for x in v)
    return str(v)
