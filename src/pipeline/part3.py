"""Library behind scripts/part3_run.py: jet-feature density scores (3A) and the WNAE (3B).

No function here computes a test AUC during training. Test AUCs appear only in `make_reports`, after all
choices (including the WNAE epoch count) are fixed in configs/part3_jet.yaml.
"""
import json
import os
import random
import time

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve

from src.benchmarks.baselines import jet_baselines
from src.benchmarks.report import auc_ci, benchmark_row, evaluate_gate, mean_auc, within_bin_min, write_report
from src.data.jet_features import FEATURE_SETS, load_features, prepare_sets, split_rows
from src.model.mlp_ae import MLPAE, recon_error, train_ae
from src.model.wnae import WNAETrainer
from src.utils.latent_scores import GMMScorer, KNNScorer

COVARIATES = ('ht', 'multiplicity', 'max_pt')
DENSITY_SCORES = ('gmm', 'knn', 'ae')
SCORE_LABELS = {'gmm': 'GMM NLL', 'knn': 'kNN dist', 'ae': 'AE recon', 'wnae': 'WNAE recon'}


def load_config(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _save_npz_atomic(path, **arrays):
    tmp = path[:-4] + '.tmp.npz'
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _load_bank(bank_path):
    with np.load(bank_path) as loaded:
        bank = {k: loaded[k] for k in loaded.files}
    with open(bank_path.replace('.npz', '.json'), encoding='utf-8') as handle:
        meta = json.load(handle)
    return bank, meta


def load_data(cfg):
    df = load_features(cfg['data']['features_path'])
    rows = split_rows(df['label'].to_numpy().astype(int), cfg['split'])
    test_labels = df['label'].to_numpy().astype(int)[rows['test']]
    if not np.array_equal(test_labels, rows['test_labels']):
        raise AssertionError('test labels from the features file disagree with the split plan')
    for role in ('fit', 'val'):
        if df['label'].to_numpy()[rows[role]].any():
            raise AssertionError(f'{role} set contains signal events')
    return df, rows


def verify_bank_alignment(df, rows, bank_path):
    """The bank has no row ids: recompute mjj and -tau21 (leading jet) for our test rows and compare."""
    bank, meta = _load_bank(bank_path)
    names = [m['name'] for m in meta]
    if not np.array_equal(bank['y'].astype(int), rows['test_labels']):
        raise AssertionError('bank labels differ from the Part 3 test labels')
    mine = {b.name: b.scores for b in jet_baselines(df.iloc[rows['test']].reset_index(drop=True))}
    out = {}
    for name in ('mjj', '-tau21 (leading jet)'):
        ok = bool(np.allclose(mine[name], bank[f's{names.index(name)}']))
        out[name] = ok
        if not ok:
            raise AssertionError(f'bank column {name!r} does not match the recomputed values: rows misaligned')
    return out


def _covariates(bank):
    return {c: bank[f'cov_{c}'] for c in COVARIATES}


# ----------------------------------------------------------------------------------------------- Part 3A
def run_density(cfg, seeds=None, log=print):
    """GMM, kNN and AE scores on every feature set; one npz per seed with labels, rows and all score arrays."""
    df, rows = load_data(cfg)
    alignment = verify_bank_alignment(df, rows, cfg['data']['bank_path'])
    log(f'bank alignment: {alignment}')
    dcfg, q = cfg['density'], cfg['features']['quantile']
    seeds = dcfg['seeds'] if seeds is None else seeds
    out_dir = os.path.join(cfg['output_dir'], 'density')
    os.makedirs(out_dir, exist_ok=True)
    sets = [cfg['features']['headline'], *cfg['features']['variants']]
    data = {fs: prepare_sets(df, rows, fs, q['n_quantiles'], q['random_state'])[0] for fs in sets}
    for seed in seeds:
        scores, metrics = {}, {'seed': seed, 'n_fit': int(len(rows['fit'])), 'n_val': int(len(rows['val'])),
                               'bank_alignment': alignment, 'feature_sets': {}}
        for fs in sets:
            x = data[fs]
            started = time.time()
            gmm = GMMScorer(dcfg['gmm_components'], seed).fit(x['fit'])
            knn = KNNScorer(dcfg['knn_k']).fit(x['fit'])
            assert gmm.n_fit == knn.n_fit == len(rows['fit'])
            scores[f'gmm_{fs}'] = gmm.score(x['test'])
            scores[f'knn_{fs}'] = knn.score(x['test'])
            a = dcfg['ae']
            _seed_all(seed)
            model = MLPAE(x['fit'].shape[1], tuple(a['hidden']), a['latent'])
            history, best_epoch = train_ae(model, x['fit'], x['val'], epochs=a['epochs'], batch_size=a['batch_size'],
                                           lr=a['learning_rate'], weight_decay=float(a['weight_decay']),
                                           patience=a['patience'], seed=seed)
            scores[f'ae_{fs}'] = recon_error(model, x['test'])
            torch.save({'model_state_dict': model.state_dict(), 'feature_set': fs, 'config': cfg},
                       os.path.join(out_dir, f'ae_{fs}_s{seed}.pt'))
            metrics['feature_sets'][fs] = {
                'columns': list(FEATURE_SETS[fs]), 'gmm_converged': bool(gmm._gmm.converged_),
                'ae_best_epoch': best_epoch, 'ae_epochs_run': len(history),
                'ae_best_val_loss': min(h['val_loss'] for h in history), 'ae_history': history,
                'seconds': round(time.time() - started, 1)}
            log(f'seed {seed} {fs}: AE best epoch {best_epoch}/{len(history)} '
                f"({metrics['feature_sets'][fs]['seconds']} s)")
        with open(os.path.join(out_dir, f'metrics_s{seed}.json'), 'w', encoding='utf-8') as handle:
            json.dump(metrics, handle, indent=2)
        _save_npz_atomic(os.path.join(out_dir, f'scores_s{seed}.npz'), labels=rows['test_labels'],
                         rows=rows['test'], **scores)
    return out_dir


# ----------------------------------------------------------------------------------------------- Part 3B
WNAE_RUNS = ('wnae', 'wnae_long')


def wnae_settings(cfg, run='wnae'):
    """Settings of one WNAE run: the pilot section `wnae`, updated by the keys of section `run`."""
    if run not in WNAE_RUNS:
        raise ValueError(f'unknown WNAE run {run!r}; choose from {WNAE_RUNS}')
    return {**cfg['wnae'], **(cfg.get(run) or {})} if run != 'wnae' else dict(cfg['wnae'])


def _wnae_trainer(cfg, n_inputs, seed, run='wnae'):
    w, a = wnae_settings(cfg, run), cfg['density']['ae']
    _seed_all(seed)
    model = MLPAE(n_inputs, tuple(a['hidden']), a['latent'])
    mcmc = dict(w['mcmc'])
    mcmc['bound'] = tuple(mcmc['bound']) if mcmc.get('bound') is not None else None
    trainer = WNAETrainer(model, n_inputs, mcmc, batch_size=w['batch_size'], lr=float(w['learning_rate']),
                          scheduler=w['scheduler'], seed=seed, buffer_size=w['buffer_size'],
                          ot_num_iter_max=int(w['ot_num_iter_max']))
    return model, trainer


def _wnae_inputs(cfg, run='wnae'):
    df, rows = load_data(cfg)
    q = cfg['features']['quantile']
    x, _ = prepare_sets(df, rows, wnae_settings(cfg, run)['feature_set'], q['n_quantiles'], q['random_state'])
    return df, rows, x


def time_wnae(cfg, seed=0, threads=None, log=print, run='wnae', epochs=1):
    """`epochs` training epochs + validation, no test scores. Writes results/part3/<run>/timing_s<seed>.json."""
    w = wnae_settings(cfg, run)
    torch.set_num_threads(threads or w['torch_threads'])
    _, rows, x = _wnae_inputs(cfg, run)
    model, trainer = _wnae_trainer(cfg, x['fit'].shape[1], seed, run)
    history, _, diverged = trainer.fit(x['fit'], x['val'], max_epochs=epochs, log=log)
    out_dir = os.path.join(cfg['output_dir'], run)
    os.makedirs(out_dir, exist_ok=True)
    info = {'seed': seed, 'run': run, 'threads': torch.get_num_threads(), 'diverged': diverged,
            'seconds_epoch': float(np.mean([h['seconds'] for h in history])) if history else None,
            'history': history}
    with open(os.path.join(out_dir, f'timing_s{seed}.json'), 'w', encoding='utf-8') as handle:
        json.dump(info, handle, indent=2)
    return info


def run_wnae(cfg, seed, threads=None, log=print, run='wnae'):
    w = wnae_settings(cfg, run)
    max_epochs = w['max_epochs']
    if not isinstance(max_epochs, int) or max_epochs < 1:
        raise ValueError(f'{run}.max_epochs must be fixed (an int) before training, got {max_epochs!r}')
    torch.set_num_threads(threads or w['torch_threads'])
    out_dir = os.path.join(cfg['output_dir'], run)
    os.makedirs(out_dir, exist_ok=True)
    df, rows, x = _wnae_inputs(cfg, run)
    verify_bank_alignment(df, rows, cfg['data']['bank_path'])
    model, trainer = _wnae_trainer(cfg, x['fit'].shape[1], seed, run)
    progress_path = os.path.join(out_dir, f'progress_s{seed}.json')
    progress = []

    def on_epoch(rec, _state):                    # training curve only (no test-set quantity)
        progress.append(rec)
        with open(progress_path, 'w', encoding='utf-8') as handle:
            json.dump(progress, handle)

    started = time.time()
    history, best_epoch, diverged = trainer.fit(x['fit'], x['val'], max_epochs=max_epochs,
                                                log=lambda m: log(f'[s{seed}] {m}'),
                                                es_patience=w.get('es_patience'),
                                                es_threshold=float(w.get('es_threshold', 1e-3)), on_epoch=on_epoch)
    seconds = time.time() - started
    best = next((h for h in history if h['epoch'] == best_epoch), None)
    metrics = {'seed': seed, 'run': run, 'max_epochs': max_epochs, 'epochs_run': len(history),
               'best_epoch': best_epoch, 'diverged': diverged, 'early_stopped_at': trainer.early_stopped,
               'es_patience': w.get('es_patience'), 'batch_size': w['batch_size'],
               'seconds_total': round(seconds, 1),
               'best_val_w': best['val_w'] if best else None,
               'first_val_w': history[0]['val_w'] if history else None,
               'final_val_w': history[-1]['val_w'] if history else None,
               'final_lr': history[-1]['lr'] if history else None,
               'energies_at_best': ({k: best[k] for k in ('val_e_pos', 'val_e_neg', 'train_e_pos', 'train_e_neg')}
                                    if best else None),
               'ot_warnings_total': int(sum(h['ot_warnings'] for h in history)),
               'negatives_at_best': ({k: best[k] for k in ('neg_mean', 'neg_std', 'pos_mean', 'pos_std')}
                                     if best else None),
               'threads': torch.get_num_threads(), 'history': history}
    with open(os.path.join(out_dir, f'metrics_s{seed}.json'), 'w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2)
    if best is None:
        log(f'[s{seed}] no finite epoch; no scores written')
        return metrics
    torch.save({'model_state_dict': model.state_dict(), 'best_epoch': best_epoch, 'config': cfg, 'run': run},
               os.path.join(out_dir, f'model_s{seed}.pt'))
    _save_npz_atomic(os.path.join(out_dir, f'scores_s{seed}.npz'), labels=rows['test_labels'], rows=rows['test'],
                     wnae_jet5=recon_error(model, x['test']))
    log(f'[s{seed}] done: best epoch {best_epoch}, {seconds / 60:.1f} min')
    return metrics


# ----------------------------------------------------------------------------------------------- reports
def _load_runs(directory, y):
    files = sorted(f for f in os.listdir(directory) if f.startswith('scores_s') and f.endswith('.npz'))
    runs = {}
    for f in files:
        with np.load(os.path.join(directory, f)) as loaded:
            run = {k: loaded[k] for k in loaded.files}
        if not np.array_equal(run['labels'].astype(int), np.asarray(y).astype(int)):
            raise ValueError(f'{f}: test labels differ from the bank')
        runs[int(f[len('scores_s'):-4])] = run
    return runs


def spearman_diagnostics(score, y, bank):
    mean_pt = bank['cov_ht'] / bank['cov_multiplicity']
    covs = {'multiplicity': bank['cov_multiplicity'], 'ht': bank['cov_ht'], 'mean_pt': mean_pt}
    subsets = {'all': np.ones(len(y), bool), 'bg': y == 0, 'sig': y == 1}
    return {c: {s: float(spearmanr(score[m], v[m])[0]) for s, m in subsets.items()} for c, v in covs.items()}


def within_bin_ranges(row):
    return {c: [min(b['auc'] for b in bins), max(b['auc'] for b in bins)] if bins else None
            for c, bins in row['within_bin'].items()}


def _score_rows(y, runs, key, label, group, level, b, covariates):
    seeds = sorted(runs)
    arrays = [runs[s][key] for s in seeds]
    row = benchmark_row(label, y, arrays, covariates, group, level, b['n_boot'], b['n_bins'],
                        f'{len(seeds)} seed(s)')
    extra = {'name': label, 'per_seed_auc': {int(s): float(roc_auc_score(y, runs[s][key])) for s in seeds},
             'within_bin_ranges': within_bin_ranges(row)}
    return row, extra


def _extra_markdown(extras, gate_note=''):
    lines = ['', '## Per-seed AUC and within-bin AUC ranges (seed-averaged scores)', '',
             '| Score | per-seed AUC | HT bins | multiplicity bins | leading-pT bins |', '|---|---|---|---|---|']
    for e in extras:
        ps = ' / '.join(f'{v:.3f}' for _, v in sorted(e['per_seed_auc'].items()))
        rng = [e['within_bin_ranges'].get(c) for c in COVARIATES]
        cells = [f'{r[0]:.3f}–{r[1]:.3f}' if r else 'n/a' for r in rng]
        lines.append(f"| {e['name']} | {ps} | {' | '.join(cells)} |")
    lines += ['', '## Spearman correlation of the seed-0 score (all | background | signal)', '',
              '| Score | multiplicity | HT | mean pT (= HT / multiplicity) |', '|---|---|---|---|']
    for e in extras:
        sp = e['spearman_seed0']
        cells = [' \\| '.join(f"{sp[c][s]:+.2f}" for s in ('all', 'bg', 'sig')) for c in ('multiplicity', 'ht', 'mean_pt')]
        lines.append(f"| {e['name']} | {' | '.join(cells)} |")
    if gate_note:
        lines += ['', gate_note]
    return '\n'.join(lines) + '\n'


def _wnae_markdown(wnae_metrics, names):
    lines = ['', '## WNAE training', '',
             '| Seed | epochs run / max | best epoch | best val W | final val W | diverged | minutes |',
             '|---|---|---|---|---|---|---|']
    for s, m in sorted(wnae_metrics.items()):
        bw = f"{m['best_val_w']:.4f}" if m['best_val_w'] is not None else 'n/a'
        fw = f"{m['final_val_w']:.4f}" if m['final_val_w'] is not None else 'n/a'
        lines.append(f"| {s} | {m['epochs_run']} / {m['max_epochs']} | {m['best_epoch']} | {bw} | {fw} | "
                     f"{m['diverged'] or 'no'} | {m['seconds_total'] / 60:.1f} |")
    lines += ['', '### Negative samples vs positives (validation set, selected epoch): mean / std per feature', '',
              '| Seed | feature | positive mean | negative mean | positive std | negative std |',
              '|---|---|---|---|---|---|']
    for s, m in sorted(wnae_metrics.items()):
        n = m['negatives_at_best']
        if not n:
            continue
        for i, f in enumerate(names):
            lines.append(f"| {s} | {f} | {n['pos_mean'][i]:+.3f} | {n['neg_mean'][i]:+.3f} | "
                         f"{n['pos_std'][i]:.3f} | {n['neg_std'][i]:.3f} |")
    return '\n'.join(lines) + '\n'


# ----------------------------------------------------------------------------------------------- SIC
def sic_metrics(y, score, min_fpr=1e-3, fprs=(0.01, 0.1)):
    """Significance improvement: max TPR/sqrt(FPR) over ROC points with FPR >= min_fpr, and TPR and SIC at
    the given FPRs (linear interpolation of the ROC curve). Higher score = more anomalous; y = 1 is signal."""
    fpr, tpr, _ = roc_curve(np.asarray(y).astype(int), np.asarray(score, dtype=np.float64))
    sel = fpr >= min_fpr
    f_pts = np.r_[min_fpr, fpr[sel]]                 # ROC vertices with FPR >= min_fpr + the interpolated point at min_fpr
    t_pts = np.r_[np.interp(min_fpr, fpr, tpr), tpr[sel]]
    sic = t_pts / np.sqrt(f_pts)
    i = int(np.argmax(sic))
    out = {'max_sic': float(sic[i]), 'max_sic_fpr': float(f_pts[i]), 'max_sic_tpr': float(t_pts[i])}
    for f in fprs:
        t = float(np.interp(f, fpr, tpr))
        out[f'tpr@{f:g}'] = t
        out[f'sic@{f:g}'] = t / np.sqrt(f)
    return out


def sic_summary(y, arrays, min_fpr=1e-3, fprs=(0.01, 0.1)):
    """Per-array SIC metrics and their mean over arrays (seeds)."""
    per = [sic_metrics(y, a, min_fpr, fprs) for a in arrays]
    mean = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
    return {'mean': mean, 'per_seed': per}


def _sic_markdown(rows, fprs):
    head = ' | '.join(f'TPR@{f:g} | SIC@{f:g}' for f in fprs)
    lines = ['', '## Significance improvement (SIC = TPR / sqrt(FPR); max over FPR >= 1e-3; mean over seeds)', '',
             f'| Score | Group | max SIC | FPR at max | {head} |', '|---|---|---|---|' + '---|---|' * len(fprs)]
    for r in sorted(rows, key=lambda r: -r['sic']['mean']['max_sic']):
        m = r['sic']['mean']
        cells = ' | '.join(f"{m[f'tpr@{f:g}']:.3f} | {m[f'sic@{f:g}']:.2f}" for f in fprs)
        lines.append(f"| {r['name']} | {r['group']} | {m['max_sic']:.2f} | {m['max_sic_fpr']:.4f} | {cells} |")
    return '\n'.join(lines) + '\n'


def _wnae_long_markdown(wmetrics):
    lines = ['', '## WNAE-long training (validation W curve)', '',
             '| Seed | epochs run / max | early stop at | first val W | min val W | epoch of min | last val W | '
             'final lr | E+ / E- at best (val) | OT non-optimal exits | minutes |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for s, m in sorted(wmetrics.items()):
        e = m.get('energies_at_best') or {}
        lines.append(f"| {s} | {m['epochs_run']} / {m['max_epochs']} | {m.get('early_stopped_at') or 'no'} | "
                     f"{m['first_val_w']:.4f} | {m['best_val_w']:.4f} | {m['best_epoch']} | {m['final_val_w']:.4f} | "
                     f"{m['final_lr']:.2e} | {e.get('val_e_pos', float('nan')):.4f} / "
                     f"{e.get('val_e_neg', float('nan')):.4f} | {m.get('ot_warnings_total')} | "
                     f"{m['seconds_total'] / 60:.1f} |")
    return '\n'.join(lines) + '\n'


def _wnae_metrics(directory):
    out = {}
    for f in sorted(os.listdir(directory)):
        if f.startswith('metrics_s') and f.endswith('.json'):
            with open(os.path.join(directory, f), encoding='utf-8') as handle:
                m = json.load(handle)
            m.pop('history', None)
            out[m['seed']] = m
    return out


def make_reports(cfg, log=print):
    b = cfg['benchmark']
    min_fpr, fprs = float(b.get('sic_min_fpr', 1e-3)), tuple(b.get('sic_fprs', (0.01, 0.1)))
    bank, meta = _load_bank(cfg['data']['bank_path'])
    y = bank['y']
    covariates = _covariates(bank)
    base_rows = []
    for i, m in enumerate(meta):
        row = benchmark_row(m['name'], y, bank[f's{i}'], covariates, m['group'], b['baseline_ci_level'],
                            b['n_boot'], b['n_bins'], m['notes'])
        row['sic'] = sic_summary(y, [bank[f's{i}']], min_fpr, fprs)
        base_rows.append(row)
    out = cfg['output_dir']
    density_runs = _load_runs(os.path.join(out, 'density'), y)
    if not density_runs:
        raise FileNotFoundError('no density scores; run `density` first')
    sets = [cfg['features']['headline'], *cfg['features']['variants']]
    rows, extras = list(base_rows), []

    def add(runs, key, label, group, level, rows_, extras_):
        row, extra = _score_rows(y, runs, key, label, group, level, b, covariates)
        row['sic'] = sic_summary(y, [runs[s][key] for s in sorted(runs)], min_fpr, fprs)
        extra['spearman_seed0'] = spearman_diagnostics(runs[min(runs)][key], y, bank)
        extra['per_seed_sic'] = {int(s): v for s, v in zip(sorted(runs), row['sic']['per_seed'])}
        rows_.append(row)
        extras_.append(extra)

    for fs in sets:
        group = 'model' if fs == cfg['features']['headline'] else 'variant'
        for score in DENSITY_SCORES:
            add(density_runs, f'{score}_{fs}', f'{SCORE_LABELS[score]} [{fs}]', group, b['model_ci_level'], rows, extras)
    gate = evaluate_gate(rows, b['min_within_bin'])
    meta_out = {'n_test': int(len(y)), 'seeds': sorted(density_runs), 'extras': extras,
                'sic_definition': {'min_fpr': min_fpr, 'fprs': list(fprs)}}
    write_report(rows, gate, None, os.path.join(out, 'jet_density.json'), os.path.join(out, 'jet_density.md'),
                 'Part 3A: jet-feature density scores', meta=meta_out)
    with open(os.path.join(out, 'jet_density.md'), 'a', encoding='utf-8') as handle:
        handle.write(_extra_markdown(extras))
        handle.write(_sic_markdown(rows, fprs))
    log(f"jet_density: gate {'PASSED' if gate['passed'] else 'NOT PASSED'}")
    result = {'jet_density': {'rows': rows, 'gate': gate, 'extras': extras}}

    rows2, extras2, wmeta = list(rows), list(extras), {}
    for run, label in (('wnae', 'WNAE recon [jet5]'), ('wnae_long', 'WNAE-long recon [jet5]')):
        wdir = os.path.join(out, run)
        runs = _load_runs(wdir, y) if os.path.isdir(wdir) else {}
        if not runs:
            continue
        add(runs, 'wnae_jet5', label, 'variant', b['wnae_ci_level'], rows2, extras2)
        wmeta[run] = _wnae_metrics(wdir)
    if wmeta:
        gate2 = evaluate_gate(rows2, b['min_within_bin'])
        write_report(rows2, gate2, None, os.path.join(out, 'wnae.json'), os.path.join(out, 'wnae.md'),
                     'Part 3B: WNAE on jet5 (with the Part 3A rows)',
                     meta={'n_test': int(len(y)), 'extras': extras2, 'wnae_metrics': wmeta.get('wnae', {}),
                           'wnae_long_metrics': wmeta.get('wnae_long', {}),
                           'sic_definition': {'min_fpr': min_fpr, 'fprs': list(fprs)}})
        names = FEATURE_SETS[cfg['wnae']['feature_set']]
        with open(os.path.join(out, 'wnae.md'), 'a', encoding='utf-8') as handle:
            handle.write(_extra_markdown(extras2))
            handle.write(_sic_markdown(rows2, fprs))
            if 'wnae' in wmeta:
                handle.write(_wnae_markdown(wmeta['wnae'], names).replace('## WNAE training', '## WNAE pilot training'))
            if 'wnae_long' in wmeta:
                handle.write(_wnae_long_markdown(wmeta['wnae_long']))
                handle.write(_wnae_markdown(wmeta['wnae_long'], names)
                             .replace('## WNAE training', '## WNAE-long training (summary)'))
        log('wnae report written')
        result['wnae'] = {'rows': rows2, 'gate': gate2, 'extras': extras2, 'wnae_metrics': wmeta}
    return result


# ----------------------------------------------------------------------------------------------- reverse
def reverse_rows(labels, cfg):
    """Signal rows for the reverse fit: signal index skip_sig .. skip_sig+n_fit (fit), then n_val (AE val)."""
    r = cfg['reverse']
    sig = np.flatnonzero(np.asarray(labels) == 1)
    need = r['skip_sig'] + r['n_fit'] + r['n_val']
    if len(sig) < need:
        raise ValueError(f'need {need} signal events, file has {len(sig)}')
    return sig[r['skip_sig']:r['skip_sig'] + r['n_fit']], sig[r['skip_sig'] + r['n_fit']:need]


def run_reverse(cfg, seeds=None, log=print):
    """DIAGNOSTIC (uses signal labels): fit GMM/kNN/AE on signal events outside the test set, score the frozen
    test set, and report AUC with background as the positive class next to the forward AUC."""
    df, rows = load_data(cfg)
    verify_bank_alignment(df, rows, cfg['data']['bank_path'])
    labels = df['label'].to_numpy().astype(int)
    fit, val = reverse_rows(labels, cfg)
    if np.intersect1d(np.r_[fit, val], rows['test']).size or labels[fit].min() != 1 or labels[val].min() != 1:
        raise AssertionError('reverse fit/val rows must be signal and disjoint from the test set')
    rrows = {'fit': fit, 'val': val, 'test': rows['test']}
    dcfg, q, b = cfg['density'], cfg['features']['quantile'], cfg['benchmark']
    seeds = dcfg['seeds'] if seeds is None else seeds
    out_dir = os.path.join(cfg['output_dir'], 'reverse')
    os.makedirs(out_dir, exist_ok=True)
    sets = [cfg['features']['headline'], *cfg['features']['variants']]
    data = {fs: prepare_sets(df, rrows, fs, q['n_quantiles'], q['random_state'])[0] for fs in sets}
    for seed in seeds:
        path = os.path.join(out_dir, f'scores_s{seed}.npz')
        if os.path.exists(path):
            log(f'reverse seed {seed}: exists, skipped')
            continue
        scores, metrics = {}, {'seed': seed, 'feature_sets': {}}
        for fs in sets:
            x = data[fs]
            gmm = GMMScorer(dcfg['gmm_components'], seed).fit(x['fit'])
            knn = KNNScorer(dcfg['knn_k']).fit(x['fit'])
            scores[f'gmm_{fs}'], scores[f'knn_{fs}'] = gmm.score(x['test']), knn.score(x['test'])
            a = dcfg['ae']
            _seed_all(seed)
            model = MLPAE(x['fit'].shape[1], tuple(a['hidden']), a['latent'])
            history, best_epoch = train_ae(model, x['fit'], x['val'], epochs=a['epochs'], batch_size=a['batch_size'],
                                           lr=a['learning_rate'], weight_decay=float(a['weight_decay']),
                                           patience=a['patience'], seed=seed)
            scores[f'ae_{fs}'] = recon_error(model, x['test'])
            metrics['feature_sets'][fs] = {'gmm_converged': bool(gmm._gmm.converged_), 'ae_best_epoch': best_epoch,
                                           'ae_epochs_run': len(history),
                                           'ae_best_val_loss': min(h['val_loss'] for h in history)}
            log(f'reverse seed {seed} {fs}: AE best epoch {best_epoch}/{len(history)}')
        with open(os.path.join(out_dir, f'metrics_s{seed}.json'), 'w', encoding='utf-8') as handle:
            json.dump(metrics, handle, indent=2)
        _save_npz_atomic(path, labels=rows['test_labels'], rows=rows['test'], **scores)

    y = rows['test_labels']
    rev_runs = _load_runs(out_dir, y)
    fwd_runs = _load_runs(os.path.join(cfg['output_dir'], 'density'), y)
    table = []
    for fs in sets:
        for score in DENSITY_SCORES:
            key = f'{score}_{fs}'
            rev = [rev_runs[s][key] for s in sorted(rev_runs)]
            lo, hi = auc_ci(1 - y, rev, level=0.95, n_boot=b['n_boot'])
            entry = {'name': f'{SCORE_LABELS[score]} [{fs}]',
                     'reverse_auc': mean_auc(1 - y, rev), 'reverse_ci': [lo, hi],
                     'reverse_per_seed': [float(roc_auc_score(1 - y, a)) for a in rev]}
            if fwd_runs:
                fwd = [fwd_runs[s][key] for s in sorted(fwd_runs)]
                entry.update({'forward_auc': mean_auc(y, fwd),
                              'forward_per_seed': [float(roc_auc_score(y, a)) for a in fwd]})
            table.append(entry)
    note = ('DIAGNOSTIC ONLY: the reverse scorers are fit on signal events (this uses signal labels). '
            'This is not an anomaly-detection result.')
    info = {'note': note, 'fit_rows': [int(fit[0]), int(fit[-1])], 'val_rows': [int(val[0]), int(val[-1])],
            'n_fit': int(len(fit)), 'n_val': int(len(val)), 'positive_class': 'background (label 0)',
            'seeds': sorted(rev_runs), 'rows': table}
    with open(os.path.join(cfg['output_dir'], 'reverse.json'), 'w', encoding='utf-8') as handle:
        json.dump(info, handle, indent=2)
    lines = ['# Part 3 reverse-direction check (diagnostic)', '', f'**{note}**', '',
             f'Reverse fit set: signal rows {fit[0]:,}–{fit[-1]:,} ({len(fit):,} events); AE validation: signal rows '
             f'{val[0]:,}–{val[-1]:,} ({len(val):,}). Quantile transform refit on the reverse fit set. Test set: the '
             'frozen Part 2 test set, scored with background as the positive ("anomalous") class.', '',
             '| Score | forward AUC (bg-trained, signal positive) | forward per seed | reverse AUC (signal-trained, '
             'bg positive) | reverse 95% CI | reverse per seed |', '|---|---|---|---|---|---|']
    for e in table:
        fps = ' / '.join(f'{v:.3f}' for v in e.get('forward_per_seed', []))
        rps = ' / '.join(f'{v:.3f}' for v in e['reverse_per_seed'])
        fa = f"{e['forward_auc']:.3f}" if 'forward_auc' in e else 'n/a'
        lines.append(f"| {e['name']} | {fa} | {fps} | {e['reverse_auc']:.3f} | "
                     f"{e['reverse_ci'][0]:.3f}–{e['reverse_ci'][1]:.3f} | {rps} |")
    with open(os.path.join(cfg['output_dir'], 'reverse.md'), 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
    log('reverse report written')
    return info
