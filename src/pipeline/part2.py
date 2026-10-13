"""Library behind scripts/part2_run.py so each step can be tested without the CLI."""
import glob
import json
import os
import random
import time

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score

from src.benchmarks.baselines import Baseline, jet_baselines, load_jet_features, particle_baselines
from src.benchmarks.report import benchmark_row, check_reference, evaluate_gate, write_report
from src.data.event_features import fit_feature_stats
from src.data.event_store import build_cache, load_role
from src.model.event_ae import EventAE
from src.training.event_trainer import BatchContext, TrainingError, infer, train_event_ae
from src.utils.latent_scores import fit_scorers

COVARIATES = ('ht', 'multiplicity', 'max_pt')
SCORE_NAMES = ('chamfer', 'knn', 'gmm')


def load_config(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def apply_overrides(cfg, overrides):
    """'a.b=value' strings; the value is parsed as YAML (so null, 0.5, [x,y] work)."""
    for item in overrides:
        key, _, raw = item.partition('=')
        node = cfg
        parts = key.split('.')
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def prepare_cache(cfg):
    s, d = cfg['split'], cfg['data']
    split = {k: s[k] for k in ('skip_bg', 'n_test_bg', 'n_test_sig', 'n_train', 'n_contam_sig')}
    return build_cache(d['h5_path'], d['cache_dir'], split, d['max_particles'], d['max_truncated_fraction'])


def parse_external(spec):
    """'name=path.npz:key' -> (name, path, key)."""
    name, _, rest = spec.partition('=')
    path, _, key = rest.rpartition(':')
    if not (name and path and key):
        raise ValueError(f"external must look like 'name=path.npz:key', got {spec!r}")
    return name, path, key


def _test_set(cfg):
    cache = cfg['data']['cache_dir']
    bg, sig = load_role(cache, 'test_bg'), load_role(cache, 'test_sig')
    y = np.r_[np.zeros(len(bg.rows)), np.ones(len(sig.rows))]
    summary = {k: np.concatenate([bg.summary[k], sig.summary[k]]) for k in bg.summary}
    rows = np.concatenate([bg.rows, sig.rows])
    return y, summary, rows


def build_bank(cfg, externals=(), bank_path=None):
    """Computes every baseline (and loads external score files) on the frozen test set and stores them,
    so reports can be rebuilt later without the HDF5 file."""
    bank_path = bank_path or os.path.join(cfg['output_dir'], 'bank.npz')
    os.makedirs(os.path.dirname(bank_path) or '.', exist_ok=True)
    y, summary, rows = _test_set(cfg)
    items = list(particle_baselines(summary))
    features_path = cfg['data'].get('features_path')
    if features_path and os.path.exists(features_path):
        df = load_jet_features(features_path, rows, y)
        items += jet_baselines(df, cfg['benchmark']['jet_oracle_mass'])
    else:
        print(f'WARNING: jet features not found ({features_path}); jet-level baselines skipped', flush=True)
    for spec in externals:
        name, path, key = parse_external(spec)
        with np.load(path) as data:
            if not np.array_equal(np.asarray(data['labels']).astype(int), y.astype(int)):
                raise ValueError(f'{path}: test labels differ from the Part 2 test set; scores are not comparable')
            scores = np.asarray(data[key], dtype=np.float64)
        items.append(Baseline(name, scores, 'external', f'from {os.path.basename(path)}'))
    arrays = {'y': y, **{f'cov_{k}': summary[k] for k in COVARIATES}}
    meta = []
    for i, item in enumerate(items):
        arrays[f's{i}'] = np.asarray(item.scores, dtype=np.float64)
        meta.append({'name': item.name, 'group': item.group, 'notes': item.notes})
    np.savez(bank_path, **arrays)
    with open(bank_path.replace('.npz', '.json'), 'w', encoding='utf-8') as handle:
        json.dump(meta, handle, indent=2)
    return bank_path


def make_report(cfg, bank_path, run_tags=(), name='report'):
    """Benchmark table = bank baselines + seed-averaged model scores of the given run tags.
    Tag 'main' rows form the gate (group 'model'); other tags are reported as 'variant'."""
    with np.load(bank_path) as loaded:
        bank = {k: loaded[k] for k in loaded.files}
    with open(bank_path.replace('.npz', '.json'), encoding='utf-8') as handle:
        meta = json.load(handle)
    y = bank['y']
    covariates = {c: bank[f'cov_{c}'] for c in COVARIATES}
    b = cfg['benchmark']
    rows = [benchmark_row(m['name'], y, bank[f's{i}'], covariates, m['group'], b['baseline_ci_level'],
                          b['n_boot'], b['n_bins'], m['notes']) for i, m in enumerate(meta)]
    for tag in run_tags:
        files = sorted(glob.glob(os.path.join(cfg['output_dir'], 'runs', tag, 'scores_s*.npz')))
        if not files:
            raise FileNotFoundError(f'no scores for run tag {tag!r} under {cfg["output_dir"]}/runs')
        runs = []
        for f in files:
            with np.load(f) as loaded:
                runs.append({k: loaded[k] for k in loaded.files})
        for r in runs:
            if not np.array_equal(np.asarray(r['labels']).astype(int), y.astype(int)):
                raise ValueError(f'run {tag}: test labels differ from the bank')
        for score in SCORE_NAMES:
            label = f'EventAE {score}' + ('' if tag == 'main' else f' [{tag}]')
            rows.append(benchmark_row(label, y, [r[score] for r in runs], covariates,
                                      'model' if tag == 'main' else 'variant', b['model_ci_level'],
                                      b['n_boot'], b['n_bins'], f'{len(runs)} seed(s)'))
    gate = evaluate_gate(rows, b['min_within_bin']) if any(r['group'] == 'model' for r in rows) else None
    expected = cfg.get('stage0_expected') or {}
    reference = check_reference(rows, expected) if expected else None
    write_report(rows, gate, reference, os.path.join(cfg['output_dir'], f'{name}.json'),
                 os.path.join(cfg['output_dir'], f'{name}.md'), f'Part 2 benchmark: {name}',
                 meta={'run_tags': list(run_tags), 'n_test': int(len(y))})
    return {'rows': rows, 'gate': gate, 'reference': reference}


LOSSES = ('chamfer', 'chamfer_ptw')


def run_training(cfg, seed, contamination=None, tag='main'):
    """Trains one EventAE, computes the three scores on the frozen test set, writes the run files."""
    if cfg['training']['loss'] not in LOSSES:
        raise ValueError(f"training.loss must be one of {LOSSES}, got {cfg['training']['loss']!r}")
    run_dir = os.path.join(cfg['output_dir'], 'runs', tag)
    os.makedirs(run_dir, exist_ok=True)
    scores_path = os.path.join(run_dir, f'scores_s{seed}.npz')
    metrics_path = os.path.join(run_dir, f'metrics_s{seed}.json')
    if os.path.exists(scores_path):
        print(f'run {tag}/s{seed} already done; skipping', flush=True)
        with open(metrics_path, encoding='utf-8') as handle:
            return json.load(handle)

    s, d, m, t = cfg['split'], cfg['data'], cfg['model'], cfg['training']
    contamination = s['contamination'] if contamination is None else contamination
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cache = d['cache_dir']
    train_bg, contam = load_role(cache, 'train_bg'), load_role(cache, 'contam_sig')
    test_bg, test_sig = load_role(cache, 'test_bg'), load_role(cache, 'test_sig')
    if len(train_bg.rows) != s['n_train']:
        raise ValueError(f"cache {cache} holds {len(train_bg.rows)} training events but split.n_train is "
                         f"{s['n_train']}: the cache does not match the config (re-run prepare or use another "
                         'cache_dir)')
    n_val = int(round(s['val_fraction'] * s['n_train']))
    n_fit = s['n_train'] - n_val
    n_sig = int(round(contamination * n_fit))
    if n_sig > len(contam.rows):
        raise ValueError(f'contamination {contamination} needs {n_sig} signal events, cache has {len(contam.rows)}')
    fit_np = np.concatenate([train_bg.particles[:n_fit - n_sig], contam.particles[:n_sig]])
    val_np = train_bg.particles[n_fit:]
    test_np = np.concatenate([test_bg.particles, test_sig.particles])
    y = np.r_[np.zeros(len(test_bg.rows)), np.ones(len(test_sig.rows))].astype(int)

    fit_t = torch.from_numpy(fit_np).to(device)
    val_t = torch.from_numpy(val_np).to(device)
    test_t = torch.from_numpy(test_np).to(device)
    ctx = BatchContext(stats=fit_feature_stats(fit_t).to(device), k=m['k_recon'], radius=d['radius'],
                       wrap_phi=d['wrap_phi'], loss=t['loss'])
    model = EventAE(width=m['width'], layers=m['layers'], hidden=m['hidden'], latent_dim=m['latent_dim'],
                    k_recon=m['k_recon'], pooling=tuple(m['pooling'])).to(device)

    started = time.time()
    try:
        history = train_event_ae(
            model, fit_t, val_t, ctx, epochs=t['epochs'], batch_size=t['batch_size'], lr=t['learning_rate'],
            weight_decay=float(t['weight_decay']), patience=t['patience'], grad_clip=t['grad_clip'],
            collapse_std=t['collapse_std'], seed=seed)
    except TrainingError as err:
        with open(os.path.join(run_dir, f'failed_s{seed}.json'), 'w', encoding='utf-8') as handle:
            json.dump({'seed': seed, 'tag': tag, 'reason': str(err)}, handle)
        raise

    chamfer_test, z_test = infer(model, test_t, ctx)
    _, z_fit = infer(model, fit_t, ctx)
    scorers = fit_scorers(z_fit, cfg['scores']['knn_k'], cfg['scores']['gmm_components'], seed)
    if not all(sc.n_fit == len(fit_np) for sc in scorers.values()):
        raise RuntimeError('density scorers must be fit on training latents only '
                           f'(expected n_fit={len(fit_np)})')
    scores = {'chamfer': chamfer_test, 'knn': scorers['knn'].score(z_test), 'gmm': scorers['gmm'].score(z_test)}

    torch.save({'model_state_dict': model.state_dict(), 'config': cfg}, os.path.join(run_dir, f'model_s{seed}.pt'))
    metrics = {
        'tag': tag, 'seed': seed, 'contamination': contamination, 'n_contaminating_signal': n_sig,
        'n_fit': int(len(fit_np)), 'n_val': int(len(val_np)),
        'auc': {k: float(roc_auc_score(y, v)) for k, v in scores.items()},
        'epochs_run': len(history), 'last_epoch_val_loss': history[-1]['val_loss'],
        'best_val_loss': min(h['val_loss'] for h in history),
        'seconds_per_epoch': float(np.mean([h['seconds'] for h in history])),
        'seconds_total': round(time.time() - started, 1),
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'history': history,
    }
    with open(metrics_path, 'w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2)
    # The scores file is written last: it is the "run finished" marker (skip logic, scripts/part2_sweep.py).
    # Atomic (np.savez appends .npz to names lacking it, and the temp name must not match the scores_s*.npz glob).
    tmp_scores = os.path.join(run_dir, f'tmp_scores_s{seed}.npz')
    try:
        np.savez(tmp_scores, labels=y, **scores)
        os.replace(tmp_scores, scores_path)
    finally:
        if os.path.exists(tmp_scores):
            os.remove(tmp_scores)
    stale = os.path.join(run_dir, f'failed_s{seed}.json')
    if os.path.exists(stale):
        os.remove(stale)
    print(f"run {tag}/s{seed}: AUC {metrics['auc']}", flush=True)
    return metrics
