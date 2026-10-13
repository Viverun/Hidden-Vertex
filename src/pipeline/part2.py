"""Library behind scripts/part2_run.py so each step can be tested without the CLI."""
import glob
import json
import os

import numpy as np
import yaml

from src.benchmarks.baselines import Baseline, jet_baselines, load_jet_features, particle_baselines
from src.benchmarks.report import benchmark_row, check_reference, evaluate_gate, write_report
from src.data.event_store import build_cache, load_role

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
