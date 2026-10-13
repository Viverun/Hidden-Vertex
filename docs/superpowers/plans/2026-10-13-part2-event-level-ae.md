# Part 2 Event-Level Graph Autoencoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a benchmark harness and an unsupervised event-level graph autoencoder for the LHC Olympics 2020 R&D data, run it on Kaggle, and report honestly whether it beats the simple particle-level baselines (best: Σ pT, AUC ≈ 0.79).

**Architecture:** A model-agnostic benchmark layer (`src/benchmarks`) turns any per-event score into AUC, bootstrap CI and within-bin AUC. An event store caches pT-sorted padded particle arrays plus full-event summary quantities. `EventAE` works on dense padded GPU tensors with an on-the-fly ΔR < 0.4 adjacency (no PyG), is trained with a symmetric Chamfer loss on the 48 hardest particles, and yields three scores: reconstruction error, kNN distance in latent space, and Gaussian-mixture likelihood in latent space. Stage 0 (harness only) must reproduce the verified Part 1 numbers before any model result is trusted.

**Tech Stack:** Python 3.12, PyTorch (dense tensors), scikit-learn (AUC, kNN, GMM), h5py/hdf5plugin, pandas (jet features), PyYAML, pytest; Kaggle CLI for 2×T4 runs.

**Spec:** `docs/superpowers/specs/2026-10-13-part2-event-level-ae-design.md` (read it first; this plan implements it, including the Stage 0–3 gates and the pass/fail rule in its Section 1).

## Global Constraints

- **Branch:** work on `part2-event-level-ae` (already created). Do not touch `main`.
- **Commits:** plain messages, **no `Co-Authored-By` line and no "Generated with Claude Code" footer** (user preference, overrides the default attribution). Verify with `git log -1 --format=%B | grep -ci co-authored` → `0`. Per-task commits are part of this plan; confirm with the user once before the first one if they have not approved auto-committing.
- **Do not modify Part 1 behavior.** The only edit to Part 1 code is the additive change in Task 4 (save both score reductions). Part 1 tests (`tests/test_part1_pipeline.py`, 20 tests) must keep passing.
- **Data file:** `events_anomalydetection_v2.h5` is sorted: rows 0–999,999 background, 1,000,000–1,099,999 signal. Row layout: 700 particles × (pT, η, φ) = 2100 columns, label in column 2100.
- **Frozen test set:** background rows 40,000–44,999 and signal rows 1,000,000–1,004,999 (5,000 + 5,000). Train rows are background rows from 45,000 on; contamination signal from rows 1,005,000 on.
- **Score orientation:** every score (baseline or model) is "higher = more anomalous".
- **Honesty rules (spec Section 5):** the headline configuration is `configs/part2_main.yaml` as committed in Task 4/10 and must not be tuned on test AUC. Early stopping uses only the pure-background validation slice. Ablations are sensitivity analysis only. The gate rule is in `evaluate_gate` and is not changed after results are seen.
- **Numbers to reproduce in Stage 0 (spec Section 5):** HT 0.792, mjj 0.788, Part 1 max reduction ≈ 0.478, Part 1 mean reduction ≈ 0.781 (seed 0), each inside the harness's own CI.
- **Local test environment:** `.venv` in the repo root (gitignored). Run tests with `.venv/bin/python -m pytest`.
- **Kaggle:** user `viveriun`; the CLI cannot cancel runs (the user stops sessions in the web UI); always pass a timeout (`-t`); 2×T4, one training job per GPU; quota 30 h/week.

## File Structure

| File | Responsibility |
|---|---|
| `src/benchmarks/__init__.py` | package marker |
| `src/benchmarks/report.py` | AUC, bootstrap CI (single score or seed list), within-bin AUC, benchmark rows, gate, reference check, Markdown/JSON output. No model knowledge. |
| `src/benchmarks/baselines.py` | Label-free baseline scores from event summaries (`particle_baselines`) and from the jet-feature file (`jet_baselines`, `load_jet_features`) |
| `src/data/event_store.py` | Row planning, label scan, HDF5 block reader, padded particle cache, `EventSet` load/save |
| `src/data/event_features.py` | `particle_summary`, per-particle features, feature statistics, decoder targets, `EventBatch`, `make_event_batch` |
| `src/data/dense_graph.py` | Dense ΔR adjacency on padded tensors |
| `src/model/event_ae.py` | `EventAE` (`forward(batch) -> (reconstruction, z)`) |
| `src/training/set_loss.py` | Chamfer loss with masking and optional pT weights |
| `src/training/event_trainer.py` | `BatchContext`, `per_event_loss`, `infer`, `train_event_ae`, `TrainingError` |
| `src/utils/latent_scores.py` | `KNNScorer`, `GMMScorer`, `fit_scorers` (fit on training latents only) |
| `src/pipeline/__init__.py`, `src/pipeline/part2.py` | `load_config`, `apply_overrides`, `prepare_cache`, `run_training`, `build_bank`, `make_report` |
| `scripts/part2_run.py` | argparse CLI: `prepare`, `bank`, `train`, `report` |
| `scripts/part2_sweep.py` | stage runner (`main`, `ablations`, `contamination`) with per-run skip-if-exists |
| `scripts/kaggle_bundle.py`, `scripts/kaggle/part2_*.py` | Kaggle script-kernel bundler and one body per stage |
| `configs/part2_main.yaml` | headline configuration |
| `tests/conftest.py` | synthetic HDF5 fixture shared by Part 2 tests |
| `tests/test_part2_*.py`, `tests/test_kaggle_bundle.py` | tests, one file per unit |
| `docs/PART2_RESULTS.md` | final results write-up (Task 13) |

---

### Task 0: Local test environment

**Files:** none (environment only).

- [ ] **Step 1: Create the venv and install dependencies**

```bash
cd /home/jamil/Projects/Research_Project/Hidden-Vertex
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cpu "torch==2.5.1+cpu" \
  torch-geometric h5py hdf5plugin numpy pyyaml tqdm scikit-learn matplotlib pytest tables pandas scipy
```

- [ ] **Step 2: Confirm the Part 1 baseline tests pass**

Run: `.venv/bin/python -m pytest tests -q`
Expected: `20 passed`

No commit (nothing tracked changes; `.venv/` is gitignored).

---

### Task 1: Benchmark report module

**Files:**
- Create: `src/benchmarks/__init__.py` (empty), `src/benchmarks/report.py`
- Test: `tests/test_part2_report.py`

**Interfaces:**
- Produces (all used by later tasks):
  - `mean_auc(y, scores) -> float`
  - `auc_ci(y, scores, level=0.95, n_boot=1000, seed=0) -> (lo, hi)`
  - `binned_auc(y, scores, covariate, n_bins=10, min_per_class=20) -> list[{'lo','hi','n','auc'}]`
  - `benchmark_row(name, y, scores, covariates, group, level, n_boot, n_bins=10, notes='') -> dict` with keys `name, group, auc, ci_lo, ci_hi, level, notes, within_bin`
  - `within_bin_min(row) -> float`
  - `evaluate_gate(rows, min_within_bin=0.6) -> dict` with `bar_name, bar_auc, min_within_bin, models, passed`
  - `check_reference(rows, expected) -> dict`
  - `render_markdown(rows, gate=None, reference=None, title='Benchmark') -> str`
  - `write_report(rows, gate, reference, json_path, md_path, title, meta=None)`
- `scores` anywhere means one 1-D array, or a list/tuple of arrays (one per seed; results are seed-averaged).
- Row `group` values: `'particle'`, `'jet'`, `'oracle'`, `'external'`, `'model'` (headline model), `'variant'` (ablation or contamination model rows).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_part2_report.py
import json

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.benchmarks.report import (
    auc_ci,
    benchmark_row,
    binned_auc,
    check_reference,
    evaluate_gate,
    mean_auc,
    render_markdown,
    within_bin_min,
    write_report,
)


def _data(n=4000, seed=0, shift=1.0):
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    return y, rng.normal(size=n) + shift * y


def _row(name, group, auc, lo, within):
    return {
        'name': name, 'group': group, 'auc': auc, 'ci_lo': lo, 'ci_hi': auc + 0.01,
        'level': 0.95, 'notes': '',
        'within_bin': {'ht': [{'lo': 0.0, 'hi': 1.0, 'n': 100, 'auc': within}]},
    }


def test_mean_auc_matches_sklearn():
    y, s = _data()
    assert mean_auc(y, s) == pytest.approx(roc_auc_score(y, s))


def test_perfect_and_inverted_scores():
    y = np.r_[np.zeros(50), np.ones(50)]
    s = y.copy()
    assert mean_auc(y, s) == 1.0
    assert mean_auc(y, -s) == 0.0
    assert auc_ci(y, s, n_boot=50) == (1.0, 1.0)


def test_ci_contains_point_estimate_and_widens_with_level():
    y, s = _data()
    lo95, hi95 = auc_ci(y, s, level=0.95, n_boot=300)
    lo99, hi99 = auc_ci(y, s, level=0.99, n_boot=300)
    assert lo95 < mean_auc(y, s) < hi95
    assert lo99 <= lo95 and hi99 >= hi95


def test_seed_list_is_averaged():
    y, s1 = _data(seed=1)
    _, s2 = _data(seed=2)
    expected = (roc_auc_score(y, s1) + roc_auc_score(y, s2)) / 2
    assert mean_auc(y, [s1, s2]) == pytest.approx(expected)
    lo, hi = auc_ci(y, [s1, s2], n_boot=200)
    assert lo < expected < hi


def test_binned_auc_removes_covariate_information():
    rng = np.random.default_rng(0)
    n = 20000
    cov = rng.normal(size=n)
    y = (cov + rng.normal(size=n) > 0).astype(int)
    assert roc_auc_score(y, cov) > 0.75
    bins = binned_auc(y, cov, cov, n_bins=5)
    assert len(bins) == 5
    assert all(b['auc'] < 0.7 for b in bins)


def test_binned_auc_skips_bins_without_both_classes():
    y = np.r_[np.zeros(100), np.ones(100)]
    cov = np.r_[np.arange(100), np.arange(100, 200)]
    s = np.random.default_rng(0).normal(size=200)
    assert binned_auc(y, s, cov, n_bins=4, min_per_class=20) == []


def test_benchmark_row_and_within_bin_min():
    y, s = _data(n=6000)
    cov = {'ht': np.random.default_rng(1).normal(size=6000)}
    row = benchmark_row('toy', y, s, cov, 'particle', level=0.95, n_boot=50, notes='x')
    assert row['name'] == 'toy' and row['group'] == 'particle' and row['notes'] == 'x'
    assert row['ci_lo'] <= row['auc'] <= row['ci_hi']
    assert within_bin_min(row) == min(b['auc'] for b in row['within_bin']['ht'])


def test_gate_uses_best_particle_baseline_and_ignores_jet_rows():
    rows = [_row('HT', 'particle', 0.79, 0.78, 0.7),
            _row('mjj', 'jet', 0.85, 0.84, 0.7),
            _row('m', 'model', 0.80, 0.795, 0.65)]
    gate = evaluate_gate(rows)
    assert gate['bar_name'] == 'HT'
    assert gate['passed'] is True

    rows[2] = _row('m', 'model', 0.80, 0.785, 0.65)   # CI overlaps the bar
    assert evaluate_gate(rows)['passed'] is False

    rows[2] = _row('m', 'model', 0.82, 0.81, 0.55)    # beats the bar, but within-bin AUC too low
    gate = evaluate_gate(rows)
    assert gate['models'][0]['beats_bar'] is True
    assert gate['passed'] is False


def test_gate_requires_both_row_kinds():
    with pytest.raises(ValueError):
        evaluate_gate([_row('HT', 'particle', 0.79, 0.78, 0.7)])


def test_check_reference():
    rows = [_row('HT', 'particle', 0.79, 0.78, 0.7)]
    rows[0]['ci_hi'] = 0.80
    ref = check_reference(rows, {'HT': 0.792, 'missing': 0.5})
    assert ref['HT']['ok'] is True
    assert ref['missing']['ok'] is False and ref['missing']['missing'] is True
    assert check_reference(rows, {'HT': 0.9})['HT']['ok'] is False


def test_render_and_write(tmp_path):
    rows = [_row('HT', 'particle', 0.79, 0.78, 0.7), _row('m', 'model', 0.80, 0.795, 0.65)]
    gate = evaluate_gate(rows)
    md = render_markdown(rows, gate, check_reference(rows, {'HT': 0.79}), title='T')
    assert '| HT |' in md and 'PASSED' in md
    write_report(rows, gate, None, tmp_path / 'r.json', tmp_path / 'r.md', 'T', meta={'a': 1})
    assert json.loads((tmp_path / 'r.json').read_text())['meta'] == {'a': 1}
    assert (tmp_path / 'r.md').read_text().startswith('# T')
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_part2_report.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'src.benchmarks'`

- [ ] **Step 3: Implement**

```python
# src/benchmarks/report.py
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
```

Also create the empty package marker: `touch src/benchmarks/__init__.py`.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_part2_report.py -q`
Expected: `11 passed`

- [ ] **Step 5: Commit**

```bash
git add src/benchmarks tests/test_part2_report.py
git commit -m "Add model-agnostic benchmark report: AUC, bootstrap CI, within-bin AUC, gate"
```

---

### Task 2: Event features, summaries and dense graph

**Files:**
- Create: `src/data/event_features.py`, `src/data/dense_graph.py`
- Test: `tests/test_part2_features.py`, `tests/test_part2_graph.py`

**Interfaces:**
- Produces:
  - `SUMMARY_KEYS = ('ht','multiplicity','mean_pt','max_pt','m_all')`
  - `particle_summary(particles: np.ndarray[N,P,3]) -> dict[str, np.ndarray[N]]` (pT == 0 means padding; works on full 700-slot arrays)
  - `wrap_delta_phi(d: Tensor) -> Tensor`
  - `build_event_features(particles: Tensor[B,P,3], mask: Tensor[B,P] bool) -> Tensor[B,P,7]` (columns `FEATURE_NAMES = ['log_pt','eta','phi','log_e','pt_frac','d_eta_lead','d_phi_lead']`; zero at padded slots; expects pT-sorted slots, slot 0 = leading particle)
  - `FeatureStats(mean, std)` with `.to(device)`; `fit_feature_stats(particles: Tensor[N,P,3], chunk=20000) -> FeatureStats`
  - `build_targets(particles, mask, k) -> (target Tensor[B,k,4], target_mask Tensor[B,k] bool)` (columns: standardized log pT, standardized η, cos φ, sin φ)
  - `EventBatch(NamedTuple)`: `feats, mask, adj, pt_frac, target, target_mask, target_pt`
  - `make_event_batch(particles, stats, k, radius=0.4, wrap_phi=True) -> EventBatch`
  - `radius_adjacency(eta, phi, mask, radius=0.4, wrap_phi=True) -> Tensor[B,P,P] bool` (no self loops, no padded slots)
- Standardization constants for decoder targets: `LOGPT_MEAN, LOGPT_STD = 1.1831, 1.5210`; `ETA_MEAN, ETA_STD = 0.0012, 1.4121` (the Part 1 notebook constants).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_part2_features.py
import math

import numpy as np
import pytest
import torch

from src.data.event_features import (
    FEATURE_NAMES,
    EventBatch,
    build_event_features,
    build_targets,
    fit_feature_stats,
    make_event_batch,
    particle_summary,
    wrap_delta_phi,
)


def _particles():
    # event 0: three particles (pT-sorted), event 1: one particle, event 2: empty
    p = torch.zeros(3, 4, 3)
    p[0, :3] = torch.tensor([[30.0, 0.5, 3.1], [20.0, -0.5, -3.1], [10.0, 1.0, 0.0]])
    p[1, 0] = torch.tensor([50.0, 0.0, 1.0])
    return p


def test_particle_summary_values():
    particles = np.zeros((1, 5, 3))
    particles[0, :2] = [[10.0, 0.0, 0.0], [30.0, 1.0, 2.0]]
    s = particle_summary(particles)
    assert s['ht'][0] == 40.0 and s['multiplicity'][0] == 2 and s['mean_pt'][0] == 20.0
    assert s['max_pt'][0] == 30.0
    pt = np.array([10.0, 30.0]); eta = np.array([0.0, 1.0]); phi = np.array([0.0, 2.0])
    e = (pt * np.cosh(eta)).sum()
    px, py, pz = (pt * np.cos(phi)).sum(), (pt * np.sin(phi)).sum(), (pt * np.sinh(eta)).sum()
    assert s['m_all'][0] == pytest.approx(np.sqrt(e**2 - px**2 - py**2 - pz**2))


def test_particle_summary_empty_event_is_finite():
    s = particle_summary(np.zeros((1, 5, 3)))
    assert all(np.isfinite(v).all() for v in s.values())
    assert s['multiplicity'][0] == 0 and s['mean_pt'][0] == 0


def test_wrap_delta_phi():
    d = torch.tensor([(math.pi - 0.1) - (-math.pi + 0.1), 0.3, -0.3])
    out = wrap_delta_phi(d)
    assert torch.allclose(out, torch.tensor([-0.2, 0.3, -0.3]), atol=1e-6)


def test_build_event_features_masks_padding_and_wraps_phi():
    p = _particles()
    mask = p[..., 0] > 0
    f = build_event_features(p, mask)
    assert f.shape == (3, 4, len(FEATURE_NAMES))
    assert torch.all(f[~mask] == 0)
    i = {n: k for k, n in enumerate(FEATURE_NAMES)}
    assert torch.allclose(f[0, :3, i['pt_frac']].sum(), torch.tensor(1.0))
    assert f[0, 0, i['d_eta_lead']] == 0 and f[0, 0, i['d_phi_lead']] == 0
    # particle 1 is at phi=-3.1, leader at 3.1: wrapped delta is +0.0832, not -6.2
    assert abs(f[0, 1, i['d_phi_lead']].item() - (2 * math.pi - 6.2)) < 1e-5
    assert abs(f[0, 0, i['log_pt']].item() - math.log(30.0)) < 1e-6
    assert torch.isfinite(f).all()


def test_build_targets_masks_events_with_fewer_than_k_particles():
    p = _particles()
    mask = p[..., 0] > 0
    target, tmask = build_targets(p, mask, k=4)
    assert target.shape == (3, 4, 4) and tmask.sum(1).tolist() == [3, 1, 0]
    assert torch.all(target[~tmask] == 0)
    norm = target[..., 2] ** 2 + target[..., 3] ** 2
    assert torch.allclose(norm[tmask], torch.ones(4))


def test_fit_feature_stats_matches_direct_computation():
    torch.manual_seed(0)
    p = torch.zeros(50, 10, 3)
    n = torch.randint(2, 10, (50,))
    for i in range(50):
        k = int(n[i])
        pt = torch.sort(torch.rand(k) * 100 + 1, descending=True).values
        p[i, :k] = torch.stack([pt, torch.randn(k), torch.rand(k) * 6 - 3], -1)
    mask = p[..., 0] > 0
    feats = build_event_features(p, mask)[mask]
    stats = fit_feature_stats(p, chunk=7)
    assert torch.allclose(stats.mean, feats.mean(0), atol=1e-4)
    assert torch.allclose(stats.std, feats.std(0, unbiased=False), atol=1e-4)


def test_make_event_batch_shapes():
    p = _particles()
    stats = fit_feature_stats(p)
    batch = make_event_batch(p, stats, k=3, radius=0.4, wrap_phi=True)
    assert isinstance(batch, EventBatch)
    assert batch.feats.shape == (3, 4, len(FEATURE_NAMES))
    assert batch.adj.shape == (3, 4, 4) and batch.adj.dtype == torch.bool
    assert batch.target.shape == (3, 3, 4) and batch.target_pt.shape == (3, 3)
    assert torch.all(batch.feats[~batch.mask] == 0)
```

```python
# tests/test_part2_graph.py
import numpy as np
import torch

from src.data.dataset import create_radius_graph
from src.data.dense_graph import radius_adjacency


def _event_row(rng, n=40, spread=0.6):
    row = np.zeros(2101)
    pt = rng.uniform(1, 100, n)
    eta = rng.normal(0, spread, n)
    phi = rng.uniform(-np.pi, np.pi, n)
    phi[:4] = [3.1, -3.1, 3.05, -3.14]          # force pairs that only connect when phi wraps
    row[:3 * n] = np.stack([pt, eta, phi], 1).ravel()
    return row


def _edges(adj):
    return set(map(tuple, adj.nonzero().tolist()))


def test_dense_adjacency_matches_pyg_radius_graph():
    rng = np.random.default_rng(0)
    for wrap in (True, False):
        for _ in range(5):
            row = _event_row(rng)
            graph = create_radius_graph(row, radius=0.4, wrap_phi=wrap)
            x = torch.from_numpy(row[:2100].astype(np.float32).reshape(-1, 3))
            x = x[x[:, 0] > 0][None]
            mask = torch.ones(1, x.shape[1], dtype=torch.bool)
            adj = radius_adjacency(x[..., 1], x[..., 2], mask, radius=0.4, wrap_phi=wrap)[0]
            assert _edges(adj) == set(map(tuple, graph.edge_index.t().tolist()))


def test_wrap_adds_edges_across_pi():
    eta = torch.zeros(1, 2); phi = torch.tensor([[3.1, -3.1]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    assert radius_adjacency(eta, phi, mask, wrap_phi=False).sum() == 0
    assert radius_adjacency(eta, phi, mask, wrap_phi=True).sum() == 2


def test_padding_and_diagonal_excluded():
    eta = torch.zeros(1, 3); phi = torch.zeros(1, 3)
    mask = torch.tensor([[True, True, False]])
    adj = radius_adjacency(eta, phi, mask)[0]
    assert adj.tolist() == [[False, True, False], [True, False, False], [False, False, False]]
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_part2_features.py tests/test_part2_graph.py -q`
Expected: collection errors `ModuleNotFoundError` for `src.data.event_features` / `src.data.dense_graph`

- [ ] **Step 3: Implement**

```python
# src/data/dense_graph.py
import math

import torch


def radius_adjacency(eta, phi, mask, radius=0.4, wrap_phi=True):
    """Dense ΔR <= radius adjacency on padded tensors.

    eta, phi: [B, P]; mask: [B, P] bool (False = padding). Returns [B, P, P] bool without self
    loops or padded slots. The arithmetic mirrors `create_radius_graph` in src/data/dataset.py
    (squared distance, same wrap expression) so both give identical edges.
    """
    d_eta = eta[:, :, None] - eta[:, None, :]
    d_phi = phi[:, :, None] - phi[:, None, :]
    if wrap_phi:
        d_phi = torch.remainder(d_phi + math.pi, 2 * math.pi) - math.pi
    adj = d_eta.square() + d_phi.square() <= radius * radius
    adj = adj & mask[:, :, None] & mask[:, None, :]
    eye = torch.eye(eta.shape[1], dtype=torch.bool, device=eta.device)
    return adj & ~eye
```

```python
# src/data/event_features.py
"""Event-level particle features, decoder targets and batches for the Part 2 model.

Particles arrive as dense tensors [B, P, 3] = (pT, eta, phi), sorted by pT (slot 0 is the
leading particle) and zero-padded; a slot is a real particle iff pT > 0.
"""
import math
from typing import NamedTuple

import numpy as np
import torch

from .dense_graph import radius_adjacency

SUMMARY_KEYS = ('ht', 'multiplicity', 'mean_pt', 'max_pt', 'm_all')
FEATURE_NAMES = ['log_pt', 'eta', 'phi', 'log_e', 'pt_frac', 'd_eta_lead', 'd_phi_lead']
PT_FRAC_INDEX = FEATURE_NAMES.index('pt_frac')

# Part 1 notebook constants, reused to put decoder targets on a unit scale
LOGPT_MEAN, LOGPT_STD = 1.1831, 1.5210
ETA_MEAN, ETA_STD = 0.0012, 1.4121


def particle_summary(particles):
    """Label-free event quantities from [N, P, 3] arrays (pT == 0 is padding). Use full, untruncated
    events so multiplicity and HT are exact."""
    particles = np.asarray(particles, dtype=np.float64)
    pt, eta, phi = particles[..., 0], particles[..., 1], particles[..., 2]
    n = (pt > 0).sum(axis=1)
    ht = pt.sum(axis=1)
    e = (pt * np.cosh(eta)).sum(axis=1)
    px = (pt * np.cos(phi)).sum(axis=1)
    py = (pt * np.sin(phi)).sum(axis=1)
    pz = (pt * np.sinh(eta)).sum(axis=1)
    return {
        'ht': ht,
        'multiplicity': n.astype(np.float64),
        'mean_pt': ht / np.maximum(n, 1),
        'max_pt': pt.max(axis=1),
        'm_all': np.sqrt(np.maximum(e**2 - px**2 - py**2 - pz**2, 0.0)),
    }


def wrap_delta_phi(d):
    return torch.remainder(d + math.pi, 2 * math.pi) - math.pi


def build_event_features(particles, mask):
    """Raw (unstandardized) per-particle features [B, P, 7]; zero at padded slots."""
    pt, eta, phi = particles.unbind(-1)
    safe_pt = torch.where(mask, pt, torch.ones_like(pt))
    log_pt = torch.log(safe_pt)
    log_e = torch.log(safe_pt * torch.cosh(eta))
    ht = (pt * mask).sum(1, keepdim=True).clamp(min=1e-8)
    pt_frac = pt * mask / ht
    d_eta = eta - eta[:, :1]
    d_phi = wrap_delta_phi(phi - phi[:, :1])
    feats = torch.stack([log_pt, eta, phi, log_e, pt_frac, d_eta, d_phi], dim=-1)
    return feats * mask.unsqueeze(-1)


class FeatureStats:
    """Per-feature mean/std over real particles of the training set."""

    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def to(self, device):
        return FeatureStats(self.mean.to(device), self.std.to(device))


@torch.no_grad()
def fit_feature_stats(particles, chunk=20000):
    n_feat = len(FEATURE_NAMES)
    total = torch.zeros(n_feat, dtype=torch.float64)
    total_sq = torch.zeros(n_feat, dtype=torch.float64)
    count = 0
    for start in range(0, len(particles), chunk):
        p = particles[start:start + chunk]
        mask = p[..., 0] > 0
        f = build_event_features(p, mask)[mask].double().cpu()
        total += f.sum(0)
        total_sq += (f * f).sum(0)
        count += f.shape[0]
    mean = total / count
    std = (total_sq / count - mean * mean).clamp(min=1e-12).sqrt()
    return FeatureStats(mean.float(), std.float().clamp(min=1e-6))


def standardize(feats, mask, stats):
    return ((feats - stats.mean) / stats.std) * mask.unsqueeze(-1)


def build_targets(particles, mask, k):
    """Decoder targets for the k hardest particles: (std log pT, std eta, cos phi, sin phi)."""
    if particles.shape[1] < k:
        raise ValueError(f'need at least k={k} particle slots, got {particles.shape[1]}')
    pt, eta, phi = particles[:, :k].unbind(-1)
    m = mask[:, :k]
    safe_pt = torch.where(m, pt, torch.ones_like(pt))
    target = torch.stack([
        (torch.log(safe_pt) - LOGPT_MEAN) / LOGPT_STD,
        (eta - ETA_MEAN) / ETA_STD,
        torch.cos(phi),
        torch.sin(phi),
    ], dim=-1)
    return target * m.unsqueeze(-1), m


class EventBatch(NamedTuple):
    feats: torch.Tensor        # [B, P, 7] standardized, zero at padding
    mask: torch.Tensor         # [B, P] bool
    adj: torch.Tensor          # [B, P, P] bool
    pt_frac: torch.Tensor      # [B, P] pT / HT, zero at padding
    target: torch.Tensor       # [B, k, 4]
    target_mask: torch.Tensor  # [B, k] bool
    target_pt: torch.Tensor    # [B, k] raw pT of the targets (weights for the pT-weighted loss)


def make_event_batch(particles, stats, k, radius=0.4, wrap_phi=True):
    mask = particles[..., 0] > 0
    raw = build_event_features(particles, mask)
    target, target_mask = build_targets(particles, mask, k)
    return EventBatch(
        feats=standardize(raw, mask, stats),
        mask=mask,
        adj=radius_adjacency(particles[..., 1], particles[..., 2], mask, radius, wrap_phi),
        pt_frac=raw[..., PT_FRAC_INDEX],
        target=target,
        target_mask=target_mask,
        target_pt=particles[:, :k, 0] * target_mask,
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_part2_features.py tests/test_part2_graph.py -q`
Expected: `10 passed`

- [ ] **Step 5: Commit**

```bash
git add src/data/event_features.py src/data/dense_graph.py tests/test_part2_features.py tests/test_part2_graph.py
git commit -m "Add event features, decoder targets and dense radius adjacency for Part 2"
```

---

### Task 3: Event store and baselines

**Files:**
- Create: `src/data/event_store.py`, `src/benchmarks/baselines.py`, `tests/conftest.py`
- Test: `tests/test_part2_store.py`, `tests/test_part2_baselines.py`

**Interfaces:**
- Consumes: `particle_summary`, `SUMMARY_KEYS` from `src/data/event_features.py`; `open_events`, `LABEL_COLUMN`, `NUM_PARTICLE_COLUMNS` from `src/data/dataset.py`.
- Produces (`event_store.py`):
  - `ROLES = ('train_bg','test_bg','test_sig','contam_sig')`
  - `plan_rows(labels, skip_bg=40000, n_test_bg=5000, n_test_sig=5000, n_train=200000, n_contam_sig=0) -> dict[role, np.ndarray[int64]]`
  - `read_labels(h5_path, cache_path=None, block=50000) -> np.ndarray[int8]`
  - `iter_row_blocks(dset, rows, block=10000)`
  - `EventSet` dataclass: `particles [N,P,3] float32 (pT-sorted, zero-padded)`, `rows [N]`, `labels [N]`, `summary dict[str,np.ndarray[N]]`
  - `build_role(h5_path, rows, max_particles, expected_label=None, block=10000) -> EventSet`
  - `save_role(path, es)`, `load_role(cache_dir, role) -> EventSet`
  - `build_cache(h5_path, cache_dir, split, max_particles, max_truncated_fraction=0.02) -> plan`; `split` = kwargs of `plan_rows`
- Produces (`baselines.py`):
  - `Baseline` NamedTuple `(name, scores, group, notes)`
  - `particle_baselines(summary) -> list[Baseline]` (group `'particle'`)
  - `load_jet_features(path, rows, labels) -> pandas.DataFrame` (checks label agreement)
  - `jet_baselines(df, oracle_mass=3500.0) -> list[Baseline]` (groups `'jet'` and `'oracle'`)
- Baseline names (used by config and tests): `HT (sum pT)`, `mean pT per particle`, `leading pT`, `-multiplicity`, `m_all`, `mjj`, `mj1`, `mj2`, `-tau21 (leading jet)`, `-|mjj-3500| (oracle)`.

- [ ] **Step 1: Write the shared fixture**

```python
# tests/conftest.py
import h5py
import numpy as np
import pytest


def make_events(n_bg, n_sig, seed=0):
    """Sorted synthetic LHCO-like rows (bg first, then signal); signal = fewer, harder particles."""
    rng = np.random.default_rng(seed)
    rows = np.zeros((n_bg + n_sig, 2101))
    for i in range(n_bg + n_sig):
        sig = i >= n_bg
        n = int(rng.integers(8, 15)) if sig else int(rng.integers(15, 40))
        pt = rng.exponential(30.0 if sig else 10.0, n) + 1.0
        eta = rng.normal(0.0, 1.0, n)
        phi = rng.uniform(-np.pi, np.pi, n)
        rows[i, :3 * n] = np.stack([pt, eta, phi], 1).ravel()
        rows[i, 2100] = float(sig)
    return rows


@pytest.fixture
def events_h5(tmp_path):
    """HDF5 file with 200 background rows followed by 60 signal rows, in the LHCO layout."""
    path = tmp_path / 'events.h5'
    with h5py.File(path, 'w') as f:
        f.create_group('df').create_dataset('block0_values', data=make_events(200, 60))
    return str(path)
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_part2_store.py
import numpy as np
import pytest

from src.data.event_store import (
    build_cache,
    build_role,
    iter_row_blocks,
    load_role,
    plan_rows,
    read_labels,
)


def test_plan_rows_layout():
    labels = np.r_[np.zeros(100, int), np.ones(30, int)]
    plan = plan_rows(labels, skip_bg=10, n_test_bg=5, n_test_sig=5, n_train=20, n_contam_sig=4)
    assert plan['test_bg'].tolist() == list(range(10, 15))
    assert plan['train_bg'].tolist() == list(range(15, 35))
    assert plan['test_sig'].tolist() == list(range(100, 105))
    assert plan['contam_sig'].tolist() == list(range(105, 109))


def test_plan_rows_is_disjoint_and_label_correct_on_shuffled_labels():
    rng = np.random.default_rng(0)
    labels = rng.permutation(np.r_[np.zeros(300, int), np.ones(80, int)])
    plan = plan_rows(labels, skip_bg=20, n_test_bg=30, n_test_sig=30, n_train=100, n_contam_sig=20)
    allrows = np.concatenate(list(plan.values()))
    assert len(np.unique(allrows)) == len(allrows)
    assert (labels[plan['train_bg']] == 0).all() and (labels[plan['test_bg']] == 0).all()
    assert (labels[plan['test_sig']] == 1).all() and (labels[plan['contam_sig']] == 1).all()


def test_plan_rows_raises_when_quota_unmet():
    with pytest.raises(ValueError, match='signal'):
        plan_rows(np.zeros(500, int), skip_bg=10, n_test_bg=5, n_test_sig=5, n_train=20)


def test_iter_row_blocks_returns_requested_rows_in_any_order(events_h5):
    import h5py
    rows = np.array([205, 3, 150, 4, 3])
    with h5py.File(events_h5) as f:
        dset = f['df']['block0_values']
        out = np.zeros((len(rows), 2101))
        for positions, chunk in iter_row_blocks(dset, rows, block=64):
            out[positions] = chunk
        assert np.array_equal(out, dset[:][rows])


def test_read_labels_uses_and_writes_cache(events_h5, tmp_path):
    cache = tmp_path / 'labels.npy'
    labels = read_labels(events_h5, str(cache), block=50)
    assert labels.shape == (260,) and labels[:200].sum() == 0 and labels[200:].sum() == 60
    assert cache.exists()
    assert np.array_equal(read_labels(events_h5, str(cache)), labels)


def test_build_role_sorts_pads_and_keeps_exact_summary(events_h5):
    rows = np.array([3, 0, 150])
    es = build_role(events_h5, rows, max_particles=10, expected_label=0)
    assert es.particles.shape == (3, 10, 3) and es.particles.dtype == np.float32
    pt = es.particles[..., 0]
    assert (np.diff(pt, axis=1) <= 0).all()                      # descending, zeros at the end
    assert (es.summary['multiplicity'] > 10).any()               # truncated events keep true multiplicity
    assert (es.summary['ht'] > pt.sum(1) - 1e-3).all()
    assert np.array_equal(es.rows, rows)


def test_build_role_checks_expected_label(events_h5):
    with pytest.raises(ValueError, match='label'):
        build_role(events_h5, np.array([210]), max_particles=10, expected_label=0)


def test_build_cache_roundtrip_and_guards(events_h5, tmp_path):
    split = dict(skip_bg=10, n_test_bg=30, n_test_sig=30, n_train=100, n_contam_sig=10)
    cache = str(tmp_path / 'cache')
    plan = build_cache(events_h5, cache, split, max_particles=40)
    sig = load_role(cache, 'test_sig')
    assert sig.rows[0] == 200 and (sig.labels == 1).all() and sig.particles.shape == (30, 40, 3)
    assert len(load_role(cache, 'train_bg').rows) == 100 and len(plan['contam_sig']) == 10
    with pytest.raises(RuntimeError, match='truncated'):
        build_cache(events_h5, str(tmp_path / 'c2'), split, max_particles=5)
    with pytest.raises(RuntimeError, match='different'):
        build_cache(events_h5, cache, split, max_particles=41)
```

```python
# tests/test_part2_baselines.py
import numpy as np
import pandas as pd
import pytest

from src.benchmarks.baselines import jet_baselines, load_jet_features, particle_baselines


def test_particle_baselines_orientation():
    summary = {'ht': np.array([10.0, 20]), 'multiplicity': np.array([4.0, 2]), 'mean_pt': np.array([2.5, 10]),
               'max_pt': np.array([5.0, 15]), 'm_all': np.array([1.0, 3])}
    got = {b.name: b for b in particle_baselines(summary)}
    assert set(got) == {'HT (sum pT)', 'mean pT per particle', 'leading pT', '-multiplicity', 'm_all'}
    assert all(b.group == 'particle' for b in got.values())
    assert got['-multiplicity'].scores.tolist() == [-4.0, -2.0]
    assert got['HT (sum pT)'].scores.tolist() == [10.0, 20.0]


def _jets():
    return pd.DataFrame({
        'mj1': [0.0, 50.0], 'pxj1': [100.0, 100.0], 'pyj1': [0.0, 0.0], 'pzj1': [0.0, 0.0],
        'mj2': [0.0, 60.0], 'pxj2': [-100.0, -100.0], 'pyj2': [0.0, 0.0], 'pzj2': [0.0, 0.0],
        'tau1j1': [0.5, 0.0], 'tau2j1': [0.25, 0.3], 'label': [0, 1],
    })


def test_jet_baselines_values():
    got = {b.name: b for b in jet_baselines(_jets(), oracle_mass=200.0)}
    assert got['mjj'].scores[0] == pytest.approx(200.0)
    assert got['-|mjj-200| (oracle)'].scores[0] == pytest.approx(0.0)
    assert got['-|mjj-200| (oracle)'].group == 'oracle'
    assert got['-tau21 (leading jet)'].scores.tolist() == [-0.5, -1.0]   # tau1 == 0 -> tau21 := 1
    assert got['mj1'].group == 'jet'


def test_load_jet_features_checks_label_alignment(tmp_path):
    df = _jets()
    path = tmp_path / 'f.h5'
    df.to_hdf(path, key='df')
    out = load_jet_features(str(path), np.array([1, 0]), np.array([1, 0]))
    assert out['mj1'].tolist() == [50.0, 0.0]
    with pytest.raises(ValueError, match='labels'):
        load_jet_features(str(path), np.array([0, 1]), np.array([1, 0]))
```

- [ ] **Step 3: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_part2_store.py tests/test_part2_baselines.py -q`
Expected: collection errors (`ModuleNotFoundError: src.data.event_store` / `src.benchmarks.baselines`)

- [ ] **Step 4: Implement**

```python
# src/data/event_store.py
"""Row planning and the compact particle cache for Part 2.

Events are stored as pT-sorted, zero-padded arrays [N, max_particles, 3] (not graphs). Event-level
baseline quantities are computed from the full 700-slot rows before any truncation.
"""
import json
import os
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from .dataset import LABEL_COLUMN, NUM_PARTICLE_COLUMNS, open_events
from .event_features import SUMMARY_KEYS, particle_summary

ROLES = ('train_bg', 'test_bg', 'test_sig', 'contam_sig')
ROLE_LABEL = {'train_bg': 0, 'test_bg': 0, 'test_sig': 1, 'contam_sig': 1}


@dataclass
class EventSet:
    particles: np.ndarray   # [N, max_particles, 3] float32
    rows: np.ndarray        # [N] HDF5 row of each event
    labels: np.ndarray      # [N] int8
    summary: dict           # SUMMARY_KEYS -> [N] float64, from the full events


def plan_rows(labels, skip_bg=40000, n_test_bg=5000, n_test_sig=5000, n_train=200000, n_contam_sig=0):
    """Chooses the HDF5 rows of each role from the label column (works for sorted or shuffled files).

    test_bg  = background events skip_bg .. skip_bg+n_test_bg (Part 1 trained on the first 40,000)
    test_sig = first n_test_sig signal events
    train_bg = the next n_train background events
    contam_sig = the n_contam_sig signal events after the test ones
    """
    labels = np.asarray(labels)
    bg = np.flatnonzero(labels == 0)
    sig = np.flatnonzero(labels == 1)
    need_bg = skip_bg + n_test_bg + n_train
    need_sig = n_test_sig + n_contam_sig
    if len(bg) < need_bg:
        raise ValueError(f'need {need_bg} background events, file has {len(bg)}')
    if len(sig) < need_sig:
        raise ValueError(f'need {need_sig} signal events, file has {len(sig)}')
    plan = {
        'test_bg': bg[skip_bg:skip_bg + n_test_bg],
        'test_sig': sig[:n_test_sig],
        'train_bg': bg[skip_bg + n_test_bg:need_bg],
        'contam_sig': sig[n_test_sig:need_sig],
    }
    allrows = np.concatenate(list(plan.values()))
    if len(np.unique(allrows)) != len(allrows):
        raise AssertionError('row overlap between roles')
    return {k: v.astype(np.int64) for k, v in plan.items()}


def read_labels(h5_path, cache_path=None, block=50000):
    if cache_path and os.path.exists(cache_path):
        return np.load(cache_path)
    h5_file, dset = open_events(h5_path)
    with h5_file:
        n = dset.shape[0]
        labels = np.empty(n, dtype=np.int8)
        for start in tqdm(range(0, n, block), desc='Reading labels'):
            labels[start:start + block] = dset[start:start + block, LABEL_COLUMN]
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
        np.save(cache_path, labels)
    return labels


def iter_row_blocks(dset, rows, block=10000):
    """Yields (positions, chunk) with chunk[i] == dset[rows[positions[i]]]; each HDF5 block is read once."""
    rows = np.asarray(rows)
    order = np.argsort(rows, kind='stable')
    sorted_rows = rows[order]
    i = 0
    while i < len(sorted_rows):
        start = (int(sorted_rows[i]) // block) * block
        j = int(np.searchsorted(sorted_rows, start + block, side='left'))
        data = dset[start:start + block]
        yield order[i:j], data[sorted_rows[i:j] - start]
        i = j


def build_role(h5_path, rows, max_particles, expected_label=None, block=10000):
    rows = np.asarray(rows, dtype=np.int64)
    n = len(rows)
    particles = np.zeros((n, max_particles, 3), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int8)
    summary = {k: np.zeros(n) for k in SUMMARY_KEYS}
    if n == 0:
        return EventSet(particles, rows, labels, summary)
    h5_file, dset = open_events(h5_path)
    with h5_file:
        for positions, chunk in tqdm(iter_row_blocks(dset, rows, block), desc='Reading events'):
            chunk = np.asarray(chunk, dtype=np.float64)
            chunk_labels = chunk[:, LABEL_COLUMN].astype(np.int8)
            if expected_label is not None and (chunk_labels != expected_label).any():
                raise ValueError(f'unexpected label in rows (expected {expected_label})')
            raw = chunk[:, :NUM_PARTICLE_COLUMNS].reshape(len(chunk), -1, 3)
            for key, value in particle_summary(raw).items():
                summary[key][positions] = value
            order = np.argsort(-raw[..., 0], axis=1, kind='stable')
            ranked = np.take_along_axis(raw, order[..., None], axis=1)[:, :max_particles]
            particles[positions] = ranked
            labels[positions] = chunk_labels
    return EventSet(particles, rows, labels, summary)


def save_role(path, es):
    np.savez(path, particles=es.particles, rows=es.rows, labels=es.labels,
             **{f'summary_{k}': v for k, v in es.summary.items()})


def load_role(cache_dir, role):
    data = np.load(os.path.join(cache_dir, f'{role}.npz'))
    return EventSet(
        particles=data['particles'], rows=data['rows'], labels=data['labels'],
        summary={k: data[f'summary_{k}'] for k in SUMMARY_KEYS},
    )


def build_cache(h5_path, cache_dir, split, max_particles, max_truncated_fraction=0.02):
    """Builds `<cache_dir>/<role>.npz` for every role (skipping existing ones) and returns the plan."""
    os.makedirs(cache_dir, exist_ok=True)
    meta = {'split': dict(split), 'max_particles': max_particles}
    meta_path = os.path.join(cache_dir, 'cache_meta.json')
    if os.path.exists(meta_path):
        if json.load(open(meta_path)) != meta:
            raise RuntimeError(f'{cache_dir} was built with different split/max_particles settings; '
                               'use another cache_dir or delete it')
    else:
        json.dump(meta, open(meta_path, 'w'))
    labels = read_labels(h5_path, os.path.join(cache_dir, 'labels.npy'))
    plan = plan_rows(labels, **split)
    for role in ROLES:
        path = os.path.join(cache_dir, f'{role}.npz')
        if os.path.exists(path):
            continue
        es = build_role(h5_path, plan[role], max_particles, ROLE_LABEL[role])
        frac = float((es.summary['multiplicity'] > max_particles).mean()) if len(es.rows) else 0.0
        print(f'{role}: {len(es.rows)} events, {frac:.2%} truncated at {max_particles} particles', flush=True)
        if frac > max_truncated_fraction:
            raise RuntimeError(f'{role}: {frac:.2%} of events are truncated at max_particles={max_particles} '
                               f'(limit {max_truncated_fraction:.0%}); raise data.max_particles')
        save_role(path, es)
    return plan
```

Note: `build_cache` writes `cache_meta.json` before building, so a failed truncation check leaves a meta file for that `cache_dir`; the test uses a fresh directory for the failing case.

```python
# src/benchmarks/baselines.py
"""Label-free baseline scores. Orientation is fixed here: higher = more anomalous.

Signs follow the physics of a heavy resonance decaying to four quarks (more energy, fewer and
harder particles, jet mass and two-prong substructure). The signs of -multiplicity and -tau21 were
first seen on labelled data in Part 1, so they are reference points, not blind predictions.
`-|mjj - m| (oracle)` uses the signal mass and is not model independent.
"""
from typing import NamedTuple

import numpy as np
import pandas as pd


class Baseline(NamedTuple):
    name: str
    scores: np.ndarray
    group: str      # 'particle' | 'jet' | 'oracle'
    notes: str


def particle_baselines(summary):
    return [
        Baseline('HT (sum pT)', np.asarray(summary['ht']), 'particle', 'more energy = anomalous'),
        Baseline('mean pT per particle', np.asarray(summary['mean_pt']), 'particle',
                 '= HT / multiplicity'),
        Baseline('leading pT', np.asarray(summary['max_pt']), 'particle', ''),
        Baseline('-multiplicity', -np.asarray(summary['multiplicity']), 'particle',
                 'fewer particles = anomalous (sign seen on labelled data in Part 1)'),
        Baseline('m_all', np.asarray(summary['m_all']), 'particle', 'invariant mass of all particles'),
    ]


def load_jet_features(path, rows, labels):
    """Jet features for the given HDF5 rows; raises if their label column disagrees with `labels`."""
    df = pd.read_hdf(path).iloc[np.asarray(rows)].reset_index(drop=True)
    if not np.array_equal(df['label'].to_numpy().astype(int), np.asarray(labels).astype(int)):
        raise ValueError('jet-feature file labels do not match the event labels (row misalignment)')
    return df


def jet_baselines(df, oracle_mass=3500.0):
    def energy(i):
        return np.sqrt(df[f'mj{i}'] ** 2 + df[f'pxj{i}'] ** 2 + df[f'pyj{i}'] ** 2 + df[f'pzj{i}'] ** 2)

    px = df['pxj1'] + df['pxj2']
    py = df['pyj1'] + df['pyj2']
    pz = df['pzj1'] + df['pzj2']
    mjj = np.sqrt(np.maximum((energy(1) + energy(2)) ** 2 - px**2 - py**2 - pz**2, 0.0)).to_numpy()
    tau1 = df['tau1j1'].to_numpy(dtype=float)
    tau2 = df['tau2j1'].to_numpy(dtype=float)
    tau21 = np.where(tau1 > 0, tau2 / np.where(tau1 > 0, tau1, 1.0), 1.0)
    mass_label = f'{oracle_mass:g}'
    return [
        Baseline('mjj', mjj, 'jet', 'dijet invariant mass'),
        Baseline('mj1', df['mj1'].to_numpy(dtype=float), 'jet', 'leading jet mass'),
        Baseline('mj2', df['mj2'].to_numpy(dtype=float), 'jet', 'second jet mass'),
        Baseline('-tau21 (leading jet)', -tau21, 'jet', 'two-prong substructure'),
        Baseline(f'-|mjj-{mass_label}| (oracle)', -np.abs(mjj - oracle_mass), 'oracle',
                 'needs the signal mass; not model independent'),
    ]
```

Create `src/benchmarks/__init__.py` if not already present (Task 1 created it).

- [ ] **Step 5: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_part2_store.py tests/test_part2_baselines.py -q`
Expected: `11 passed`

- [ ] **Step 6: Commit**

```bash
git add src/data/event_store.py src/benchmarks/baselines.py tests/conftest.py tests/test_part2_store.py tests/test_part2_baselines.py
git commit -m "Add Part 2 event store, row planning and label-free baselines"
```

---

### Task 4: Stage 0 pipeline (config, bank, report, CLI) and Part 1 score export

**Files:**
- Create: `configs/part2_main.yaml`, `src/pipeline/__init__.py` (empty), `src/pipeline/part2.py`, `scripts/part2_run.py`
- Modify: `scripts/run_part1_eval.py` (save both reductions in the scores file)
- Test: `tests/test_part2_pipeline.py` (Stage 0 part; Task 10 appends the training tests), `tests/test_part2_stage0.py`

**Interfaces:**
- Consumes: everything from Tasks 1–3.
- Produces (`src/pipeline/part2.py`):
  - `load_config(path) -> dict`, `apply_overrides(cfg, ['a.b=value', ...]) -> dict` (values parsed with `yaml.safe_load`)
  - `prepare_cache(cfg) -> plan`
  - `parse_external(spec) -> (name, path, key)` for `"name=path.npz:key"`
  - `build_bank(cfg, externals=(), bank_path=None) -> str` (writes `<output_dir>/bank.npz` + `bank.json`)
  - `make_report(cfg, bank_path, run_tags=(), name='report') -> dict` (writes `<output_dir>/<name>.json|.md`; returns `{'rows','gate','reference'}`)
- Produces (`scripts/part2_run.py`): subcommands `prepare`, `bank [--external name=path:key ...]`, `report [--tags a,b] [--name n] [--check-reference]`, plus `train` added in Task 10. Every subcommand accepts `--config` and repeated `--set key=value`.
- Config keys used downstream: see the YAML below (this is the headline configuration; do not tune it on test AUC).

- [ ] **Step 1: Write the config**

```yaml
# configs/part2_main.yaml  --  headline Part 2 configuration (spec Section 4). Not tuned on test AUC.
data:
  h5_path: "dataset/raw/events_anomalydetection_v2.h5"
  features_path: "dataset/raw/events_anomalydetection_v2.features.h5"
  cache_dir: "results/part2/cache"
  max_particles: 300            # padded width; prep fails if > max_truncated_fraction of events exceed it
  max_truncated_fraction: 0.02
  radius: 0.4
  wrap_phi: true

split:
  skip_bg: 40000                # Part 1 trained on the first 40,000 background events
  n_test_bg: 5000
  n_test_sig: 5000
  n_train: 200000               # background; the last val_fraction of it is the pure validation slice
  n_contam_sig: 16200           # = 9% of the 180k training events
  val_fraction: 0.1
  contamination: 0.0            # fraction of the training events replaced by signal

model:
  width: 64
  layers: 3
  hidden: 128
  latent_dim: 16
  k_recon: 48
  pooling: [mean, max, ptsum]

training:
  epochs: 30
  batch_size: 256
  learning_rate: 0.001
  weight_decay: 0.00001
  patience: 5
  grad_clip: 5.0
  loss: chamfer                 # chamfer | chamfer_ptw
  collapse_std: 0.0001
  seed: 0

scores:
  knn_k: 20
  gmm_components: 8

benchmark:
  n_boot: 1000
  baseline_ci_level: 0.95
  model_ci_level: 0.983         # Bonferroni for the best-of-three score rule
  n_bins: 10
  min_within_bin: 0.6
  jet_oracle_mass: 3500

stage0_expected:                # verified Part 1 numbers (docs/PART1_VERIFICATION.md)
  "HT (sum pT)": 0.792
  "mjj": 0.788
  "Part 1 AE (max)": 0.478
  "Part 1 AE (mean)": 0.781

output_dir: "results/part2"
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_part2_stage0.py
import importlib.util
import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_part1_eval_saves_both_reductions(tmp_path):
    spec = importlib.util.spec_from_file_location('run_part1_eval', os.path.join(ROOT, 'scripts/run_part1_eval.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    config = {'training': {'seed': 0}, 'evaluation': {
        'score_reduction': 'max', 'model_path': str(tmp_path / 'm.pt'), 'metrics_path': str(tmp_path / 'm.json'),
        'roc_path': str(tmp_path / 'r.png'), 'scores_path': str(tmp_path / 's.npz')}}
    labels = [0, 0, 1, 1]
    mod.save_outputs(torch.nn.Linear(1, 1), config, 0.6, (0.5, 0.7), np.array([0, 1.0]), np.array([0, 1.0]),
                     np.array([0.1, 0.2, 0.3, 0.4]), labels, None,
                     scores_by_reduction={'max': np.array([0.1, 0.2, 0.3, 0.4]), 'mean': np.array([4, 3, 2, 1.0])})
    data = np.load(tmp_path / 's.npz')
    assert set(data.files) == {'scores', 'labels', 'scores_max', 'scores_mean'}
    assert data['scores_mean'].tolist() == [4, 3, 2, 1]
```

```python
# tests/test_part2_pipeline.py
import json
import os

import numpy as np
import pytest

from src.pipeline.part2 import (
    apply_overrides,
    build_bank,
    load_config,
    make_report,
    parse_external,
    prepare_cache,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TINY = [
    'data.features_path=null', 'data.max_particles=40',
    'split.skip_bg=10', 'split.n_test_bg=30', 'split.n_test_sig=30', 'split.n_train=100', 'split.n_contam_sig=10',
    'model.width=16', 'model.layers=2', 'model.latent_dim=4', 'model.k_recon=8', 'model.hidden=32',
    'training.epochs=2', 'training.batch_size=32', 'scores.knn_k=5', 'scores.gmm_components=2',
    'benchmark.n_boot=30',
]


@pytest.fixture
def tiny_cfg(events_h5, tmp_path):
    cfg = load_config(os.path.join(ROOT, 'configs/part2_main.yaml'))
    return apply_overrides(cfg, [f'data.h5_path={events_h5}', f'data.cache_dir={tmp_path}/cache',
                                 f'output_dir={tmp_path}/out', *TINY])


def test_apply_overrides_parses_yaml_values():
    cfg = apply_overrides({'a': {'b': 1}}, ['a.b=2', 'a.c=null', 'x.y=[mean,max]', 'z=0.5'])
    assert cfg == {'a': {'b': 2, 'c': None}, 'x': {'y': ['mean', 'max']}, 'z': 0.5}


def test_parse_external():
    assert parse_external('Part 1 AE (max)=/a/b.npz:scores_max') == ('Part 1 AE (max)', '/a/b.npz', 'scores_max')


def test_bank_and_report_without_models(tiny_cfg, tmp_path):
    prepare_cache(tiny_cfg)
    scores = tmp_path / 'ext.npz'
    labels = np.r_[np.zeros(30), np.ones(30)]
    np.savez(scores, scores=np.random.default_rng(0).normal(size=60), labels=labels)
    bank = build_bank(tiny_cfg, externals=[f'Ext={scores}:scores'])
    out = make_report(tiny_cfg, bank, name='stage0')
    names = {r['name'] for r in out['rows']}
    assert {'HT (sum pT)', 'mean pT per particle', '-multiplicity', 'Ext'} <= names
    assert out['gate'] is None
    ht = next(r for r in out['rows'] if r['name'] == 'HT (sum pT)')
    assert 0.0 <= ht['auc'] <= 1.0
    assert os.path.exists(os.path.join(tiny_cfg['output_dir'], 'stage0.md'))
    assert json.load(open(os.path.join(tiny_cfg['output_dir'], 'stage0.json')))['gate'] is None


def test_bank_rejects_external_with_different_labels(tiny_cfg, tmp_path):
    prepare_cache(tiny_cfg)
    bad = tmp_path / 'bad.npz'
    np.savez(bad, scores=np.zeros(60), labels=np.zeros(60))
    with pytest.raises(ValueError, match='labels'):
        build_bank(tiny_cfg, externals=[f'Bad={bad}:scores'])
```

- [ ] **Step 3: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_part2_pipeline.py tests/test_part2_stage0.py -q`
Expected: failures (`ModuleNotFoundError: src.pipeline`; `TypeError ... scores_by_reduction`)

- [ ] **Step 4: Modify `scripts/run_part1_eval.py`**

Change the `save_outputs` signature and the `np.savez` call, and pass the new argument at the call site.

```python
def save_outputs(model, config, auc, auc_ci, fpr, tpr, y_scores, test_labels, extra=None, scores_by_reduction=None):
```

Replace
```python
        np.savez(scores_path, scores=np.asarray(y_scores), labels=np.asarray(test_labels))
```
with
```python
        np.savez(scores_path, scores=np.asarray(y_scores), labels=np.asarray(test_labels),
                 **{f'scores_{r}': np.asarray(s) for r, s in (scores_by_reduction or {}).items()})
```
and replace the last line of `main()`
```python
    save_outputs(model, config, auc, auc_ci, fpr, tpr, y_scores, test_labels, extra)
```
with
```python
    save_outputs(model, config, auc, auc_ci, fpr, tpr, y_scores, test_labels, extra, scores_by_reduction)
```

- [ ] **Step 5: Implement the pipeline (Stage 0 part)**

```python
# src/pipeline/part2.py
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
        data = np.load(path)
        if not np.array_equal(np.asarray(data['labels']).astype(int), y.astype(int)):
            raise ValueError(f'{path}: test labels differ from the Part 2 test set; scores are not comparable')
        items.append(Baseline(name, np.asarray(data[key], dtype=np.float64), 'external', f'from {os.path.basename(path)}'))
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
    bank = np.load(bank_path)
    meta = json.load(open(bank_path.replace('.npz', '.json'), encoding='utf-8'))
    y = bank['y']
    covariates = {c: bank[f'cov_{c}'] for c in COVARIATES}
    b = cfg['benchmark']
    rows = [benchmark_row(m['name'], y, bank[f's{i}'], covariates, m['group'], b['baseline_ci_level'],
                          b['n_boot'], b['n_bins'], m['notes']) for i, m in enumerate(meta)]
    for tag in run_tags:
        files = sorted(glob.glob(os.path.join(cfg['output_dir'], 'runs', tag, 'scores_s*.npz')))
        if not files:
            raise FileNotFoundError(f'no scores for run tag {tag!r} under {cfg["output_dir"]}/runs')
        runs = [np.load(f) for f in files]
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
```

```python
# scripts/part2_run.py
"""Part 2 entry point.

    python scripts/part2_run.py prepare --config configs/part2_main.yaml
    python scripts/part2_run.py bank    --config configs/part2_main.yaml --external "Part 1 AE (max)=scores.npz:scores_max"
    python scripts/part2_run.py report  --config configs/part2_main.yaml --name stage0
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import part2  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description='Part 2 pipeline')
    parser.add_argument('command', choices=['prepare', 'bank', 'train', 'report'])
    parser.add_argument('--config', default='configs/part2_main.yaml')
    parser.add_argument('--set', action='append', default=[], dest='overrides', help='key=value override')
    parser.add_argument('--external', action='append', default=[], help="bank: 'name=path.npz:key'")
    parser.add_argument('--tags', default='', help='report: comma-separated run tags')
    parser.add_argument('--name', default='report', help='report: output file stem')
    parser.add_argument('--check-reference', action='store_true', help='report: exit 2 if Stage 0 references fail')
    parser.add_argument('--seed', type=int, default=None, help='train: overrides training.seed')
    parser.add_argument('--tag', default='main', help='train: run directory name under <output_dir>/runs')
    args = parser.parse_args()

    cfg = part2.apply_overrides(part2.load_config(args.config), args.overrides)
    if args.command == 'prepare':
        part2.prepare_cache(cfg)
    elif args.command == 'bank':
        print('wrote', part2.build_bank(cfg, args.external))
    elif args.command == 'train':
        seed = args.seed if args.seed is not None else cfg['training']['seed']
        part2.run_training(cfg, seed=seed, tag=args.tag)
    elif args.command == 'report':
        tags = [t for t in args.tags.split(',') if t]
        out = part2.make_report(cfg, os.path.join(cfg['output_dir'], 'bank.npz'), tags, args.name)
        print(open(os.path.join(cfg['output_dir'], f'{args.name}.md'), encoding='utf-8').read())
        if args.check_reference and out['reference'] and not all(r['ok'] for r in out['reference'].values()):
            print('STAGE 0 REFERENCE CHECK FAILED: harness does not reproduce the verified Part 1 numbers')
            sys.exit(2)


if __name__ == '__main__':
    main()
```

- [ ] **Step 6: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_part2_pipeline.py tests/test_part2_stage0.py -q`
Expected: `5 passed` (the `train` subcommand is wired to `part2.run_training`, which Task 10 adds).

Also run the full suite: `.venv/bin/python -m pytest tests -q` → everything passes (Part 1's 20 tests included).

- [ ] **Step 7: Commit**

```bash
git add configs/part2_main.yaml src/pipeline scripts/part2_run.py scripts/run_part1_eval.py tests/test_part2_pipeline.py tests/test_part2_stage0.py
git commit -m "Add Part 2 Stage 0 pipeline: config, score bank, report CLI; export Part 1 scores for both reductions"
```

---

### Task 5: Kaggle bundler and Stage 0 run (gate: harness reproduces Part 1)

**Files:**
- Create: `scripts/kaggle_bundle.py`, `scripts/kaggle/part2_stage0.py`
- Test: `tests/test_kaggle_bundle.py`

**Interfaces:**
- Produces: `build_script(body_text, repo_root) -> str` (full kernel script) and CLI `python scripts/kaggle_bundle.py --slug S --title T --body FILE --out DIR` which writes `DIR/<slug>.py` and `DIR/kernel-metadata.json` for user `viveriun`.
- Variables available to body files at run time on Kaggle: `HV` (extracted repo, also the cwd), `H5`, `FEATURES` (path or `None`), `OUT` (`/kaggle/working/out`, saved as kernel output), `CACHE` (`/tmp/part2_cache`, not saved), and `sh(*args, **env)` which runs a command in `HV` and raises on failure.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kaggle_bundle.py
import base64
import io
import os
import re
import tarfile

from scripts.kaggle_bundle import build_script

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_bundle_contains_source_and_excludes_data_dirs():
    script = build_script("print('body')", ROOT)
    payload = re.search(r'b64decode\("([A-Za-z0-9+/=]+)"\)', script).group(1)
    names = tarfile.open(fileobj=io.BytesIO(base64.b64decode(payload)), mode='r:gz').getnames()
    assert 'src/benchmarks/report.py' in names and 'scripts/part2_run.py' in names
    assert not any(n.startswith(('results/', 'dataset/', 'notebooks/', '.git/', '.venv/')) for n in names)
    assert script.rstrip().endswith("shutil.rmtree(HV, ignore_errors=True)")
    assert "print('body')" in script
    compile(script, 'kernel.py', 'exec')
```

(`scripts` is not a package; add an empty `scripts/__init__.py` so the import works.)

- [ ] **Step 2: Run to verify it fails**

Run: `touch scripts/__init__.py && .venv/bin/python -m pytest tests/test_kaggle_bundle.py -q`
Expected: `ModuleNotFoundError: No module named 'scripts.kaggle_bundle'`

- [ ] **Step 3: Implement the bundler and the Stage 0 body**

```python
# scripts/kaggle_bundle.py
"""Bundle the repo into a Kaggle script kernel (script + kernel-metadata.json).

    python scripts/kaggle_bundle.py --slug hidden-vertex-part2-stage0 --title "Hidden Vertex Part2 Stage0" \
        --body scripts/kaggle/part2_stage0.py --out <dir>
    kaggle kernels push -p <dir> -t 3600
"""
import argparse
import base64
import io
import json
import os
import subprocess
import tarfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCLUDE_PREFIXES = ('results/', 'dataset/', 'notebooks/', 'docs/', '.venv/', '.git/')
USER = 'viveriun'
DATASET = 'atulchikku/lhc-olympics-2020-ad-r-and-d'

HEADER = '''import base64, glob, io, os, shutil, subprocess, sys, tarfile
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "hdf5plugin", "torch-geometric"], check=False)
HV = "/kaggle/working/hv"
os.makedirs(HV, exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode("__PAYLOAD__")), mode="r:gz").extractall(HV)
os.chdir(HV)
sys.path.insert(0, HV)
H5 = glob.glob("/kaggle/input/**/events_anomalydetection_v2.h5", recursive=True)[0]
FEATURES = (glob.glob("/kaggle/input/**/events_anomalydetection_v2.features.h5", recursive=True) or [None])[0]
OUT = "/kaggle/working/out"
CACHE = "/tmp/part2_cache"
os.makedirs(OUT, exist_ok=True)


def sh(*args, **env):
    """Run a command inside the extracted repo; raises on a non-zero exit."""
    subprocess.run([str(a) for a in args], cwd=HV, check=True, env=dict(os.environ, TQDM_DISABLE="1", **env))

'''
FOOTER = '\nshutil.rmtree(HV, ignore_errors=True)\n'


def _tracked_files(repo_root):
    out = subprocess.run(['git', 'ls-files', '-co', '--exclude-standard'], cwd=repo_root, check=True,
                         capture_output=True, text=True).stdout.splitlines()
    return [f for f in out if not f.startswith(EXCLUDE_PREFIXES) and os.path.isfile(os.path.join(repo_root, f))]


def build_script(body_text, repo_root=REPO):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as tar:
        for rel in _tracked_files(repo_root):
            tar.add(os.path.join(repo_root, rel), arcname=rel)
    payload = base64.b64encode(buffer.getvalue()).decode()
    return HEADER.replace('__PAYLOAD__', payload) + body_text + FOOTER


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slug', required=True)
    parser.add_argument('--title', required=True)
    parser.add_argument('--body', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    script_name = f'{args.slug}.py'
    with open(args.body, encoding='utf-8') as handle:
        script = build_script(handle.read())
    with open(os.path.join(args.out, script_name), 'w', encoding='utf-8') as handle:
        handle.write(script)
    metadata = {
        'id': f'{USER}/{args.slug}', 'title': args.title, 'code_file': script_name, 'language': 'python',
        'kernel_type': 'script', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False,
        'enable_internet': True, 'keywords': ['gpu'], 'dataset_sources': [DATASET], 'kernel_sources': [],
        'competition_sources': [], 'model_sources': [], 'machine_shape': 'NvidiaTeslaT4',
    }
    with open(os.path.join(args.out, 'kernel-metadata.json'), 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2)
    print(f'wrote {args.out}/{script_name} ({len(script) / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
```

```python
# scripts/kaggle/part2_stage0.py
# Stage 0 body (runs inside the Kaggle kernel; see scripts/kaggle_bundle.py for the variables it can use).
# Builds the event cache, re-runs Part 1 (seed 0) to export both score reductions, builds the baseline
# bank on the frozen test set, and writes the Stage 0 table with the reference check.
import shutil
import yaml

OVERRIDES = [f"data.h5_path={H5}", f"data.features_path={FEATURES}", f"data.cache_dir={CACHE}", f"output_dir={OUT}"]


def p2(command, *extra, check=True):
    flags = [x for o in OVERRIDES for x in ("--set", o)]
    cmd = [sys.executable, "scripts/part2_run.py", command, "--config", "configs/part2_main.yaml", *flags, *extra]
    if check:
        sh(*cmd)
    else:  # reference-check failure must not skip the output cleanup in the footer
        print("exit code:", subprocess.run(cmd, cwd=HV, env=dict(os.environ, TQDM_DISABLE="1")).returncode)


p2("prepare")

part1 = yaml.safe_load(open("configs/part1_notebook_parity.yaml"))
part1["data"]["h5_path"] = H5
part1["data"]["split_path"] = f"{OUT}/part1/split.pt"
part1["evaluation"].update(model_path=f"{OUT}/part1/model.pt", metrics_path=f"{OUT}/part1/metrics.json",
                           scores_path=f"{OUT}/part1/scores.npz", roc_path=f"{OUT}/part1/roc.png")
os.makedirs(f"{OUT}/part1", exist_ok=True)
yaml.safe_dump(part1, open("/tmp/part1.yaml", "w"))
sh(sys.executable, "main.py", "prepare-part1", "--config", "/tmp/part1.yaml")
sh(sys.executable, "scripts/run_part1_eval.py", "--config", "/tmp/part1.yaml", "--seed", "0")
for junk in ("split.pt", "model.pt"):
    os.remove(f"{OUT}/part1/{junk}")

p2("bank", "--external", f"Part 1 AE (max)={OUT}/part1/scores.npz:scores_max",
   "--external", f"Part 1 AE (mean)={OUT}/part1/scores.npz:scores_mean")
p2("report", "--name", "stage0", "--check-reference", check=False)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_kaggle_bundle.py -q`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/__init__.py scripts/kaggle_bundle.py scripts/kaggle tests/test_kaggle_bundle.py
git commit -m "Add Kaggle script-kernel bundler and Stage 0 kernel body"
```

- [ ] **Step 6: Run Stage 0 on Kaggle (GATE: harness must reproduce Part 1)**

```bash
S=/tmp/claude-1000/-home-jamil-Projects-Research-Project-Hidden-Vertex/b7dfad6c-7e70-40b6-a8cf-55eb13e8970a/scratchpad
.venv/bin/python scripts/kaggle_bundle.py --slug hidden-vertex-part2-stage0 --title "Hidden Vertex Part2 Stage0" \
  --body scripts/kaggle/part2_stage0.py --out $S/k_stage0
kaggle kernels push -p $S/k_stage0 -t 3600
kaggle kernels status viveriun/hidden-vertex-part2-stage0      # poll until complete (expect ~15–25 min)
kaggle kernels output viveriun/hidden-vertex-part2-stage0 -p results/part2/kaggle/stage0 -o -q
```

Expected: `results/part2/kaggle/stage0/stage0.md` exists; its reference table shows `yes` for HT 0.792, mjj 0.788, Part 1 max ≈ 0.478, Part 1 mean ≈ 0.781. If any is outside its CI the log shows `STAGE 0 REFERENCE CHECK FAILED` and `exit code: 2` (the kernel still saves its outputs).

**If the reference check fails:** stop. Do not proceed to Task 6+. Likely causes, in order: test rows differ from Part 1 (compare `rows` in the cached `test_*.npz` with Part 1's `test_indices`), jet-feature row misalignment, or a bug in `particle_summary`. Report to the user with the numbers.

Also copy the bank for local report building: `cp results/part2/kaggle/stage0/bank.npz results/part2/kaggle/stage0/bank.json results/part2/` (the local `results/` directory is gitignored data).

Report the Stage 0 table to the user (it already answers the open question: does `mean pT per particle` explain the Part 1 mean score?) before continuing.

---

### Task 6: Chamfer set loss

**Files:**
- Create: `src/training/set_loss.py`
- Test: `tests/test_part2_loss.py`

**Interfaces:**
- Produces: `chamfer_per_event(pred, target, target_mask, target_weights=None) -> Tensor[B]`, `chamfer_loss(...) -> scalar mean`.
  - `pred` `[B, K, D]` (all K slots are predictions), `target` `[B, K, D]` (zero at invalid slots), `target_mask` `[B, K] bool`.
  - Value = mean over prediction slots of squared distance to the nearest *valid* target + weighted mean over valid targets of squared distance to the nearest prediction. Events with no valid target return 0. `target_weights` `[B, K]` (e.g. raw pT) are normalized over valid targets and apply to the target→prediction term only.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_part2_loss.py
import numpy as np
import torch

from src.training.set_loss import chamfer_loss, chamfer_per_event


def _brute(pred, target, mask, weights=None):
    out = []
    for b in range(pred.shape[0]):
        valid = np.flatnonzero(mask[b])
        if len(valid) == 0:
            out.append(0.0)
            continue
        d = ((pred[b][:, None, :] - target[b][None, valid, :]) ** 2).sum(-1)   # [K, n_valid]
        pred_term = d.min(1).mean()
        w = np.ones(len(valid)) if weights is None else weights[b][valid]
        w = w / w.sum()
        out.append(pred_term + (d.min(0) * w).sum())
    return np.array(out)


def test_identical_sets_have_zero_loss_and_permutation_invariance():
    torch.manual_seed(0)
    target = torch.randn(2, 6, 4)
    mask = torch.ones(2, 6, dtype=torch.bool)
    assert torch.allclose(chamfer_per_event(target, target, mask), torch.zeros(2), atol=1e-7)
    perm = torch.randperm(6)
    assert torch.allclose(chamfer_per_event(target[:, perm], target, mask), torch.zeros(2), atol=1e-7)


def test_matches_bruteforce_with_masks_and_weights():
    torch.manual_seed(1)
    pred, target = torch.randn(5, 7, 4), torch.randn(5, 7, 4)
    mask = torch.rand(5, 7) > 0.3
    target = target * mask.unsqueeze(-1)
    w = torch.rand(5, 7) + 0.1
    for weights in (None, w):
        got = chamfer_per_event(pred, target, mask, weights).numpy()
        want = _brute(pred.numpy(), target.numpy(), mask.numpy(), None if weights is None else weights.numpy())
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


def test_invalid_target_slots_do_not_matter():
    torch.manual_seed(2)
    pred = torch.randn(1, 5, 4)
    target = torch.randn(1, 5, 4)
    mask = torch.tensor([[True, True, True, False, False]])
    a = chamfer_per_event(pred, target * mask.unsqueeze(-1), mask)
    garbage = target.clone(); garbage[:, 3:] = 1e3
    assert torch.allclose(a, chamfer_per_event(pred, garbage, mask))


def test_fewer_particles_than_k_can_be_matched_exactly():
    real = torch.randn(1, 3, 4)
    target = torch.zeros(1, 5, 4); target[:, :3] = real
    mask = torch.tensor([[True, True, True, False, False]])
    pred = torch.cat([real, real[:, :1], real[:, :1]], dim=1)     # extra slots sit on real particles
    assert chamfer_per_event(pred, target, mask).item() < 1e-10


def test_empty_event_is_zero_with_finite_gradients():
    pred = torch.randn(2, 4, 4, requires_grad=True)
    target = torch.zeros(2, 4, 4)
    mask = torch.zeros(2, 4, dtype=torch.bool); mask[1, :2] = True
    loss = chamfer_loss(pred, target, mask)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pred.grad).all()
    assert chamfer_per_event(pred, target, mask)[0].item() == 0.0
    assert torch.all(pred.grad[0] == 0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_part2_loss.py -q`
Expected: `ModuleNotFoundError: No module named 'src.training.set_loss'`

- [ ] **Step 3: Implement**

```python
# src/training/set_loss.py
import torch

_BIG = 1e9


def chamfer_per_event(pred, target, target_mask, target_weights=None):
    """Symmetric Chamfer distance per event, masked for events with fewer than K real particles.

    pred [B, K, D]: K predicted particles. target [B, K, D] with target_mask [B, K] marking real ones.
    Term 1: each prediction to its nearest *real* target (extra predictions in events with n < K are
    pulled onto real particles). Term 2: each real target to its nearest prediction, optionally weighted
    (weights normalised over the real targets). Events without real targets give 0.
    """
    diff = pred[:, :, None, :] - target[:, None, :, :]
    d = diff.square().sum(-1)                                            # [B, K_pred, K_tgt]
    has = target_mask.any(dim=1)
    pred_to_target = d.masked_fill(~target_mask[:, None, :], _BIG).min(dim=2).values
    term_pred = torch.where(has[:, None], pred_to_target, torch.zeros_like(pred_to_target)).mean(dim=1)
    target_to_pred = d.min(dim=1).values                                 # [B, K_tgt]
    w = target_mask.to(d.dtype) if target_weights is None else target_weights * target_mask
    w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return term_pred + (target_to_pred * w).sum(dim=1)


def chamfer_loss(pred, target, target_mask, target_weights=None):
    return chamfer_per_event(pred, target, target_mask, target_weights).mean()
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_part2_loss.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/training/set_loss.py tests/test_part2_loss.py
git commit -m "Add masked symmetric Chamfer loss with optional pT weights"
```

---

### Task 7: EventAE model

**Files:**
- Create: `src/model/event_ae.py`
- Test: `tests/test_part2_model.py`

**Interfaces:**
- Consumes: `EventBatch`, `make_event_batch`, `fit_feature_stats` (Task 2).
- Produces: `EventAE(in_dim=7, width=64, layers=3, hidden=128, latent_dim=16, k_recon=48, out_dim=4, pooling=('mean','max','ptsum'))`; `forward(batch: EventBatch) -> (reconstruction [B, k_recon, 4], z [B, latent_dim])`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_part2_model.py
import torch

from src.data.event_features import fit_feature_stats, make_event_batch
from src.model.event_ae import EventAE


def _particles(n_events=6, p=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = torch.zeros(n_events, p, 3)
    for i in range(n_events):
        n = int(torch.randint(3, p, (1,), generator=g))
        pt = torch.sort(torch.rand(n, generator=g) * 100 + 1, descending=True).values
        out[i, :n] = torch.stack([pt, torch.randn(n, generator=g), torch.rand(n, generator=g) * 6 - 3], -1)
    return out


def _model(**kw):
    torch.manual_seed(0)
    return EventAE(width=16, layers=2, hidden=32, latent_dim=4, k_recon=8, **kw).eval()


def test_output_shapes_and_gradients():
    p = _particles()
    batch = make_event_batch(p, fit_feature_stats(p), k=8)
    model = _model().train()
    recon, z = model(batch)
    assert recon.shape == (6, 8, 4) and z.shape == (6, 4)
    (recon.square().mean() + z.square().mean()).backward()
    assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in model.parameters())


def test_permutation_invariance():
    p = _particles()
    stats = fit_feature_stats(p)
    batch = make_event_batch(p, stats, k=8)
    model = _model()
    perm = torch.randperm(p.shape[1])
    permuted = batch._replace(
        feats=batch.feats[:, perm], mask=batch.mask[:, perm], pt_frac=batch.pt_frac[:, perm],
        adj=batch.adj[:, perm][:, :, perm],
    )
    with torch.no_grad():
        _, z = model(batch)
        _, z_perm = model(permuted)
    assert torch.allclose(z, z_perm, atol=1e-5)


def test_padding_contributes_nothing():
    p = _particles(p=20)
    stats = fit_feature_stats(p)
    padded = torch.cat([p, torch.zeros(p.shape[0], 15, 3)], dim=1)
    model = _model()
    with torch.no_grad():
        _, z = model(make_event_batch(p, stats, k=8))
        _, z_padded = model(make_event_batch(padded, stats, k=8))
    assert torch.allclose(z, z_padded, atol=1e-5)


def test_pooling_options_and_empty_event():
    p = _particles()
    p[0] = 0                                           # an event with no particles
    batch = make_event_batch(p, fit_feature_stats(_particles()), k=8)
    for pooling in (('mean', 'max'), ('mean', 'max', 'ptsum'), ('ptsum',)):
        _, z = _model(pooling=pooling)(batch)
        assert torch.isfinite(z).all()
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_part2_model.py -q`
Expected: `ModuleNotFoundError: No module named 'src.model.event_ae'`

- [ ] **Step 3: Implement**

```python
# src/model/event_ae.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.event_features import EventBatch


class NeighbourLayer(nn.Module):
    """h <- LayerNorm(h + GELU(W [h, mean of neighbours' h])), neighbours = ΔR < radius particles."""

    def __init__(self, width):
        super().__init__()
        self.lin = nn.Linear(2 * width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, h, adj, mask):
        adj_f = adj.to(h.dtype)
        agg = torch.bmm(adj_f, h) / adj_f.sum(-1, keepdim=True).clamp(min=1)
        h = self.norm(h + F.gelu(self.lin(torch.cat([h, agg], dim=-1))))
        return h * mask.unsqueeze(-1)


class EventAE(nn.Module):
    """Event-level graph autoencoder on dense padded tensors.

    Encoder: embed -> `layers` neighbour-averaging layers -> pool (mean / max / pT-weighted sum) -> MLP -> z.
    Decoder: MLP z -> k_recon x out_dim (the hardest particles: log pT, eta, cos phi, sin phi).
    """

    def __init__(self, in_dim=7, width=64, layers=3, hidden=128, latent_dim=16, k_recon=48, out_dim=4,
                 pooling=('mean', 'max', 'ptsum')):
        super().__init__()
        unknown = set(pooling) - {'mean', 'max', 'ptsum'}
        if unknown or not pooling:
            raise ValueError(f'bad pooling {pooling}')
        self.pooling = tuple(pooling)
        self.k_recon, self.out_dim = k_recon, out_dim
        self.embed = nn.Linear(in_dim, width)
        self.layers = nn.ModuleList(NeighbourLayer(width) for _ in range(layers))
        self.to_latent = nn.Sequential(nn.Linear(width * len(self.pooling), hidden), nn.GELU(),
                                       nn.Linear(hidden, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden),
                                     nn.GELU(), nn.Linear(hidden, k_recon * out_dim))

    def encode(self, batch: EventBatch):
        mask = batch.mask
        h = self.embed(batch.feats) * mask.unsqueeze(-1)
        for layer in self.layers:
            h = layer(h, batch.adj, mask)
        n = mask.sum(1, keepdim=True)
        parts = []
        if 'mean' in self.pooling:
            parts.append((h * mask.unsqueeze(-1)).sum(1) / n.clamp(min=1))
        if 'max' in self.pooling:
            hmax = h.masked_fill(~mask.unsqueeze(-1), -1e9).max(dim=1).values
            parts.append(torch.where(n > 0, hmax, torch.zeros_like(hmax)))
        if 'ptsum' in self.pooling:
            parts.append((h * batch.pt_frac.unsqueeze(-1)).sum(1))
        return self.to_latent(torch.cat(parts, dim=-1))

    def forward(self, batch: EventBatch):
        z = self.encode(batch)
        return self.decoder(z).view(-1, self.k_recon, self.out_dim), z
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_part2_model.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/model/event_ae.py tests/test_part2_model.py
git commit -m "Add EventAE: dense neighbour-averaging graph encoder with mean/max/pT-weighted pooling"
```

---

### Task 8: Latent-space scorers

**Files:**
- Create: `src/utils/latent_scores.py`
- Test: `tests/test_part2_latent.py`

**Interfaces:**
- Produces: `KNNScorer(k=20)`, `GMMScorer(n_components=8, seed=0)` — each with `.fit(z_train) -> self`, `.score(z) -> np.ndarray` (higher = more anomalous), attribute `n_fit`; `fit_scorers(z_train, knn_k, gmm_components, seed) -> {'knn': KNNScorer, 'gmm': GMMScorer}`. Scoring before fitting raises `RuntimeError`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_part2_latent.py
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.utils.latent_scores import GMMScorer, KNNScorer, fit_scorers


def _blobs(seed=0):
    rng = np.random.default_rng(seed)
    train = rng.normal(0, 1, (2000, 6))
    inliers = rng.normal(0, 1, (300, 6))
    outliers = rng.normal(0, 1, (300, 6)) + 5.0
    return train, np.vstack([inliers, outliers]), np.r_[np.zeros(300), np.ones(300)]


@pytest.mark.parametrize('scorer', [KNNScorer(k=10), GMMScorer(n_components=3, seed=0)])
def test_outliers_score_higher(scorer):
    train, test, y = _blobs()
    scorer.fit(train)
    assert scorer.n_fit == 2000
    assert roc_auc_score(y, scorer.score(test)) > 0.99


def test_score_before_fit_raises():
    with pytest.raises(RuntimeError):
        KNNScorer().score(np.zeros((2, 3)))
    with pytest.raises(RuntimeError):
        GMMScorer().score(np.zeros((2, 3)))


def test_fit_scorers_and_constant_dimension_is_safe():
    train, test, y = _blobs()
    train[:, 0] = 1.0; test[:, 0] = 1.0                    # a dead latent dimension must not break scoring
    scorers = fit_scorers(train, knn_k=10, gmm_components=2, seed=0)
    assert set(scorers) == {'knn', 'gmm'}
    for s in scorers.values():
        assert np.isfinite(s.score(test)).all()
        assert roc_auc_score(y, s.score(test)) > 0.99
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_part2_latent.py -q`
Expected: `ModuleNotFoundError: No module named 'src.utils.latent_scores'`

- [ ] **Step 3: Implement**

```python
# src/utils/latent_scores.py
"""Density-style anomaly scores in latent space. Always fit on training latents only."""
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors


class _Standardizer:
    def fit(self, z):
        z = np.asarray(z, dtype=np.float64)
        self.mean = z.mean(0)
        self.std = np.where(z.std(0) > 1e-8, z.std(0), 1.0)      # dead dimensions pass through unscaled
        return self

    def __call__(self, z):
        return (np.asarray(z, dtype=np.float64) - self.mean) / self.std


class KNNScorer:
    """Mean distance to the k nearest training latents."""

    def __init__(self, k=20):
        self.k, self.n_fit, self._nn = k, 0, None

    def fit(self, z_train):
        self._scale = _Standardizer().fit(z_train)
        self._nn = NearestNeighbors(n_neighbors=self.k).fit(self._scale(z_train))
        self.n_fit = len(z_train)
        return self

    def score(self, z):
        if self._nn is None:
            raise RuntimeError('KNNScorer.score called before fit')
        distances, _ = self._nn.kneighbors(self._scale(z))
        return distances.mean(axis=1)


class GMMScorer:
    """Negative log-likelihood of z under a Gaussian mixture fit to the training latents."""

    def __init__(self, n_components=8, seed=0):
        self.n_components, self.seed, self.n_fit, self._gmm = n_components, seed, 0, None

    def fit(self, z_train):
        self._scale = _Standardizer().fit(z_train)
        self._gmm = GaussianMixture(self.n_components, covariance_type='full', reg_covar=1e-4,
                                    max_iter=200, random_state=self.seed).fit(self._scale(z_train))
        self.n_fit = len(z_train)
        return self

    def score(self, z):
        if self._gmm is None:
            raise RuntimeError('GMMScorer.score called before fit')
        return -self._gmm.score_samples(self._scale(z))


def fit_scorers(z_train, knn_k=20, gmm_components=8, seed=0):
    return {'knn': KNNScorer(knn_k).fit(z_train), 'gmm': GMMScorer(gmm_components, seed).fit(z_train)}
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_part2_latent.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/utils/latent_scores.py tests/test_part2_latent.py
git commit -m "Add kNN and Gaussian-mixture latent scorers fit on training latents"
```

---

### Task 9: Trainer

**Files:**
- Create: `src/training/event_trainer.py`
- Test: `tests/test_part2_trainer.py`

**Interfaces:**
- Consumes: `make_event_batch`, `FeatureStats` (Task 2), `chamfer_per_event` (Task 6), `EventAE` (Task 7).
- Produces:
  - `TrainingError(RuntimeError)`
  - `BatchContext(stats, k, radius, wrap_phi, loss)` dataclass (`loss` ∈ `'chamfer' | 'chamfer_ptw'`)
  - `per_event_loss(model, particles, ctx) -> (loss [B], z [B, L])`
  - `infer(model, particles, ctx, batch_size=1024) -> (loss np[N], z np[N, L])` (eval mode, no grad; `particles` may be on CPU, chunks are moved to the model's device)
  - `train_event_ae(model, train_particles, val_particles, ctx, *, epochs, batch_size, lr, weight_decay, patience, grad_clip, collapse_std, seed, log=print) -> list[dict]` with keys `epoch, train_loss, val_loss, z_std_max, seconds`; the model is left holding the best-validation weights. Raises `TrainingError` on a non-finite loss or a collapsed latent (max per-dimension std of the validation latents below `collapse_std`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_part2_trainer.py
import pytest
import torch

from src.data.event_features import fit_feature_stats
from src.model.event_ae import EventAE
from src.training.event_trainer import BatchContext, TrainingError, infer, per_event_loss, train_event_ae


def _events(n=256, p=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = torch.zeros(n, p, 3)
    for i in range(n):
        k = int(torch.randint(4, p, (1,), generator=g))
        pt = torch.sort(torch.rand(k, generator=g) * 50 + 1, descending=True).values
        out[i, :k] = torch.stack([pt, torch.randn(k, generator=g), torch.rand(k, generator=g) * 6 - 3], -1)
    return out


def _ctx(p, loss='chamfer'):
    return BatchContext(stats=fit_feature_stats(p), k=8, radius=0.4, wrap_phi=True, loss=loss)


def _model():
    torch.manual_seed(0)
    return EventAE(width=16, layers=2, hidden=32, latent_dim=4, k_recon=8)


KW = dict(epochs=4, batch_size=64, lr=3e-3, weight_decay=0.0, patience=10, grad_clip=5.0,
          collapse_std=1e-6, seed=0, log=lambda *_: None)


def test_training_reduces_validation_loss_and_restores_best_weights():
    p = _events()
    train, val = p[:200], p[200:]
    ctx, model = _ctx(train), _model()
    before, _ = infer(model, val, ctx)
    history = train_event_ae(model, train, val, ctx, **KW)
    after, z = infer(model, val, ctx)
    assert len(history) >= 1 and {'epoch', 'train_loss', 'val_loss', 'z_std_max', 'seconds'} <= set(history[0])
    assert after.mean() < before.mean()
    assert abs(after.mean() - min(h['val_loss'] for h in history)) < 1e-4      # best weights restored
    assert z.shape == (56, 4)


def test_early_stopping_stops_before_max_epochs():
    p = _events()
    kw = dict(KW, epochs=50, patience=1, lr=0.0)                                # nothing learns, so epoch 2 cannot improve
    history = train_event_ae(_model(), p[:200], p[200:], _ctx(p[:200]), **kw)
    assert len(history) == 2


def test_ptweighted_loss_runs():
    p = _events(n=32)
    loss, z = per_event_loss(_model(), p, _ctx(p, 'chamfer_ptw'))
    assert loss.shape == (32,) and torch.isfinite(loss).all()


class _NaNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, batch):
        b = batch.feats.shape[0]
        return self.w * float('nan') + torch.zeros(b, 8, 4), torch.zeros(b, 4) + self.w


class _CollapsedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, batch):
        b = batch.feats.shape[0]
        return self.w + torch.zeros(b, 8, 4), torch.zeros(b, 4)


def test_nonfinite_loss_aborts():
    p = _events(n=64)
    with pytest.raises(TrainingError, match='non-finite'):
        train_event_ae(_NaNModel(), p[:48], p[48:], _ctx(p), **KW)


def test_collapsed_latent_aborts():
    p = _events(n=64)
    with pytest.raises(TrainingError, match='collapsed'):
        train_event_ae(_CollapsedModel(), p[:48], p[48:], _ctx(p), **dict(KW, collapse_std=1e-3))
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_part2_trainer.py -q`
Expected: `ModuleNotFoundError: No module named 'src.training.event_trainer'`

- [ ] **Step 3: Implement**

```python
# src/training/event_trainer.py
import copy
import time
from dataclasses import dataclass

import numpy as np
import torch

from src.data.event_features import FeatureStats, make_event_batch
from src.training.set_loss import chamfer_per_event


class TrainingError(RuntimeError):
    """Raised when training must be aborted (non-finite loss, collapsed latent)."""


@dataclass
class BatchContext:
    stats: FeatureStats
    k: int
    radius: float
    wrap_phi: bool
    loss: str = 'chamfer'          # 'chamfer' | 'chamfer_ptw'


def per_event_loss(model, particles, ctx):
    batch = make_event_batch(particles, ctx.stats, ctx.k, ctx.radius, ctx.wrap_phi)
    reconstruction, z = model(batch)
    weights = batch.target_pt if ctx.loss == 'chamfer_ptw' else None
    return chamfer_per_event(reconstruction, batch.target, batch.target_mask, weights), z


@torch.no_grad()
def infer(model, particles, ctx, batch_size=1024):
    """Per-event loss and latent for every event (eval mode). `particles` may live on the CPU."""
    model.eval()
    device = next(model.parameters()).device
    losses, latents = [], []
    for start in range(0, len(particles), batch_size):
        loss, z = per_event_loss(model, particles[start:start + batch_size].to(device), ctx)
        losses.append(loss.cpu())
        latents.append(z.cpu())
    return torch.cat(losses).numpy(), torch.cat(latents).numpy()


def train_event_ae(model, train_particles, val_particles, ctx, *, epochs, batch_size, lr, weight_decay,
                   patience, grad_clip, collapse_std, seed, log=print):
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    n = len(train_particles)
    best_val, best_state, bad_epochs, history = float('inf'), None, 0, []

    for epoch in range(1, epochs + 1):
        started = time.time()
        model.train()
        order = torch.randperm(n, generator=generator)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size].to(train_particles.device)
            loss = per_event_loss(model, train_particles[idx].to(device), ctx)[0].mean()
            if not torch.isfinite(loss):
                raise TrainingError(f'non-finite loss at epoch {epoch}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total += loss.item() * len(idx)

        val_losses, z_val = infer(model, val_particles, ctx)
        val_loss = float(val_losses.mean())
        z_std_max = float(np.std(z_val, axis=0).max())
        if not np.isfinite(val_loss):
            raise TrainingError(f'non-finite validation loss at epoch {epoch}')
        if z_std_max < collapse_std:
            raise TrainingError(f'latent collapsed at epoch {epoch}: max per-dimension std {z_std_max:.2e}')
        history.append({'epoch': epoch, 'train_loss': total / n, 'val_loss': val_loss,
                        'z_std_max': z_std_max, 'seconds': round(time.time() - started, 1)})
        log(f"epoch {epoch:02d} | train {total / n:.4f} | val {val_loss:.4f} | z_std_max {z_std_max:.3f} | "
            f"{history[-1]['seconds']}s")

        if val_loss < best_val - 1e-6:
            best_val, bad_epochs = val_loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_part2_trainer.py -q`
Expected: `5 passed`

If `test_training_reduces_validation_loss_and_restores_best_weights` is flaky on the random data, raise `epochs` to 8 in `KW` for that test only; do not loosen the assertions.

- [ ] **Step 5: Commit**

```bash
git add src/training/event_trainer.py tests/test_part2_trainer.py
git commit -m "Add Chamfer trainer with validation early stopping and NaN/collapse guards"
```

---

### Task 10: Training run pipeline (`train` subcommand)

**Files:**
- Modify: `src/pipeline/part2.py` (add `run_training`), `tests/test_part2_pipeline.py` (append tests)

**Interfaces:**
- Consumes: Tasks 2, 3, 7, 8, 9 and the Task 4 pipeline.
- Produces: `run_training(cfg, seed, contamination=None, tag='main') -> dict` (metrics). Files under `<output_dir>/runs/<tag>/`: `scores_s<seed>.npz` (keys `chamfer, knn, gmm, labels`), `metrics_s<seed>.json`, `model_s<seed>.pt`; on `TrainingError` a `failed_s<seed>.json` is written and the error re-raised. If `scores_s<seed>.npz` exists the run is skipped and its metrics returned. `contamination=None` uses `cfg['split']['contamination']`.
- Training set = first `n_train − n_val` pure background events (`n_val = round(val_fraction × n_train)`) with `round(contamination × n_fit)` of them replaced by signal from `contam_sig`; validation = the last `n_val` background events (always pure).

- [ ] **Step 1: Append the failing tests to `tests/test_part2_pipeline.py`**

```python
from src.pipeline.part2 import run_training


def test_run_training_writes_scores_and_skips_existing(tiny_cfg):
    prepare_cache(tiny_cfg)
    metrics = run_training(tiny_cfg, seed=0, tag='main')
    run_dir = os.path.join(tiny_cfg['output_dir'], 'runs', 'main')
    scores = np.load(os.path.join(run_dir, 'scores_s0.npz'))
    assert set(scores.files) == {'chamfer', 'knn', 'gmm', 'labels'}
    assert all(scores[k].shape == (60,) and np.isfinite(scores[k]).all() for k in ('chamfer', 'knn', 'gmm'))
    assert scores['labels'].tolist() == [0] * 30 + [1] * 30
    assert set(metrics['auc']) == {'chamfer', 'knn', 'gmm'} and metrics['n_contaminating_signal'] == 0
    assert metrics['n_fit'] == 90 and metrics['num_parameters'] > 0
    mtime = os.path.getmtime(os.path.join(run_dir, 'scores_s0.npz'))
    assert run_training(tiny_cfg, seed=0, tag='main')['auc'] == metrics['auc']
    assert os.path.getmtime(os.path.join(run_dir, 'scores_s0.npz')) == mtime


def test_contamination_run_replaces_training_events_with_signal(tiny_cfg):
    prepare_cache(tiny_cfg)
    metrics = run_training(tiny_cfg, seed=1, contamination=0.1, tag='contam')
    assert metrics['n_contaminating_signal'] == 9 and metrics['n_fit'] == 90


def test_contamination_larger_than_cache_raises(tiny_cfg):
    prepare_cache(tiny_cfg)
    with pytest.raises(ValueError, match='contamination'):
        run_training(tiny_cfg, seed=0, contamination=0.5, tag='too_much')


def test_full_report_with_model_rows_and_gate(tiny_cfg):
    prepare_cache(tiny_cfg)
    for seed in (0, 1):
        run_training(tiny_cfg, seed=seed, tag='main')
    bank = build_bank(tiny_cfg)
    out = make_report(tiny_cfg, bank, run_tags=['main'], name='final')
    names = {r['name'] for r in out['rows']}
    assert {'EventAE chamfer', 'EventAE knn', 'EventAE gmm'} <= names
    assert out['gate'] is not None and out['gate']['bar_name'] in {
        'HT (sum pT)', 'mean pT per particle', 'leading pT', '-multiplicity', 'm_all'}
    assert next(r for r in out['rows'] if r['name'] == 'EventAE knn')['notes'] == '2 seed(s)'
```

(The synthetic signal has half the multiplicity of background, so the model's scores carry signal, but this test does not assert on AUC values or on whether the gate passes.)

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_part2_pipeline.py -q`
Expected: `ImportError: cannot import name 'run_training'`

- [ ] **Step 3: Implement `run_training` (append to `src/pipeline/part2.py`)**

Add these imports at the top of the file:

```python
import random
import time

import torch
from sklearn.metrics import roc_auc_score

from src.data.event_features import fit_feature_stats
from src.model.event_ae import EventAE
from src.training.event_trainer import BatchContext, TrainingError, infer, train_event_ae
from src.utils.latent_scores import fit_scorers
```

Then append:

```python
def run_training(cfg, seed, contamination=None, tag='main'):
    """Trains one EventAE, computes the three scores on the frozen test set, writes the run files."""
    run_dir = os.path.join(cfg['output_dir'], 'runs', tag)
    os.makedirs(run_dir, exist_ok=True)
    scores_path = os.path.join(run_dir, f'scores_s{seed}.npz')
    metrics_path = os.path.join(run_dir, f'metrics_s{seed}.json')
    if os.path.exists(scores_path):
        print(f'run {tag}/s{seed} already done; skipping', flush=True)
        return json.load(open(metrics_path, encoding='utf-8'))

    s, d, m, t = cfg['split'], cfg['data'], cfg['model'], cfg['training']
    contamination = s['contamination'] if contamination is None else contamination
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cache = d['cache_dir']
    train_bg, contam = load_role(cache, 'train_bg'), load_role(cache, 'contam_sig')
    test_bg, test_sig = load_role(cache, 'test_bg'), load_role(cache, 'test_sig')
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
    assert all(sc.n_fit == len(fit_np) for sc in scorers.values()), 'density scorers must be fit on training latents only'
    scores = {'chamfer': chamfer_test, 'knn': scorers['knn'].score(z_test), 'gmm': scorers['gmm'].score(z_test)}

    np.savez(scores_path, labels=y, **scores)
    torch.save({'model_state_dict': model.state_dict(), 'config': cfg}, os.path.join(run_dir, f'model_s{seed}.pt'))
    metrics = {
        'tag': tag, 'seed': seed, 'contamination': contamination, 'n_contaminating_signal': n_sig,
        'n_fit': int(len(fit_np)), 'n_val': int(len(val_np)),
        'auc': {k: float(roc_auc_score(y, v)) for k, v in scores.items()},
        'epochs_run': len(history), 'final_val_loss': history[-1]['val_loss'],
        'best_val_loss': min(h['val_loss'] for h in history),
        'seconds_per_epoch': float(np.mean([h['seconds'] for h in history])),
        'seconds_total': round(time.time() - started, 1),
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'history': history,
    }
    with open(metrics_path, 'w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2)
    print(f"run {tag}/s{seed}: AUC {metrics['auc']}", flush=True)
    return metrics
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_part2_pipeline.py -q`
Expected: `8 passed`

Then the full suite: `.venv/bin/python -m pytest tests -q` → all green.

- [ ] **Step 5: CLI smoke test on the synthetic file (end to end through the script)**

```bash
.venv/bin/python - <<'EOF'
import h5py, numpy as np
from tests.conftest import make_events
with h5py.File('/tmp/claude_p2_events.h5', 'w') as f:
    f.create_group('df').create_dataset('block0_values', data=make_events(200, 60))
EOF
O="--set data.h5_path=/tmp/claude_p2_events.h5 --set data.features_path=null --set data.cache_dir=/tmp/claude_p2/cache --set output_dir=/tmp/claude_p2/out --set data.max_particles=40 --set split.skip_bg=10 --set split.n_test_bg=30 --set split.n_test_sig=30 --set split.n_train=100 --set split.n_contam_sig=10 --set model.width=16 --set model.layers=2 --set model.latent_dim=4 --set model.k_recon=8 --set model.hidden=32 --set training.epochs=2 --set training.batch_size=32 --set scores.knn_k=5 --set scores.gmm_components=2 --set benchmark.n_boot=30"
for c in prepare "train --seed 0" "bank" "report --tags main --name smoke"; do .venv/bin/python scripts/part2_run.py $c --config configs/part2_main.yaml $O || break; done
```

Expected: the last command prints a Markdown table containing `EventAE chamfer`, `EventAE knn`, `EventAE gmm` and a `## Gate` section. (The stage0 reference table will show `missing`/NO for Part 1 rows in this smoke run; that is expected because no Part 1 scores were supplied.) Clean up: `rm -rf /tmp/claude_p2 /tmp/claude_p2_events.h5`.

- [ ] **Step 6: Commit**

```bash
git add src/pipeline/part2.py tests/test_part2_pipeline.py
git commit -m "Add Part 2 training run: split, train, latent scorers, scores and metrics per seed"
```

---

### Task 11: Sweep runner and Kaggle stage bodies

**Files:**
- Create: `scripts/part2_sweep.py`, `scripts/kaggle/part2_smoke.py`, `scripts/kaggle/part2_stage1.py`, `scripts/kaggle/part2_stage2.py`, `scripts/kaggle/part2_stage3.py`
- Test: `tests/test_part2_sweep.py`

**Interfaces:**
- Produces in `scripts/part2_sweep.py`: `GRIDS` (dict stage → list of `(tag, overrides, seeds)`), `build_jobs(stage) -> list[(tag, seed, overrides)]`; CLI `python scripts/part2_sweep.py --stage {main,ablations,contamination} --config CFG [--set k=v ...] [--jobs-per-gpu 1]`.
- Stages (spec Section 5): `main` = tag `main`, seeds 0,1,2; `ablations` = latent 4/8/32, K 24/96, depth 2/4, no pT-weighted pooling, pT-weighted loss (9 tags × seeds 0,1); `contamination` = 9% and 1% (seed 0).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_part2_sweep.py
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))

from part2_sweep import GRIDS, build_jobs  # noqa: E402


def test_main_stage_is_three_seeds_with_default_config():
    assert build_jobs('main') == [('main', 0, []), ('main', 1, []), ('main', 2, [])]


def test_ablation_tags_cover_the_spec_and_use_two_seeds():
    jobs = build_jobs('ablations')
    tags = {t for t, _, _ in jobs}
    assert tags == {'z4', 'z8', 'z32', 'k24', 'k96', 'depth2', 'depth4', 'nopt_pool', 'ptw_loss'}
    assert len(jobs) == 18
    assert dict((t, o) for t, _, o in jobs)['nopt_pool'] == ['model.pooling=[mean,max]']


def test_contamination_stage():
    jobs = build_jobs('contamination')
    assert sorted((t, o[0]) for t, _, o in jobs) == [('contam1', 'split.contamination=0.01'),
                                                     ('contam9', 'split.contamination=0.09')]
    assert set(GRIDS) == {'main', 'ablations', 'contamination'}
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_part2_sweep.py -q`
Expected: `ModuleNotFoundError: No module named 'part2_sweep'`

- [ ] **Step 3: Implement the sweep runner**

```python
# scripts/part2_sweep.py
"""Runs a Part 2 stage: one `part2_run.py train` subprocess per (tag, seed), one job per GPU slot.

    python scripts/part2_sweep.py --stage main --config configs/part2_main.yaml --set data.h5_path=... --set ...

Finished runs are skipped (scores file exists), so a re-launch after a timeout continues where it stopped.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GRIDS = {
    'main': [('main', [], (0, 1, 2))],
    'ablations': [
        ('z4', ['model.latent_dim=4'], (0, 1)),
        ('z8', ['model.latent_dim=8'], (0, 1)),
        ('z32', ['model.latent_dim=32'], (0, 1)),
        ('k24', ['model.k_recon=24'], (0, 1)),
        ('k96', ['model.k_recon=96'], (0, 1)),
        ('depth2', ['model.layers=2'], (0, 1)),
        ('depth4', ['model.layers=4'], (0, 1)),
        ('nopt_pool', ['model.pooling=[mean,max]'], (0, 1)),
        ('ptw_loss', ['training.loss=chamfer_ptw'], (0, 1)),
    ],
    'contamination': [
        ('contam9', ['split.contamination=0.09'], (0,)),
        ('contam1', ['split.contamination=0.01'], (0,)),
    ],
}


def build_jobs(stage):
    return [(tag, seed, list(overrides)) for tag, overrides, seeds in GRIDS[stage] for seed in seeds]


def gpu_count():
    try:
        import torch
        return torch.cuda.device_count()
    except ImportError:
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=sorted(GRIDS))
    parser.add_argument('--config', default='configs/part2_main.yaml')
    parser.add_argument('--set', action='append', default=[], dest='overrides')
    parser.add_argument('--jobs-per-gpu', type=int, default=1)
    args = parser.parse_args()

    output_dir = yaml.safe_load(open(os.path.join(REPO, args.config)))['output_dir']
    for item in args.overrides:
        if item.startswith('output_dir='):
            output_dir = yaml.safe_load(item.partition('=')[2])

    jobs = [j for j in build_jobs(args.stage)
            if not os.path.exists(os.path.join(output_dir, 'runs', j[0], f'scores_s{j[1]}.npz'))]
    n_gpu = gpu_count()
    slots = [str(i % n_gpu) if n_gpu else None for i in range(max(n_gpu, 1) * args.jobs_per_gpu)]
    print(f'{len(jobs)} runs pending on {n_gpu} GPU(s), {len(slots)} parallel slot(s)', flush=True)

    t0 = time.time()
    running, results, pending = {}, [], list(jobs)
    while pending or running:
        for slot, gpu in enumerate(slots):
            if slot in running or not pending:
                continue
            tag, seed, extra = pending.pop(0)
            cmd = [sys.executable, os.path.join(REPO, 'scripts/part2_run.py'), 'train', '--config', args.config,
                   '--tag', tag, '--seed', str(seed)]
            for item in args.overrides + extra:
                cmd += ['--set', item]
            env = dict(os.environ, TQDM_DISABLE='1')
            if gpu is not None:
                env['CUDA_VISIBLE_DEVICES'] = gpu
            os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
            log = open(os.path.join(output_dir, 'logs', f'{tag}_s{seed}.log'), 'w')
            running[slot] = (subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT),
                             tag, seed, time.time())
        time.sleep(2)
        for slot, (proc, tag, seed, started) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[slot]
            results.append({'tag': tag, 'seed': seed, 'returncode': proc.returncode,
                            'seconds': round(time.time() - started, 1)})
            print(f'[{time.time() - t0:.0f}s] {tag}/s{seed} rc={proc.returncode}', flush=True)

    # Summary of every finished run of this stage (including runs from earlier launches)
    rows = []
    for tag, _, seeds in GRIDS[args.stage]:
        for seed in seeds:
            path = os.path.join(output_dir, 'runs', tag, f'metrics_s{seed}.json')
            if os.path.exists(path):
                m = json.load(open(path))
                rows.append({'tag': tag, 'seed': seed, **{f'auc_{k}': round(v, 4) for k, v in m['auc'].items()},
                             'epochs': m['epochs_run'], 'sec_per_epoch': round(m['seconds_per_epoch'], 1),
                             'params': m['num_parameters']})
    if rows:
        with open(os.path.join(output_dir, f'sweep_{args.stage}.csv'), 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    failed = [r for r in results if r['returncode'] != 0]
    print(f"done; failed: {[(r['tag'], r['seed']) for r in failed] or 'none'}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Write the Kaggle stage bodies**

Each body repeats the same three override lines so a kernel is self-contained.

```python
# scripts/kaggle/part2_smoke.py
# Timing smoke test: a 20k-event run to measure seconds/epoch on the T4 before committing to Stage 1.
OVERRIDES = [f"data.h5_path={H5}", f"data.features_path={FEATURES}", f"data.cache_dir={CACHE}",
             f"output_dir={OUT}", "split.n_train=20000", "split.n_contam_sig=1800", "training.epochs=3"]
flags = [x for o in OVERRIDES for x in ("--set", o)]
sh(sys.executable, "scripts/part2_run.py", "prepare", "--config", "configs/part2_main.yaml", *flags)
sh(sys.executable, "scripts/part2_run.py", "train", "--config", "configs/part2_main.yaml", "--tag", "smoke",
   "--seed", "0", *flags)
print(open(f"{OUT}/runs/smoke/metrics_s0.json").read()[:600])
```

```python
# scripts/kaggle/part2_stage1.py
# Stage 1: headline configuration, 3 seeds (2 GPUs in parallel), then the full benchmark report.
OVERRIDES = [f"data.h5_path={H5}", f"data.features_path={FEATURES}", f"data.cache_dir={CACHE}", f"output_dir={OUT}"]
flags = [x for o in OVERRIDES for x in ("--set", o)]
sh(sys.executable, "scripts/part2_run.py", "prepare", "--config", "configs/part2_main.yaml", *flags)
sh(sys.executable, "scripts/part2_sweep.py", "--stage", "main", "--config", "configs/part2_main.yaml", *flags)
# bank + report need the Part 1 scores from Stage 0; the final report is rebuilt locally (Task 12)
```

```python
# scripts/kaggle/part2_stage2.py
# Stage 2: ablations (sensitivity analysis only).
OVERRIDES = [f"data.h5_path={H5}", f"data.features_path={FEATURES}", f"data.cache_dir={CACHE}", f"output_dir={OUT}"]
flags = [x for o in OVERRIDES for x in ("--set", o)]
sh(sys.executable, "scripts/part2_run.py", "prepare", "--config", "configs/part2_main.yaml", *flags)
sh(sys.executable, "scripts/part2_sweep.py", "--stage", "ablations", "--config", "configs/part2_main.yaml", *flags)
```

```python
# scripts/kaggle/part2_stage3.py
# Stage 3: contamination runs (9% and 1% signal in the training set).
OVERRIDES = [f"data.h5_path={H5}", f"data.features_path={FEATURES}", f"data.cache_dir={CACHE}", f"output_dir={OUT}"]
flags = [x for o in OVERRIDES for x in ("--set", o)]
sh(sys.executable, "scripts/part2_run.py", "prepare", "--config", "configs/part2_main.yaml", *flags)
sh(sys.executable, "scripts/part2_sweep.py", "--stage", "contamination", "--config", "configs/part2_main.yaml", *flags)
```

Each Kaggle body drops model weights from the output to keep it small — add this as the last lines of the Stage 1, 2 and 3 bodies (the smoke body can skip it):

```python
for path in __import__("glob").glob(f"{OUT}/runs/*/model_s*.pt"):
    os.remove(path)
```

- [ ] **Step 5: Run to verify the test passes, and the full suite**

Run: `.venv/bin/python -m pytest tests -q`
Expected: all tests pass (Part 1's 20 plus all Part 2 tests).

- [ ] **Step 6: Commit**

```bash
git add scripts/part2_sweep.py scripts/kaggle tests/test_part2_sweep.py
git commit -m "Add Part 2 stage sweep runner and Kaggle stage bodies"
```

---

### Task 12: Kaggle runs: smoke test, Stage 1 (gate), Stages 2 and 3

**Files:** none new (run artifacts go to the gitignored `results/part2/`).

Order matters: Stage 0 must have passed (Task 5). The smoke run comes first because the epoch time for 180k events with the dense adjacency is unmeasured.

- [ ] **Step 1: Smoke run for timing (20k events, 3 epochs, one GPU)**

```bash
S=/tmp/claude-1000/-home-jamil-Projects-Research-Project-Hidden-Vertex/b7dfad6c-7e70-40b6-a8cf-55eb13e8970a/scratchpad
.venv/bin/python scripts/kaggle_bundle.py --slug hidden-vertex-part2-smoke --title "Hidden Vertex Part2 Smoke" \
  --body scripts/kaggle/part2_smoke.py --out $S/k_smoke
kaggle kernels push -p $S/k_smoke -t 3600
kaggle kernels status viveriun/hidden-vertex-part2-smoke          # poll; expect ~15 min (label scan + prep dominate)
kaggle kernels output viveriun/hidden-vertex-part2-smoke -p results/part2/kaggle/smoke -o -q
```

Read `seconds_per_epoch` in `results/part2/kaggle/smoke/runs/smoke/metrics_s0.json`. Epoch time for the full run ≈ 9 × that (180k / 20k events). **Decision rule:** if projected epoch time × 30 epochs > 1.5 h per seed, report the number to the user and propose reducing `training.epochs` or `split.n_train` (changing the headline configuration is allowed *here*, before any test-set result exists; record the change in the commit message). Otherwise continue unchanged. Also check in the smoke output that no `failed_s0.json` exists and the AUCs are finite.

- [ ] **Step 2: Stage 1 (3 seeds)**

```bash
.venv/bin/python scripts/kaggle_bundle.py --slug hidden-vertex-part2-stage1 --title "Hidden Vertex Part2 Stage1" \
  --body scripts/kaggle/part2_stage1.py --out $S/k_stage1
kaggle kernels push -p $S/k_stage1 -t 18000
kaggle kernels status viveriun/hidden-vertex-part2-stage1
kaggle kernels output viveriun/hidden-vertex-part2-stage1 -p results/part2/kaggle/stage1 -o -q
```

Seeds 0 and 1 run in parallel (one per T4), then seed 2. If the kernel times out, re-pushing continues from the finished runs only if their outputs persist, which Kaggle does not guarantee between kernel versions; so size the timeout from the smoke measurement (3 seeds ≈ 2 × single-seed time) with 50% margin.

- [ ] **Step 3: Build the Stage 1 report locally (the gate)**

```bash
mkdir -p results/part2/runs && cp -r results/part2/kaggle/stage1/runs/main results/part2/runs/
.venv/bin/python scripts/part2_run.py report --tags main --name stage1 --config configs/part2_main.yaml
```

(This uses `results/part2/bank.npz|json` copied from Stage 0 in Task 5, Step 6.) Read the printed table and the `## Gate` section.

**Report to the user before running Stages 2/3:** the full table, the gate result (PASSED / NOT PASSED with the bar and the CI lower bound), and for each of the three scores its within-bin minimum. If the gate is NOT PASSED, the spec's fallback applies: the deliverable is the diagnosis, so also run Task 13's diagnosis step. Do not alter the gate rule or the headline configuration after seeing this result.

- [ ] **Step 4: Stage 2 (ablations) and Stage 3 (contamination)**

Both can be submitted now; they are independent (each re-runs the 5–10 min data prep). Same pattern as Step 2 with `--slug hidden-vertex-part2-stage2 --body scripts/kaggle/part2_stage2.py` (timeout `-t 21600`) and `--slug hidden-vertex-part2-stage3 --body scripts/kaggle/part2_stage3.py` (timeout `-t 10800`); outputs to `results/part2/kaggle/stage2` and `.../stage3`.

Then:

```bash
for st in stage2 stage3; do cp -r results/part2/kaggle/$st/runs/* results/part2/runs/; done
.venv/bin/python scripts/part2_run.py report \
  --tags main,z4,z8,z32,k24,k96,depth2,depth4,nopt_pool,ptw_loss,contam9,contam1 --name final \
  --config configs/part2_main.yaml
```

Expected: `results/part2/final.md` lists the headline rows (group `model`) and every ablation or contamination variant (group `variant`). Variants never enter the gate.

No commit (result artifacts are gitignored). Keep `results/part2/final.md|json` for Task 13.

---

### Task 13: Results write-up, README link, memory

**Files:**
- Create: `docs/PART2_RESULTS.md`
- Modify: `README.md` (add a short "Part 2" pointer section), memory file `part1-verification-status.md` (stale), new memory `part2-status.md`

- [ ] **Step 1: Write `docs/PART2_RESULTS.md` from the real numbers**

Required sections, filled from `results/part2/stage0.md`, `stage1.md`, `final.md` and the sweep CSVs (no placeholders; every number copied from those files):

1. **Setup**: test set, train set, config path, Kaggle kernels and versions used, dates.
2. **Stage 0**: the baseline table and the reference check; answer the open question "is the Part 1 mean score just mean pT / HT per multiplicity?" with the measured `mean pT per particle` AUC and the Part 1 mean AUC next to it.
3. **Stage 1 headline**: the three model scores with seed-averaged AUC, 98.3% CIs, within-bin minima, the bar (best particle-level baseline), and the verbatim gate outcome.
4. **Diagnosis** (required if the gate failed, useful otherwise): for the best model score on seed 0, Spearman correlation with HT, multiplicity and leading pT, and the within-bin AUC ranges. Compute with a short script (`scripts/part2_diagnostics.py` is *not* needed; do it inline from `runs/main/scores_s0.npz` and `bank.npz`, and paste the numbers). State what the score tracks.
5. **Ablations** (sensitivity only; list all 9 with AUCs, no winner picking) and **contamination** (AUC drop vs the pure run, 9% and 1%).
6. **Conclusion and next step**: pass → what to try beyond; fail → the recommendation from the spec's fallback (for example jet-substructure inputs).
7. **Deviations from the spec**: any change made after the smoke run, if any.

- [ ] **Step 2: Add a "Part 2" pointer to `README.md`**

A short section (5–10 lines) linking `docs/PART2_RESULTS.md`, stating the gate outcome in one sentence, and the command sequence `prepare` → `bank` → `train`/sweep → `report`.

- [ ] **Step 3: Update memory**

- Edit `/home/jamil/.claude/projects/-home-jamil-Projects-Research-Project-Hidden-Vertex/memory/part1-verification-status.md`: the description says "changes uncommitted" — update to "committed in 6647d4d and fd1b126 on `part1-verification`", and drop the dangling `[[user-kaggle-workflow]]` link or write that memory.
- Create `part2-status.md` (type `project`): branch `part2-event-level-ae`, spec and plan paths, gate outcome, Kaggle kernel slugs, where results live (`results/part2/`), what is next. Add a one-line pointer to `MEMORY.md`.

- [ ] **Step 4: Final verification and commit**

Run: `.venv/bin/python -m pytest tests -q` → all pass.
Run: `git status --short` → only intended files.

```bash
git add docs/PART2_RESULTS.md README.md
git commit -m "Add Part 2 results write-up and README pointer"
git log -3 --format=%B | grep -ci co-authored     # must print 0
```

Ask the user whether to merge `part1-verification` and `part2-event-level-ae` into `main`; do not merge without approval.

---

## Self-Review (completed)

**Spec coverage**

| Spec item | Task |
|---|---|
| Success criterion, Bonferroni 98.3%, particle-level bar, within-bin ≥ 0.6, fallback diagnosis | 1 (`evaluate_gate`), 4 (`model_ci_level`), 12 (Step 3), 13 (diagnosis) |
| `baselines.py` (HT, mean pT, leading pT, multiplicity, m_all, jet scores, oracle) | 3 |
| `report.py` | 1 |
| `event_features.py`, dense padded tensors, GPU adjacency, φ wrap | 2 |
| `oracle_split` extension (frozen test set, disjoint train, contamination, larger sizes, particle cache) | 3 (`event_store.plan_rows/build_cache`; new module instead of editing the PyG-based `oracle_split.py`, which Part 1 still uses) |
| `EventAE` (3 layers, width 64, residual, mean/max/pT-sum pooling, z=16, decoder K=48) | 7 |
| Chamfer with masking, pT-weighted variant | 6 |
| Latent kNN and GMM scores fit on training latents only (asserted) | 8, 10 (`assert ... n_fit`) |
| Entry point + config + contamination run | 4, 10, 11 |
| Stage 0 reproduction gate (HT 0.792, mjj 0.788, Part 1 max/mean) | 4 (`check_reference`, `--check-reference`), 5 (Kaggle run) |
| Stage 1 (3 seeds), Stage 2 ablations (latent 4/8/16/32, K 24/48/96, depth, pooling, loss; 2 seeds), Stage 3 (9% and 1%) | 11 (`GRIDS`), 12 |
| Early stopping on pure background validation only; honesty rules | 10 (`val` = last 10% of pure bg), Global Constraints |
| Compute plan, `-t` timeouts, skip-if-exists, smoke timing | 11, 12 |
| Error handling (quota, single class, overlap, NaN, collapse, truncation) | 3 (`plan_rows`, truncation guard, label check), 9 (NaN/collapse), 4 (`labels` check in `build_bank`) |
| Testing list (features, adjacency parity, Chamfer, encoder invariance, scorers, report vs sklearn, disjointness, Stage 0 end to end) | 1, 2, 3, 6, 7, 8, 4–5 |
| Jet-feature AE reference row ("added after Stage 1") | **Not planned.** The spec calls it a reference row only; it needs a separate model and is out of the gate. Add it as a follow-up task if the user wants it after seeing Stage 1. |

**Placeholder scan:** no TBD/TODO; the one deliberate instruction to edit a test line (Task 4: replace `assert ht['auc'] < 0.5`) is explicit.

**Type consistency checked across tasks:** `EventBatch` fields (`feats, mask, adj, pt_frac, target, target_mask, target_pt`) are the same in Tasks 2, 7, 9; `FeatureStats.to()` used in Task 10; `benchmark_row` signature `(name, y, scores, covariates, group, level, n_boot, n_bins, notes)` matches its call in Task 4; row groups `particle|jet|oracle|external|model|variant` consistent between `baselines.py`, `part2.py` and `evaluate_gate`; `run_training` output keys (`auc`, `n_fit`, `n_contaminating_signal`, `seconds_per_epoch`, `num_parameters`, `epochs_run`) match the sweep summary and the tests; baseline names match `stage0_expected` keys (`HT (sum pT)`, `mjj`, `Part 1 AE (max)`, `Part 1 AE (mean)`).
