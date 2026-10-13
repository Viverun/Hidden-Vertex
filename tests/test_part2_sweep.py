# tests/test_part2_sweep.py
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))

import json  # noqa: E402

from part2_sweep import GRIDS, build_jobs, summarize_stage  # noqa: E402


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


def _write_run(output_dir, tag, seed, metrics=True, scores=True, corrupt=False):
    run_dir = os.path.join(output_dir, 'runs', tag)
    os.makedirs(run_dir, exist_ok=True)
    if metrics:
        with open(os.path.join(run_dir, f'metrics_s{seed}.json'), 'w', encoding='utf-8') as handle:
            if corrupt:
                handle.write('{"auc": {"chamfer": 0.5')
            else:
                json.dump({'auc': {'chamfer': 0.61234, 'knn': 0.7, 'gmm': 0.8}, 'epochs_run': 5,
                           'seconds_per_epoch': 1.234, 'num_parameters': 99}, handle)
    if scores:
        with open(os.path.join(run_dir, f'scores_s{seed}.npz'), 'wb') as handle:
            handle.write(b'x')


def test_summarize_stage_includes_finished_run_with_unchanged_columns(tmp_path):
    _write_run(str(tmp_path), 'main', 0)
    rows = summarize_stage(str(tmp_path), 'main')
    assert rows == [{'tag': 'main', 'seed': 0, 'auc_chamfer': 0.6123, 'auc_knn': 0.7, 'auc_gmm': 0.8,
                     'epochs': 5, 'sec_per_epoch': 1.2, 'params': 99}]


def test_summarize_stage_excludes_metrics_without_scores(tmp_path):
    _write_run(str(tmp_path), 'main', 0)
    _write_run(str(tmp_path), 'main', 1, scores=False)
    assert [r['seed'] for r in summarize_stage(str(tmp_path), 'main')] == [0]


def test_summarize_stage_skips_corrupt_metrics_with_warning(tmp_path, capsys):
    _write_run(str(tmp_path), 'main', 0)
    _write_run(str(tmp_path), 'main', 1, corrupt=True)
    assert [r['seed'] for r in summarize_stage(str(tmp_path), 'main')] == [0]
    assert 'WARNING' in capsys.readouterr().out
