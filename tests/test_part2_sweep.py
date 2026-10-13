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
