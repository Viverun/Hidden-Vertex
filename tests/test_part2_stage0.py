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
