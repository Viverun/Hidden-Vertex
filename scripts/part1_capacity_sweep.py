"""
Capacity sweep for the Part 1 node-level autoencoder.

Builds the oracle split once (via `main.py prepare-part1`), then trains/evaluates every
(hidden_layers, latent_dim, loss_mode, seed) combination with `scripts/run_part1_eval.py`,
running `--jobs-per-gpu` processes per visible GPU in parallel. Results are collected in
`<out>/sweep_results.json` and `<out>/sweep_results.csv`.

    python scripts/part1_capacity_sweep.py --h5 dataset/raw/events_anomalydetection_v2.h5
"""
import argparse
import copy
import csv
import itertools
import json
import os
import subprocess
import sys
import time

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description='Part 1 capacity sweep')
    parser.add_argument('--config', default=os.path.join(REPO, 'configs/part1_notebook_parity.yaml'))
    parser.add_argument('--h5', required=True)
    parser.add_argument('--out', default='results/sweep')
    parser.add_argument('--widths', default='16x8,32x16,64x32,128x64')
    parser.add_argument('--latents', default='1,2,4,8')
    parser.add_argument('--loss-modes', default='pt_only,weighted')
    parser.add_argument('--seeds', default='0,1')
    parser.add_argument('--epochs', type=int, default=None, help='Override training.epochs')
    parser.add_argument('--oracle', default=None, help='train,test_bg,test_sig override (smoke tests)')
    parser.add_argument('--jobs-per-gpu', type=int, default=2)
    return parser.parse_args()


def gpu_count():
    try:
        import torch
        return torch.cuda.device_count()
    except ImportError:
        return 0


def main():
    args = parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(os.path.join(out, 'configs'), exist_ok=True)
    os.makedirs(os.path.join(out, 'logs'), exist_ok=True)

    base = yaml.safe_load(open(args.config))
    base['data']['h5_path'] = args.h5
    base['data']['split_path'] = os.path.join(out, 'split.pt')
    if args.epochs is not None:
        base['training']['epochs'] = args.epochs
    if args.oracle:
        tr, bg, sig = (int(v) for v in args.oracle.split(','))
        base['oracle'] = {'train_target': tr, 'test_bg_target': bg, 'test_sig_target': sig}

    t0 = time.time()
    split_cfg = os.path.join(out, 'configs', 'split.yaml')
    yaml.safe_dump(base, open(split_cfg, 'w'))
    if not os.path.exists(base['data']['split_path']):
        subprocess.run([sys.executable, os.path.join(REPO, 'main.py'), 'prepare-part1', '--config', split_cfg],
                       cwd=REPO, check=True, env=dict(os.environ, TQDM_DISABLE='1'))
    print(f'[{time.time() - t0:.0f}s] split ready', flush=True)

    widths = [tuple(int(v) for v in w.split('x')) for w in args.widths.split(',')]
    grid = list(itertools.product(
        widths,
        [int(v) for v in args.latents.split(',')],
        args.loss_modes.split(','),
        [int(v) for v in args.seeds.split(',')],
    ))

    jobs = []
    for hidden, latent, loss_mode, seed in grid:
        tag = f"h{hidden[0]}-{hidden[1]}_z{latent}_{loss_mode}_s{seed}"
        cfg = copy.deepcopy(base)
        cfg['model']['hidden_layers'] = list(hidden)
        cfg['model']['latent_dim'] = latent
        cfg['training']['loss_mode'] = loss_mode
        cfg['training']['seed'] = seed
        cfg['evaluation'].update(
            metrics_path=os.path.join(out, 'metrics', f'{tag}.json'),
            scores_path=None,
            model_path=os.path.join(out, 'models', f'{tag}.pt'),
            roc_path=os.path.join(out, 'figures', f'{tag}.png'),
        )
        path = os.path.join(out, 'configs', f'{tag}.yaml')
        yaml.safe_dump(cfg, open(path, 'w'))
        jobs.append((tag, path, cfg['evaluation']['metrics_path']))

    n_gpu = gpu_count()
    slots = [str(i % n_gpu) if n_gpu else None for i in range(max(n_gpu, 1) * args.jobs_per_gpu)]
    pending, running, results = list(jobs), {}, []
    print(f'{len(jobs)} runs on {n_gpu} GPU(s), {len(slots)} parallel slots', flush=True)

    while pending or running:
        for slot, gpu in enumerate(slots):
            if slot in running or not pending:
                continue
            tag, path, metrics_path = pending.pop(0)
            env = dict(os.environ, TQDM_DISABLE='1')
            if gpu is not None:
                env['CUDA_VISIBLE_DEVICES'] = gpu
            log = open(os.path.join(out, 'logs', f'{tag}.log'), 'w')
            proc = subprocess.Popen([sys.executable, os.path.join(REPO, 'scripts/run_part1_eval.py'), '--config', path],
                                    cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            running[slot] = (proc, tag, metrics_path, time.time())

        time.sleep(2)
        for slot, (proc, tag, metrics_path, started) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[slot]
            row = {'tag': tag, 'returncode': proc.returncode, 'seconds': round(time.time() - started, 1)}
            if proc.returncode == 0 and os.path.exists(metrics_path):
                row.update(json.load(open(metrics_path)))
            results.append(row)
            print(f"[{time.time() - t0:.0f}s] {len(results)}/{len(jobs)} {tag} "
                  f"auc={row.get('auc', float('nan')):.4f} loss={row.get('final_train_loss', float('nan')):.4f} "
                  f"rc={proc.returncode}", flush=True)
            json.dump(results, open(os.path.join(out, 'sweep_results.json'), 'w'), indent=2)

    fields = ['tag', 'hidden_layers', 'latent_dim', 'loss_mode', 'seed', 'num_parameters',
              'final_train_loss', 'auc', 'auc_max', 'auc_mean', 'seconds', 'returncode']
    with open(os.path.join(out, 'sweep_results.csv'), 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in sorted(results, key=lambda r: r['tag']):
            by_red = row.get('auc_by_reduction', {})
            writer.writerow({**row, 'auc_max': by_red.get('max'), 'auc_mean': by_red.get('mean'),
                             'hidden_layers': 'x'.join(map(str, row.get('hidden_layers', [])))})
    failed = [r['tag'] for r in results if r['returncode'] != 0]
    print(f'[{time.time() - t0:.0f}s] done; failed: {failed or "none"}', flush=True)


if __name__ == '__main__':
    main()
