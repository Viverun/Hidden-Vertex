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


def summarize_stage(output_dir, stage):
    """Summary rows of the finished runs of a stage. The scores file is the "run finished" marker (written
    last), so a metrics file without it is ignored; an unreadable metrics file is skipped with a warning."""
    rows = []
    for tag, _, seeds in GRIDS[stage]:
        for seed in seeds:
            run_dir = os.path.join(output_dir, 'runs', tag)
            metrics_path = os.path.join(run_dir, f'metrics_s{seed}.json')
            if not (os.path.exists(metrics_path) and os.path.exists(os.path.join(run_dir, f'scores_s{seed}.npz'))):
                continue
            try:
                with open(metrics_path, encoding='utf-8') as handle:
                    m = json.load(handle)
                rows.append({'tag': tag, 'seed': seed, **{f'auc_{k}': round(v, 4) for k, v in m['auc'].items()},
                             'epochs': m['epochs_run'], 'sec_per_epoch': round(m['seconds_per_epoch'], 1),
                             'params': m['num_parameters']})
            except (OSError, ValueError, KeyError, TypeError) as err:
                print(f'WARNING: skipping unreadable metrics {metrics_path}: {err}', flush=True)
    return rows


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

    with open(os.path.join(REPO, args.config), encoding='utf-8') as handle:
        output_dir = yaml.safe_load(handle)['output_dir']
    for item in args.overrides:
        if item.startswith('output_dir='):
            output_dir = yaml.safe_load(item.partition('=')[2])
    output_dir = os.path.join(REPO, output_dir)  # the children run with cwd=REPO; keep both views identical

    jobs = [j for j in build_jobs(args.stage)
            if not os.path.exists(os.path.join(output_dir, 'runs', j[0], f'scores_s{j[1]}.npz'))]
    n_gpu = gpu_count()
    slots = [str(i % n_gpu) if n_gpu else None for i in range(max(n_gpu, 1) * args.jobs_per_gpu)]
    print(f'{len(jobs)} runs pending on {n_gpu} GPU(s), {len(slots)} parallel slot(s)', flush=True)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)

    t0 = time.time()
    running, results, pending = {}, [], list(jobs)
    try:
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
                log = open(os.path.join(output_dir, 'logs', f'{tag}_s{seed}.log'), 'w', encoding='utf-8')
                try:
                    proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
                except BaseException:
                    log.close()
                    raise
                running[slot] = (proc, log, tag, seed, time.time())
            time.sleep(2)
            for slot, (proc, log, tag, seed, started) in list(running.items()):
                if proc.poll() is None:
                    continue
                log.close()
                del running[slot]
                results.append({'tag': tag, 'seed': seed, 'returncode': proc.returncode,
                                'seconds': round(time.time() - started, 1)})
                print(f'[{time.time() - t0:.0f}s] {tag}/s{seed} rc={proc.returncode}', flush=True)
    finally:
        for proc, log, *_ in running.values():  # only non-empty if we are unwinding on an error / Ctrl-C
            proc.terminate()
            proc.wait()
            log.close()

    # Summary of every finished run of this stage (including runs from earlier launches)
    rows = summarize_stage(output_dir, args.stage)
    if rows:
        with open(os.path.join(output_dir, f'sweep_{args.stage}.csv'), 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    failed = [r for r in results if r['returncode'] != 0]
    print(f"done; failed: {[(r['tag'], r['seed']) for r in failed] or 'none'}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
