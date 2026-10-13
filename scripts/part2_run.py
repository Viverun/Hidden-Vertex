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
        with open(os.path.join(cfg['output_dir'], f'{args.name}.md'), encoding='utf-8') as handle:
            print(handle.read())
        if args.check_reference and out['reference'] and not all(r['ok'] for r in out['reference'].values()):
            print('STAGE 0 REFERENCE CHECK FAILED: harness does not reproduce the verified Part 1 numbers')
            sys.exit(2)


if __name__ == '__main__':
    main()
