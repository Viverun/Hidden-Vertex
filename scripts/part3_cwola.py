"""Part 3d CLI: CWoLa-style bump hunt. The protocol is fixed in configs/part3_cwola.yaml.

    python scripts/part3_cwola.py run --fraction 0.001 --seed 0     # one run (scores + metrics json)
    python scripts/part3_cwola.py all --workers 6 --threads 2       # every fraction and seed (skips done runs)
    python scripts/part3_cwola.py report                            # results/part3/cwola.md and cwola.json
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import part3_cwola  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['run', 'all', 'report'])
    parser.add_argument('--config', default='configs/part3_cwola.yaml')
    parser.add_argument('--fraction', type=float, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--rerun', action='store_true', help='all: rerun runs that already have scores')
    args = parser.parse_args(argv)
    cfg = part3_cwola.load_config(args.config)
    log = lambda m: print(m, flush=True)  # noqa: E731
    if args.command == 'run':
        if args.fraction is None:
            parser.error('run needs --fraction')
        if args.fraction not in [float(f) for f in cfg['injection']['fractions']]:
            parser.error(f"fraction must be one of {cfg['injection']['fractions']}")
        part3_cwola.run_one(cfg, args.fraction, args.seed, threads=args.threads, log=log)
    elif args.command == 'all':
        part3_cwola.run_all(cfg, workers=args.workers, threads=args.threads, skip_existing=not args.rerun, log=log)
    else:
        part3_cwola.make_report(cfg, log=log)


if __name__ == '__main__':
    main()
