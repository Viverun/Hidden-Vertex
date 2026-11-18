"""Part 3 CLI: jet-feature density scores (3A) and the WNAE (3B).

    python scripts/part3_run.py density                 # GMM / kNN / AE on jet5 and jet4, seeds 0,1,2
    python scripts/part3_run.py wnae-time               # time one WNAE epoch (no test scores)
    python scripts/part3_run.py wnae --seed 0           # train one WNAE seed (wnae.max_epochs must be set)
    python scripts/part3_run.py wnae --run wnae_long --seed 0   # long WNAE run (results/part3/wnae_long)
    python scripts/part3_run.py reverse                 # reverse-direction diagnostic (signal-trained)
    python scripts/part3_run.py report                  # results/part3/jet_density.md and wnae.md
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import part3  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['density', 'wnae-time', 'wnae', 'reverse', 'report'])
    parser.add_argument('--config', default='configs/part3_jet.yaml')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=None)
    parser.add_argument('--run', default='wnae', choices=['wnae', 'wnae_long'], help='WNAE run tag')
    parser.add_argument('--epochs', type=int, default=1, help='wnae-time: epochs to time')
    args = parser.parse_args(argv)
    cfg = part3.load_config(args.config)
    log = lambda m: print(m, flush=True)  # noqa: E731
    if args.command == 'density':
        part3.run_density(cfg, log=log)
    elif args.command == 'wnae-time':
        info = part3.time_wnae(cfg, seed=args.seed, threads=args.threads, log=log, run=args.run, epochs=args.epochs)
        log(f"one epoch: {info['seconds_epoch']:.1f} s with {info['threads']} threads")
    elif args.command == 'wnae':
        part3.run_wnae(cfg, args.seed, threads=args.threads, log=log, run=args.run)
    elif args.command == 'reverse':
        part3.run_reverse(cfg, log=log)
    else:
        part3.make_reports(cfg, log=log)


if __name__ == '__main__':
    main()
