# Timing smoke test: a 20k-event run to measure seconds/epoch on the T4 before committing to Stage 1.
OVERRIDES = [f"data.h5_path={H5}", f"data.features_path={FEATURES}", f"data.cache_dir={CACHE}",
             f"output_dir={OUT}", "split.n_train=20000", "split.n_contam_sig=1800", "training.epochs=3"]
flags = [x for o in OVERRIDES for x in ("--set", o)]
sh(sys.executable, "scripts/part2_run.py", "prepare", "--config", "configs/part2_main.yaml", *flags)
sh(sys.executable, "scripts/part2_run.py", "train", "--config", "configs/part2_main.yaml", "--tag", "smoke",
   "--seed", "0", *flags)
with open(f"{OUT}/runs/smoke/metrics_s0.json") as handle:
    print(handle.read()[:600])
