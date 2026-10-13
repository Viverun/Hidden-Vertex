# Stage 1: headline configuration, 3 seeds (2 GPUs in parallel), then the full benchmark report.
OVERRIDES = [f"data.h5_path={H5}", f"data.features_path={FEATURES}", f"data.cache_dir={CACHE}", f"output_dir={OUT}"]
flags = [x for o in OVERRIDES for x in ("--set", o)]
sh(sys.executable, "scripts/part2_run.py", "prepare", "--config", "configs/part2_main.yaml", *flags)
sh(sys.executable, "scripts/part2_sweep.py", "--stage", "main", "--config", "configs/part2_main.yaml", *flags)
# bank + report need the Part 1 scores from Stage 0; the final report is rebuilt locally (Task 12)
for path in __import__("glob").glob(f"{OUT}/runs/*/model_s*.pt"):
    os.remove(path)
