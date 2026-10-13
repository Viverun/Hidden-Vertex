# Stage 2: ablations (sensitivity analysis only).
OVERRIDES = [f"data.h5_path={H5}", f"data.features_path={FEATURES}", f"data.cache_dir={CACHE}", f"output_dir={OUT}"]
flags = [x for o in OVERRIDES for x in ("--set", o)]
sh(sys.executable, "scripts/part2_run.py", "prepare", "--config", "configs/part2_main.yaml", *flags)
try:
    sh(sys.executable, "scripts/part2_sweep.py", "--stage", "ablations", "--config", "configs/part2_main.yaml", *flags)
finally:
    for path in __import__("glob").glob(f"{OUT}/runs/*/model_s*.pt"):
        os.remove(path)
