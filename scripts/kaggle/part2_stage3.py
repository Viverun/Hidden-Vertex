# Stage 3: contamination runs (9% and 1% signal in the training set).
OVERRIDES = [f"data.h5_path={H5}", f"data.features_path={FEATURES}", f"data.cache_dir={CACHE}", f"output_dir={OUT}"]
flags = [x for o in OVERRIDES for x in ("--set", o)]
sh(sys.executable, "scripts/part2_run.py", "prepare", "--config", "configs/part2_main.yaml", *flags)
sh(sys.executable, "scripts/part2_sweep.py", "--stage", "contamination", "--config", "configs/part2_main.yaml", *flags)
for path in __import__("glob").glob(f"{OUT}/runs/*/model_s*.pt"):
    os.remove(path)
