# Part 3C (C2): training-side intervention. Same as the Part 2 main configuration (configs/part2_main.yaml),
# except split.match_multiplicity_to_signal=true: the 180k-event pure-background fit set is resampled with
# replacement so that its multiplicity follows the contam_sig signal reference (signal events not in the
# test set); weights p_sig(n)/p_bg(n) in 30 quantile bins of the fit-set multiplicity, clipped at the
# 99.5th percentile. Validation slice and test set are unchanged. Tag multmatch, seeds 0, 1, 2 (2 GPUs).
#
# Fixed predictions (written before the run; test AUC is not used to choose anything):
# - If the density scores only track multiplicity relative to the training distribution, then with
#   multiplicity-matched training the kNN and GMM seed-averaged AUC move to >= 0.47 (inversion mostly gone).
# - If they stay <= 0.45, the multiplicity explanation is not sufficient.
# - The Chamfer score is reported without a prediction.
#
# The benchmark report is built locally from the downloaded runs with `scripts/part2_run.py report`.
OVERRIDES = [f"data.h5_path={H5}", f"data.features_path={FEATURES}", f"data.cache_dir={CACHE}", f"output_dir={OUT}"]
flags = [x for o in OVERRIDES for x in ("--set", o)]
sh(sys.executable, "scripts/part2_run.py", "prepare", "--config", "configs/part2_main.yaml", *flags)
try:
    sh(sys.executable, "scripts/part2_sweep.py", "--stage", "multmatch", "--config", "configs/part2_main.yaml", *flags)
finally:
    for path in __import__("glob").glob(f"{OUT}/runs/*/model_s*.pt"):
        os.remove(path)
