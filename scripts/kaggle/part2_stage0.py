# Stage 0 body (runs inside the Kaggle kernel; see scripts/kaggle_bundle.py for the variables it can use).
# Builds the event cache, re-runs Part 1 (seed 0) to export both score reductions, builds the baseline
# bank on the frozen test set, and writes the Stage 0 table with the reference check.
import shutil
import yaml

OVERRIDES = [f"data.h5_path={H5}", f"data.features_path={FEATURES}", f"data.cache_dir={CACHE}", f"output_dir={OUT}"]


def p2(command, *extra, check=True):
    flags = [x for o in OVERRIDES for x in ("--set", o)]
    cmd = [sys.executable, "scripts/part2_run.py", command, "--config", "configs/part2_main.yaml", *flags, *extra]
    if check:
        sh(*cmd)
    else:  # reference-check failure must not skip the output cleanup in the footer
        print("exit code:", subprocess.run(cmd, cwd=HV, env=dict(os.environ, TQDM_DISABLE="1")).returncode)


p2("prepare")

part1 = yaml.safe_load(open("configs/part1_notebook_parity.yaml"))
part1["data"]["h5_path"] = H5
part1["data"]["split_path"] = f"{OUT}/part1/split.pt"
part1["evaluation"].update(model_path=f"{OUT}/part1/model.pt", metrics_path=f"{OUT}/part1/metrics.json",
                           scores_path=f"{OUT}/part1/scores.npz", roc_path=f"{OUT}/part1/roc.png")
os.makedirs(f"{OUT}/part1", exist_ok=True)
yaml.safe_dump(part1, open("/tmp/part1.yaml", "w"))
sh(sys.executable, "main.py", "prepare-part1", "--config", "/tmp/part1.yaml")
sh(sys.executable, "scripts/run_part1_eval.py", "--config", "/tmp/part1.yaml", "--seed", "0")
for junk in ("split.pt", "model.pt"):
    os.remove(f"{OUT}/part1/{junk}")

p2("bank", "--external", f"Part 1 AE (max)={OUT}/part1/scores.npz:scores_max",
   "--external", f"Part 1 AE (mean)={OUT}/part1/scores.npz:scores_mean")
p2("report", "--name", "stage0", "--check-reference", check=False)
