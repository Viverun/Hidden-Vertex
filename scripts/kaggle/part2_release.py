# Release: retrain the headline configuration (3 seeds) and keep the checkpoints, feature stats and training
# latents for the public model upload. Same config and sweep as Stage 1; only the checkpoint cleanup is dropped.
OVERRIDES = [f"data.h5_path={H5}", f"data.features_path={FEATURES}", f"data.cache_dir={CACHE}", f"output_dir={OUT}"]
flags = [x for o in OVERRIDES for x in ("--set", o)]
sh(sys.executable, "scripts/part2_run.py", "prepare", "--config", "configs/part2_main.yaml", *flags)
sh(sys.executable, "scripts/part2_sweep.py", "--stage", "main", "--config", "configs/part2_main.yaml", *flags)
# Export the frozen test events so the published checkpoints can be checked against the stored scores offline.
sys.path.insert(0, HV)
import numpy as np  # noqa: E402
from src.data.event_store import load_role  # noqa: E402
bg, sig = load_role(CACHE, "test_bg"), load_role(CACHE, "test_sig")
np.savez_compressed(f"{OUT}/test_events.npz", particles=np.concatenate([bg.particles, sig.particles]),
                    rows=np.concatenate([bg.rows, sig.rows]), labels=np.r_[np.zeros(len(bg.rows)), np.ones(len(sig.rows))])
