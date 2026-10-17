# Training: The Hidden Vertex

This document gives the training procedures for Part 1 and Part 2, the Kaggle procedure and the expected results.

## 1. Requirements

- Python 3.10 or later.
- The packages in `requirements.txt`. PyTorch Geometric is necessary for Part 1.
- The file `events_anomalydetection_v2.h5` (see `DATA.md`). Part 2 baselines also use
  `events_anomalydetection_v2.features.h5`.
- A GPU is recommended. All the results in this repository come from NVIDIA T4 GPUs on Kaggle.

Install the packages and run the tests:

```bash
pip install -r requirements.txt
python -m pytest tests
```

## 2. Part 1

### 2.1 Procedure

1. Make the oracle split:

   ```bash
   python main.py prepare-part1 --config configs/part1_notebook_parity.yaml
   ```

2. Train and evaluate the model:

   ```bash
   python scripts/run_part1_eval.py --config configs/part1_notebook_parity.yaml
   ```

3. To use a different seed, add `--seed N`.

### 2.2 Settings

The values are in `configs/part1_notebook_parity.yaml`:

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.005 |
| Weight decay | 0 |
| Epochs | 5 |
| Batch size | 256 |
| Loss | `pt_only` (MSE of the standardized log pT) |
| Dropout | 0.1 |
| Training set | 40,000 background events |
| Score | `max` (largest particle error) |

These settings are the same as in the notebook. The notebook did not use a seed. The repository uses
`training.seed: 0`.

### 2.3 Outputs

| File | Contents |
|---|---|
| `results/models/part1_purified_baseline.pt` | Trained model |
| `results/scores/part1_metrics.json` | AUC, bootstrap 95% confidence interval, seed |
| `results/scores/part1_scores.npz` | Score and label of each test event (`scores_max`, `scores_mean`) |
| `results/figures/part1_roc.png` | ROC curve |

### 2.4 Expected results

| Value | Expected |
|---|---|
| Training loss | approximately 0.62 at the start, approximately 0.51 at the end |
| AUC (`max`) | 0.483 ± 0.005 (seeds 0 to 4) |
| AUC (`mean`) | 0.781 (seed 0) |
| Time for one seed | 53 to 81 s on one T4 |

### 2.5 Training on the full file

The command `train-default` trains on graph chunks of the full file:

1. Make the chunks: `python scripts/process_h5.py`.
2. Train: `python main.py train-default`. The default configuration is `configs/default_search.yaml`. To use a different file, add `--config <file>`.

This command uses AdamW and selects the background events with `labels.npy`.

## 3. Part 2

### 3.1 Procedure

1. Make the event cache:

   ```bash
   python scripts/part2_run.py prepare --config configs/part2_main.yaml
   ```

2. Train the three main seeds:

   ```bash
   python scripts/part2_sweep.py --stage main --config configs/part2_main.yaml
   ```

3. Calculate the baselines (the "bank"). To include the Part 1 scores, add an `--external` option for each
   score:

   ```bash
   python scripts/part2_run.py bank --config configs/part2_main.yaml \
       --external "Part 1 AE (max)=<scores.npz>:scores_max" \
       --external "Part 1 AE (mean)=<scores.npz>:scores_mean"
   ```

4. Make the report:

   ```bash
   python scripts/part2_run.py report --config configs/part2_main.yaml --tags main --name stage1
   ```

To train one seed only, use `python scripts/part2_run.py train --config configs/part2_main.yaml --seed 0 --tag main`.

To change a configuration value, add `--set key=value`. For example: `--set model.latent_dim=8`.

### 3.2 Settings

The values are in `configs/part2_main.yaml`:

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.00001 |
| Epochs | 30 (maximum) |
| Early stop | after 5 epochs without a better validation loss |
| Batch size | 256 |
| Gradient clip | 5.0 (norm) |
| Loss | Symmetric Chamfer distance |
| Training set | 180,000 background events |
| Validation set | 20,000 background events |

### 3.3 Checks during training

`train_event_ae()` in `src/training/event_trainer.py` does these checks in each epoch:

- If the training loss or the validation loss is not finite, the training stops with `TrainingError`.
- If the largest standard deviation of a latent dimension is less than `collapse_std` (0.0001), the training
  stops. This means that the latent space collapsed.

At the end, the model goes back to the epoch with the best validation loss.

The training writes a line for each epoch: the training loss, the validation loss, the largest latent standard
deviation and the time.

### 3.4 Outputs

Each run writes to `results/part2/runs/<tag>/`:

| File | Contents |
|---|---|
| `metrics_s<seed>.json` | AUCs, settings, loss history |
| `scores_s<seed>.npz` | Chamfer, kNN and GMM score, label and HDF5 row of each test event |
| `model_s<seed>.pt` | Model weights, feature means and standard deviations, configuration |
| `latents_fit_s<seed>.npy` | Latents of the 180,000 training events |

The scores file is written last. If it exists, the run is complete. `part2_sweep.py` does not repeat a complete
run. Thus, if a run stops because of a time limit, start the sweep again and it continues.

### 3.5 Sweep stages

| Stage | Runs |
|---|---|
| `main` | Main configuration, seeds 0, 1 and 2 |
| `ablations` | Latent 4, 8, 32; K = 24, 96; depth 2, 4; no pT-weighted pooling; pT-weighted loss. Seeds 0 and 1. |
| `contamination` | 9% and 1% signal in the training set. Seed 0. |

The sweep starts one training process for each GPU.

### 3.6 Expected results

| Value | Expected |
|---|---|
| Best validation loss | 0.12 to 0.13 |
| Time for one epoch | approximately 34 s on one T4 |
| Time for one seed | approximately 18 min |
| AUC (Chamfer / kNN / GMM), seed 0 | 0.419 / 0.405 / 0.448 |

A new training run on Kaggle T4 GPUs gave the same AUCs as the original runs for all three seeds.

## 4. Kaggle procedure

The repository makes Kaggle script kernels that contain the full code:

1. Make the kernel:

   ```bash
   python scripts/kaggle_bundle.py --slug hidden-vertex-part2-stage1 --title "Hidden Vertex Part2 Stage1" \
       --body scripts/kaggle/part2_stage1.py --out <directory>
   ```

2. Send it to Kaggle:

   ```bash
   kaggle kernels push -p <directory>
   ```

3. Get the outputs when the run is complete:

   ```bash
   kaggle kernels output viveriun/hidden-vertex-part2-stage1 -p <directory>
   ```

The kernel uses two T4 GPUs and the dataset `atulchikku/lhc-olympics-2020-ad-r-and-d`. The kernel bodies for
each stage are in `scripts/kaggle/`. The kernel `part2_release.py` keeps the checkpoints. The other kernels
delete them to save space.

## 5. Reproducibility

- Each run uses a seed. Part 1: `training.seed` or `--seed`. Part 2: `--seed`.
- Repeated runs with the same seed give the same AUC on the same hardware.
- The Part 2 test set is fixed. All runs use the same 10,000 test events.
- The sign of each score was fixed before the test: a higher score means a more anomalous event.

## 6. Problems and solutions

| Problem | Cause | Solution |
|---|---|---|
| The Part 1 test set has no signal events | The v2 file is sorted, and `scan_limit` is too small | Set `scan_limit: null` |
| `cache ... was built with different split/max_particles settings` | The configuration changed after the cache was made | Delete the cache or set a different `data.cache_dir` |
| More than 2% of the events are truncated | `data.max_particles` is too small | Increase `data.max_particles` (default 350) |
| `TrainingError: latent collapsed` | The latent space collapsed | Examine the learning rate and the loss settings |
| `KeyError: 'mean'` when the code reads `norm_stats.pt` | An old version of the code | Use the current code. It reads `feat_mean` and `feat_std`. |
