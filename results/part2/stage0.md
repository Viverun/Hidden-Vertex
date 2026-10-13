# Part 2 benchmark: stage0

| Score | Group | AUC | CI | CI level | min within-bin AUC | Notes |
|---|---|---|---|---|---|---|
| -|mjj-3500| (oracle) | oracle | 0.900 | 0.894–0.906 | 95.0% | 0.777 | needs the signal mass; not model independent |
| mean pT per particle | particle | 0.811 | 0.803–0.821 | 95.0% | 0.599 | = HT / multiplicity |
| -tau21 (leading jet) | jet | 0.801 | 0.792–0.810 | 95.0% | 0.664 | two-prong substructure |
| HT (sum pT) | particle | 0.792 | 0.783–0.802 | 95.0% | 0.337 | more energy = anomalous |
| mjj | jet | 0.788 | 0.778–0.799 | 95.0% | 0.322 | dijet invariant mass |
| Part 1 AE (mean) | external | 0.781 | 0.772–0.790 | 95.0% | 0.609 | from scores.npz |
| -multiplicity | particle | 0.715 | 0.704–0.726 | 95.0% | 0.399 | fewer particles = anomalous (sign seen on labelled data in Part 1) |
| mj2 | jet | 0.637 | 0.626–0.649 | 95.0% | 0.566 | second jet mass |
| mj1 | jet | 0.634 | 0.623–0.645 | 95.0% | 0.541 | leading jet mass |
| leading pT | particle | 0.554 | 0.543–0.565 | 95.0% | 0.388 |  |
| m_all | particle | 0.516 | 0.504–0.527 | 95.0% | 0.182 | invariant mass of all particles |
| Part 1 AE (max) | external | 0.478 | 0.467–0.489 | 95.0% | 0.376 | from scores.npz |

## Stage 0 reference check (expected value must lie inside the harness CI)

| Score | Expected | Measured | CI | OK |
|---|---|---|---|---|
| HT (sum pT) | 0.792 | 0.792 | 0.783–0.802 | yes |
| mjj | 0.788 | 0.788 | 0.778–0.799 | yes |
| Part 1 AE (max) | 0.478 | 0.478 | 0.467–0.489 | yes |
| Part 1 AE (mean) | 0.781 | 0.781 | 0.772–0.790 | yes |
