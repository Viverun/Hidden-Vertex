# Part 2 benchmark: final

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
| EventAE chamfer [k24] | variant | 0.502 | 0.489–0.515 | 98.3% | 0.360 | 2 seed(s) |
| EventAE chamfer [k96] | variant | 0.490 | 0.477–0.504 | 98.3% | 0.380 | 2 seed(s) |
| EventAE gmm [z32] | variant | 0.485 | 0.474–0.496 | 98.3% | 0.376 | 2 seed(s) |
| Part 1 AE (max) | external | 0.478 | 0.467–0.489 | 95.0% | 0.376 | from scores.npz |
| EventAE gmm [k24] | variant | 0.475 | 0.462–0.487 | 98.3% | 0.367 | 2 seed(s) |
| EventAE chamfer [ptw_loss] | variant | 0.475 | 0.461–0.489 | 98.3% | 0.332 | 2 seed(s) |
| EventAE gmm [ptw_loss] | variant | 0.468 | 0.454–0.481 | 98.3% | 0.324 | 2 seed(s) |
| EventAE gmm [contam1] | variant | 0.461 | 0.448–0.474 | 98.3% | 0.313 | 1 seed(s) |
| EventAE knn [ptw_loss] | variant | 0.454 | 0.440–0.467 | 98.3% | 0.307 | 2 seed(s) |
| EventAE gmm [nopt_pool] | variant | 0.451 | 0.440–0.463 | 98.3% | 0.345 | 2 seed(s) |
| EventAE gmm [k96] | variant | 0.447 | 0.434–0.460 | 98.3% | 0.317 | 2 seed(s) |
| EventAE knn [k24] | variant | 0.447 | 0.434–0.460 | 98.3% | 0.334 | 2 seed(s) |
| EventAE gmm [depth2] | variant | 0.442 | 0.431–0.455 | 98.3% | 0.329 | 2 seed(s) |
| EventAE gmm | model | 0.436 | 0.426–0.448 | 98.3% | 0.308 | 3 seed(s) |
| EventAE chamfer [depth4] | variant | 0.429 | 0.415–0.441 | 98.3% | 0.316 | 2 seed(s) |
| EventAE chamfer | model | 0.428 | 0.415–0.442 | 98.3% | 0.311 | 3 seed(s) |
| EventAE chamfer [z32] | variant | 0.428 | 0.415–0.441 | 98.3% | 0.327 | 2 seed(s) |
| EventAE gmm [depth4] | variant | 0.428 | 0.418–0.439 | 98.3% | 0.323 | 2 seed(s) |
| EventAE knn [k96] | variant | 0.425 | 0.413–0.439 | 98.3% | 0.294 | 2 seed(s) |
| EventAE chamfer [contam1] | variant | 0.424 | 0.410–0.438 | 98.3% | 0.308 | 1 seed(s) |
| EventAE chamfer [nopt_pool] | variant | 0.424 | 0.411–0.437 | 98.3% | 0.320 | 2 seed(s) |
| EventAE chamfer [depth2] | variant | 0.421 | 0.408–0.434 | 98.3% | 0.301 | 2 seed(s) |
| EventAE knn [contam1] | variant | 0.420 | 0.408–0.434 | 98.3% | 0.276 | 1 seed(s) |
| EventAE chamfer [contam9] | variant | 0.413 | 0.399–0.427 | 98.3% | 0.315 | 1 seed(s) |
| EventAE knn [depth2] | variant | 0.412 | 0.400–0.424 | 98.3% | 0.306 | 2 seed(s) |
| EventAE knn [z32] | variant | 0.412 | 0.400–0.423 | 98.3% | 0.306 | 2 seed(s) |
| EventAE knn [nopt_pool] | variant | 0.408 | 0.396–0.420 | 98.3% | 0.306 | 2 seed(s) |
| EventAE knn [z4] | variant | 0.405 | 0.393–0.416 | 98.3% | 0.262 | 2 seed(s) |
| EventAE gmm [z4] | variant | 0.403 | 0.391–0.415 | 98.3% | 0.251 | 2 seed(s) |
| EventAE chamfer [z8] | variant | 0.401 | 0.387–0.415 | 98.3% | 0.269 | 2 seed(s) |
| EventAE knn | model | 0.401 | 0.389–0.412 | 98.3% | 0.281 | 3 seed(s) |
| EventAE knn [depth4] | variant | 0.388 | 0.377–0.400 | 98.3% | 0.288 | 2 seed(s) |
| EventAE gmm [contam9] | variant | 0.387 | 0.374–0.400 | 98.3% | 0.269 | 1 seed(s) |
| EventAE gmm [z8] | variant | 0.376 | 0.364–0.389 | 98.3% | 0.306 | 2 seed(s) |
| EventAE knn [z8] | variant | 0.372 | 0.359–0.385 | 98.3% | 0.290 | 2 seed(s) |
| EventAE chamfer [z4] | variant | 0.366 | 0.351–0.380 | 98.3% | 0.244 | 2 seed(s) |
| EventAE knn [contam9] | variant | 0.362 | 0.349–0.374 | 98.3% | 0.257 | 1 seed(s) |

## Stage 0 reference check (expected value must lie inside the harness CI)

| Score | Expected | Measured | CI | OK |
|---|---|---|---|---|
| HT (sum pT) | 0.792 | 0.792 | 0.783–0.802 | yes |
| mjj | 0.788 | 0.788 | 0.778–0.799 | yes |
| Part 1 AE (max) | 0.478 | 0.478 | 0.467–0.489 | yes |
| Part 1 AE (mean) | 0.781 | 0.781 | 0.772–0.790 | yes |

## Gate

Bar: mean pT per particle (AUC 0.811). Result: **NOT PASSED**

| Model score | AUC | CI lower | beats bar | min within-bin | within-bin ok | passes |
|---|---|---|---|---|---|---|
| EventAE chamfer | 0.428 | 0.415 | False | 0.311 | False | False |
| EventAE knn | 0.401 | 0.389 | False | 0.281 | False | False |
| EventAE gmm | 0.436 | 0.426 | False | 0.308 | False | False |
