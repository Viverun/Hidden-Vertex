# Physics Background: The Hidden Vertex

This document gives the physics background of the project. It describes the Standard Model, the LHC data
selection, the signal in the dataset and the reason for an unsupervised search.

## 1. The Standard Model and new physics

The Standard Model (SM) of particle physics describes the known elementary particles and three of the four
forces. Many experiments confirm its predictions. The last SM particle, the Higgs boson, was found at the LHC
in 2012.

The SM does not explain some observations, for example dark matter and the asymmetry between matter and
antimatter. Thus, physicists search for particles and forces beyond the Standard Model (BSM).

## 2. Collisions at the LHC

The Large Hadron Collider (LHC) at CERN collides protons. Most collisions produce QCD jets. A jet is a narrow
spray of particles from a quark or a gluon.

Each particle in an event has three main kinematic values:

| Value | Name | Meaning |
|---|---|---|
| pT | Transverse momentum | Momentum perpendicular to the beam (GeV) |
| η | Pseudorapidity | Angle relative to the beam. η = 0 is perpendicular to the beam. |
| φ | Azimuthal angle | Angle around the beam, from −π to π |

The distance between two particles is ΔR = √(Δη² + Δφ²). Jet algorithms use this distance. The anti-kt algorithm
with R = 0.4 is a standard algorithm.

## 3. Data selection (the trigger)

The LHC produces approximately 40 million bunch crossings each second. The experiments can save approximately
1,000 events each second. Thus, the trigger system keeps approximately 0.0025% of the events and discards the
others in real time.

The trigger uses selection rules. Most rules look for the signatures of known processes or of predicted new
particles. If a new particle has a signature that no rule selects, the trigger can discard its events.

## 4. Supervised and unsupervised searches

| Search | Method | Limit |
|---|---|---|
| Supervised | A theory predicts a signal. The search selects events that agree with the prediction. | It can miss a signal that no theory predicts. |
| Unsupervised (anomaly detection) | A model learns the background. The search selects events that do not agree with the background. | It needs a score that measures the correct difference between signal and background. |

An unsupervised search does not need a model of the signal. But it has a problem: an "unusual" event for the
model is not always a signal event. This project shows an example of this problem (see Section 6).

## 5. The LHC Olympics 2020 R&D dataset

The LHC Olympics 2020 was a challenge for anomaly detection methods. Its R&D dataset has simulated events
(Pythia 8 and Delphes 3):

- **Background:** 1,000,000 QCD dijet events.
- **Signal:** 100,000 events of a heavy resonance: W′ → XY, with X → qq̄ and Y → qq̄.
  The masses are m(W′) = 3.5 TeV, m(X) = 500 GeV and m(Y) = 100 GeV.

The signal events have two large-radius jets. Each jet has two prongs because X and Y each decay to two quarks.
These variables separate the signal from the background:

| Variable | Meaning | Why it separates |
|---|---|---|
| mjj | Invariant mass of the two leading jets | The signal is a resonance at 3.5 TeV |
| mj1, mj2 | Jet masses | The jets come from X (500 GeV) and Y (100 GeV) |
| τ21 = τ2 / τ1 | N-subjettiness ratio | Two-prong jets have a low τ21 |
| HT | Sum of the pT of all particles | The signal has a high total pT |
| Multiplicity N | Number of particles | The signal has fewer particles |
| HT / N | Mean pT per particle | The signal has fewer particles with higher pT |

**Note:** the project name refers to long-lived particles. A long-lived particle travels a measurable distance
before it decays. Its decay products come from a displaced vertex, not from the collision point. The LHC
Olympics R&D signal is not a long-lived particle, and the dataset does not contain vertex information. Thus,
this project does not test the detection of displaced vertices.

## 6. Complexity bias

An autoencoder reconstructs simple events more easily than complex events. In this dataset, the signal events are
simpler than the background events in one way: they have fewer particles. The mean multiplicity is 140 for
signal and 182 for background.

Thus, an anomaly score that follows the event complexity gives the signal events lower scores than the
background events. The AUC is then below 0.5. Earlier work found the same effect for jet images:

- T. Heimel et al., "QCD or What?", SciPost Phys. 6 (2019) 030, arXiv:1808.08979.
- T. Finke et al., "Autoencoders for unsupervised anomaly detection in high energy physics", JHEP 06 (2021)
  161, arXiv:2104.09051.

The Part 2 results of this project show the same effect for an event-level graph autoencoder. The scores have an
AUC of 0.40 to 0.44. The simple variable HT/N gets 0.811 (see `PART2_RESULTS.md`).

Thus, an unsupervised score must be compared with simple baselines. Also, report the AUC in bins of
multiplicity. Then the effect of the complexity is visible.

## 7. Glossary

| Term | Meaning |
|---|---|
| AUC | Area under the ROC curve. 1.0 is a perfect separation, 0.5 is random, below 0.5 is inverted. |
| BSM | Beyond the Standard Model |
| Chamfer distance | A distance between two sets of points that does not depend on their sequence |
| Displaced vertex | A decay point away from the collision point |
| GAE | Graph autoencoder |
| HT | Sum of the transverse momenta of all particles in an event |
| Jet | A narrow spray of particles from a quark or a gluon |
| LLP | Long-lived particle |
| Multiplicity | Number of particles in an event |
| QCD | Quantum chromodynamics, the theory of the strong force |
| ROC curve | Signal efficiency as a function of background efficiency for all score thresholds |
| SM | Standard Model |
| Trigger | The real-time event selection system of an LHC experiment |
| τ21 | N-subjettiness ratio. A low value means a jet with two prongs. |

## 8. Further reading

- G. Kasieczka et al., "The LHC Olympics 2020", Rep. Prog. Phys. 84 (2021) 124201, arXiv:2101.08320.
- The paper of this project: [doi:10.5281/zenodo.23432779](https://doi.org/10.5281/zenodo.23432779).
