# Overnight results, 2026-11-02

All training and independent inference have finished. No Kaggle upload was performed. The user-reported official total remains 0.81; there is no new official score.

Task2: the existing default remains overall 0.82814858, ongoing 0.88026484, onset 0.77603231. Every completed plan was compared using the same pre-existing January-only horizon threshold family. October selected the candidate identities before January; all 14 candidates are reported, without filtering by February. The October-preselected recommendation did not beat the default, so no default was replaced.

Descriptive February point leader hard_examples__hard05_a100: overall 0.82889697, ongoing 0.88176163, unchanged onset 0.77603231; ongoing FP614/FN431 versus default FP595/FN470. The main-roster ongoing day-cluster 95% delta CI is [0.00031869, 0.00294240]. This is an unadjusted descriptive interval after reporting 14 candidates on reused development data, not a blind-test or multiplicity-adjusted confirmation. Old-roster overall is 0.77102551 versus 0.76973282; its gain interval crosses zero. No road/window model splicing occurred.

Independent inference rebuilt all 394 ongoing windows across the same two February populations: raw X, R127 parent probabilities, labels and masks exactly matched caches; all 14 candidates reproduced IoU/FP/FN. Sixteen future-label mutation checks passed, onset dependency hashes remained unchanged. Full original archives and all profiles are retained locally under outputs/retained_night_horizon_family_v3_20261102.

Task1: the already published full55 candidate remains S_state 0.943117149, with observed accumulation MAE reduced 2.0239%; this is not an official physics score or total. Both additional fixed-seed replications improved aggregate state and accumulation proxies in January and February samples. The first improved state on all ten roads in both months; the second improved all ten in February but has one January road regression. They were retained without choosing seeds or ensembles on February. These sampled replications are not full55 reevaluations.

The model asset provides both hard-example candidate heads with unchanged 900-tree parents, explicit January horizon thresholds, standalone X274/parent NPZ inference, plus provenance. Frozen onset and R127 remain unchanged dependencies in the earlier published packages; inference accepts their parent probabilities as input. See the release linked below. All candidates and old models remain retained.

[Model asset](https://github.com/chengweiliu669-pixel/25-day-a-Big-Dream-submission/releases/tag/overnight-final-20261102)
