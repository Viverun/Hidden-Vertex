# Part 3 reverse-direction check (diagnostic)

**DIAGNOSTIC ONLY: the reverse scorers are fit on signal events (this uses signal labels). This is not an anomaly-detection result.**

Reverse fit set: signal rows 1,005,000–1,080,999 (76,000 events); AE validation: signal rows 1,081,000–1,089,999 (9,000). Quantile transform refit on the reverse fit set. Test set: the frozen Part 2 test set, scored with background as the positive ("anomalous") class.

| Score | forward AUC (bg-trained, signal positive) | forward per seed | reverse AUC (signal-trained, bg positive) | reverse 95% CI | reverse per seed |
|---|---|---|---|---|---|
| GMM NLL [jet5] | 0.694 | 0.694 / 0.694 / 0.695 | 0.903 | 0.898–0.909 | 0.906 / 0.904 / 0.900 |
| kNN dist [jet5] | 0.717 | 0.717 / 0.717 / 0.717 | 0.905 | 0.900–0.911 | 0.905 / 0.905 / 0.905 |
| AE recon [jet5] | 0.505 | 0.529 / 0.453 / 0.533 | 0.845 | 0.838–0.851 | 0.841 / 0.859 / 0.834 |
| GMM NLL [jet4] | 0.753 | 0.753 / 0.751 / 0.755 | 0.810 | 0.802–0.818 | 0.810 / 0.810 / 0.810 |
| kNN dist [jet4] | 0.761 | 0.761 / 0.761 / 0.761 | 0.791 | 0.782–0.799 | 0.791 / 0.791 / 0.791 |
| AE recon [jet4] | 0.539 | 0.528 / 0.543 / 0.545 | 0.635 | 0.627–0.644 | 0.646 / 0.667 / 0.592 |
