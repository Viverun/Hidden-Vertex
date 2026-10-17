"""Regenerate paper figures from the numbers in results/part2/final.md (AUC, 95%/98.3% CI)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = [  # label, auc, lo, hi, group
    (r"$-|m_{jj}-3500|$ (oracle)", 0.900, 0.894, 0.906, "oracle"),
    (r"mean $p_T$ per particle", 0.811, 0.803, 0.821, "particle"),
    (r"$-\tau_{21}$ (leading jet)", 0.801, 0.792, 0.810, "jet"),
    (r"$H_T$", 0.792, 0.783, 0.802, "particle"),
    (r"$m_{jj}$", 0.788, 0.778, 0.799, "jet"),
    ("Node-level GAE (mean)", 0.781, 0.772, 0.790, "model"),
    ("$-$multiplicity", 0.715, 0.704, 0.726, "particle"),
    ("Node-level GAE (max)", 0.478, 0.467, 0.489, "model"),
    ("Event GAE: GMM", 0.436, 0.426, 0.448, "model"),
    ("Event GAE: Chamfer", 0.428, 0.415, 0.442, "model"),
    ("Event GAE: kNN", 0.401, 0.389, 0.412, "model"),
]
colors = {"oracle": "#888888", "particle": "#2a6fb0", "jet": "#4c9a5f", "model": "#c0504d"}
fig, ax = plt.subplots(figsize=(6.2, 4.0))
for i, (lab, a, lo, hi, g) in enumerate(rows[::-1]):
    ax.errorbar(a, i, xerr=[[a - lo], [hi - a]], fmt="o", color=colors[g], capsize=2, ms=4)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=8)
ax.axvline(0.5, ls=":", c="k", lw=0.8)
ax.axvline(0.811, ls="--", c=colors["particle"], lw=0.8)
ax.set_xlabel("ROC AUC (signal vs. QCD background)")
ax.set_xlim(0.35, 0.95)
fig.tight_layout()
fig.savefig("figures/auc_summary.pdf")
fig.savefig("figures/auc_summary.png", dpi=200)
