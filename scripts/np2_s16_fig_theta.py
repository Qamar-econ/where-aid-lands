"""Figure: population elasticity theta across samples (from outputs/results_pop_elasticity.txt). theta = 1 is per-capita allocation."""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rows = []
for l in (ROOT / "outputs/results_pop_elasticity.txt").read_text(encoding="utf-8").splitlines():
    m = re.match(r"(.+?)\s+(pop only|\+ covariates)\s+theta ([\d.-]+) \(95% CI ([\d.-]+)-([\d.-]+);", l)
    if m:
        rows.append((m.group(1).strip(), m.group(2), *map(float, m.groups()[2:])))
fig, ax = plt.subplots(figsize=(5.6, 2.5), dpi=200)
for i, (s, spec, t, lo, hi) in enumerate(rows[::-1]):
    c = "#1f6feb" if hi < 1 else "#8c959f"
    ax.plot([lo, hi], [i, i], color=c, lw=1.8, solid_capstyle="round"); ax.plot(t, i, "o", color=c, ms=5, mec="white", mew=1)
ax.axvline(1, color="#1f2328", lw=0.9, ls="--"); ax.text(0.98, -0.75, "θ = 1: per capita", fontsize=6, color="#1f2328", ha="right")
ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{s} · {spec}" for s, spec, *_ in rows[::-1]], fontsize=6.5)
ax.set_xlabel("Population elasticity of new projects θ (95% CI)", fontsize=7); ax.tick_params(axis="x", labelsize=6.5, colors="#6e7781")
for sp in ["top", "right", "left"]:
    ax.spines[sp].set_visible(False)
fig.suptitle("Aid follows units, not people (blue: 95% CI excludes θ = 1)", fontsize=8, x=0.01, ha="left", color="#1f2328")
fig.tight_layout(); fig.savefig(ROOT / "outputs/figures/fig_theta_samples.png")
print(len(rows), "rows")
