"""Relative importance of the four explanatory factors (Cambodia, project counts 2017-2020).

Shapley decomposition of explained deviance: the extra deviance explained by {lights, access, conn, prior_infra} over the
baseline (ln population + province FE) is split among the four factors by averaging each factor's marginal contribution over
all 24 orderings in which they could be added. Shares sum to 100% of what the four explain together.
Also reports joint-model standardised coefficients (the 'weights' of a multiple regression).
Output: outputs/tables/kh_shapley.csv, outputs/figures/cambodia/S3_importance.png
"""
import itertools
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

SRC = Path(__file__).with_name("kh_s06_ladder.py")
ns = {"__file__": str(SRC)}
exec(SRC.read_text(encoding="utf-8").split("LADDER = [")[0], ns)  # builds DF, cur, outcome()
d = ns["outcome"](ns["cur"]); ROOT = ns["ROOT"]
X = ["lights", "access", "conn", "prior_infra"]
LAB = {"lights": "Night lights (economy)", "access": "Physical access (roads)", "conn": "Digital connectivity", "prior_infra": "Prior infrastructure"}
dev = lambda xs: smf.glm("N ~ " + " + ".join(xs + ["lnpop", "C(prov)"]), data=d, family=sm.families.Poisson()).fit(maxiter=300).deviance
D0, Dnull = dev([]), smf.glm("N ~ 1", data=d, family=sm.families.Poisson()).fit().deviance
cache = {}
def v(s):
    k = tuple(sorted(s))
    if k not in cache:
        cache[k] = (D0 - dev(list(k))) / Dnull
    return cache[k]
phi = {x: 0.0 for x in X}
perms = list(itertools.permutations(X))
for p in perms:
    s = []
    for x in p:
        phi[x] += (v(s + [x]) - v(s)) / len(perms); s.append(x)
total = v(X)
m = smf.glm("N ~ " + " + ".join(X) + " + lnpop + C(prov)", data=d, family=sm.families.Poisson()).fit(
    cov_type="cluster", cov_kwds={"groups": pd.factorize(d.prov)[0]})
R = pd.DataFrame([{"factor": LAB[x], "shapley_pp": 100 * phi[x], "share_of_explained_pct": 100 * phi[x] / total,
                   "alone_pp": 100 * v([x]), "joint_coef_pct": 100 * (np.exp(m.params[x]) - 1), "joint_p": m.pvalues[x]} for x in X])
R = R.sort_values("shapley_pp", ascending=False)
R.to_csv(ROOT / "outputs/tables/kh_shapley.csv", index=False)
print(f"baseline pseudo-R2 (pop + province) = {1 - D0 / Dnull:.3f} | four factors add {100 * total:.1f} pp | full model = {1 - D0 / Dnull + total:.3f}")
print(R.round(2).to_string(index=False))

fig, ax = plt.subplots(figsize=(6.4, 2.2), dpi=220)
left = 0; cols = ["#1f2328", "#1f6feb", "#8c959f", "#bf8700"]
for (i, r), c in zip(R.iterrows(), cols):
    ax.barh(0, r.share_of_explained_pct, left=left, color=c, height=0.5, edgecolor="white", linewidth=1.5)
    ax.text(left + r.share_of_explained_pct / 2, 0, f"{r.share_of_explained_pct:.0f}%", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
    ax.text(left + r.share_of_explained_pct / 2, -0.42, r.factor, ha="center", va="top", fontsize=6.3, color="#1f2328")
    left += r.share_of_explained_pct
ax.set_xlim(0, 100); ax.set_ylim(-0.9, 0.5); ax.axis("off")
ax.set_title(f"Who explains where aid projects go? Shares of the {100 * total:.0f} pp of variation the four factors explain together "
             f"(beyond population and province)", fontsize=7, loc="left", color="#1f2328")
fig.tight_layout(); fig.savefig(ROOT / "outputs/figures/cambodia/S3_importance.png"); plt.close(fig)
