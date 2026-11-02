"""Wild cluster score bootstrap for the main PPML models (Kline & Santos 2012), clustering by province.

Few clusters (19 provinces contribute) can make sandwich p-values too small. The score bootstrap imposes the null
(beta_X = 0), takes each province's summed score of the full model at the restricted estimates, flips it with Webb
six-point weights (better than Rademacher when clusters are few), and compares the observed score-based t statistic
with its bootstrap distribution. Reported alongside: the unrestricted variant (scores at the full-model estimates, Wald t), the share of the score
variance carried by each province, and leave-one-province-out estimates. When one or two clusters dominate, the
null-imposed version loses power and the unrestricted one over-rejects (MacKinnon and Webb 2018), so both are shown.
This is an added robustness check; the pre-registered decision rule (most conservative of clustered, Conley and
randomisation-inference p) is unchanged.
Also: the road-access model without the two most influential provinces, and a province-by-province sign test.
Output: outputs/tables/kh_wild_bootstrap.csv, kh_province_check.csv
"""
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy.stats import spearmanr, binomtest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "scripts/kh_s04_models.py").read_text(encoding="utf-8")
exec(src[:src.index("\nbase = A[A.level")])          # DF, A, outcome() as in the main script
RNG_B = np.random.default_rng(20261102)
B = 9999
WEBB = np.array([-np.sqrt(1.5), -1, -np.sqrt(.5), np.sqrt(.5), 1, np.sqrt(1.5)])


def score_bootstrap(d, var):
    prov = pd.get_dummies(d.prov, drop_first=True, dtype=float)
    Xr = np.column_stack([np.ones(len(d)), d.lnpop, prov.values])          # restricted design (null: no X)
    X = np.column_stack([d[var].values, Xr])                                 # full design, X first
    y = d.N.values
    mu = sm.GLM(y, Xr, family=sm.families.Poisson()).fit(maxiter=300).fittedvalues
    H = X.T @ (X * mu[:, None])                                              # information at the restricted fit
    Hi = np.linalg.pinv(H)
    S = X * (y - mu)[:, None]
    g = pd.factorize(d.prov)[0]; G = g.max() + 1
    Sg = np.vstack([S[g == k].sum(0) for k in range(G)])                     # province score sums
    V = (Hi @ (Sg.T @ Sg) @ Hi)[0, 0]
    t0 = (Hi @ Sg.sum(0))[0] / np.sqrt(V)
    W = WEBB[RNG_B.integers(0, 6, size=(B, G))]                              # B x G weights
    Sstar = W @ Sg                                                           # perturbed total scores
    num = Sstar @ Hi[0]
    # variance recomputed for each draw (weights change the cluster outer products for Webb weights)
    Vb = np.einsum("bg,gk,gl,k,l->b", W ** 2, Sg, Sg, Hi[0], Hi[0])
    tb = num / np.sqrt(Vb)
    p_r = float((1 + np.sum(np.abs(tb) >= abs(t0))) / (1 + len(tb)))
    # unrestricted variant: scores at the full-model estimates, observed Wald t
    r = sm.GLM(y, X, family=sm.families.Poisson()).fit(maxiter=300); mu1 = r.fittedvalues
    H1 = X.T @ (X * mu1[:, None]); H1i = np.linalg.pinv(H1); S1 = X * (y - mu1)[:, None]
    Sg1 = np.vstack([S1[g == k].sum(0) for k in range(G)])
    se1 = np.sqrt((H1i @ (Sg1.T @ Sg1) @ H1i)[0, 0]); tw = r.params[0] / se1
    num1 = (W @ Sg1) @ H1i[0]; Vb1 = np.einsum("bg,gk,gl,k,l->b", W ** 2, Sg1, Sg1, H1i[0], H1i[0])
    p_u = float((1 + np.sum(np.abs(num1 / np.sqrt(Vb1)) >= abs(tw))) / (1 + len(Vb1)))
    share = (Sg1 @ H1i[0]) ** 2; share = pd.Series(share / share.sum(), index=pd.factorize(d.prov)[1]).sort_values(ascending=False)
    lo = [100 * (np.exp(sm.GLM.from_formula(f"N ~ {var} + lnpop + C(prov)", data=d[d.prov != k], family=sm.families.Poisson()).fit(maxiter=300).params[var]) - 1)
          for k in np.unique(d.prov)]
    return float(t0), p_r, G, float(tw), p_u, ", ".join(f"{k}: {v:.2f}" for k, v in share.head(2).items()), min(lo), max(lo)


d0 = outcome(A[A.level == "adm2"])
main = pd.read_csv(T / "kh_main.csv").set_index("var")
rows = []
for v in ["combined", "phys", "conn", "ntl"]:
    t0, p, G, tw, pu, top, lo, hi = score_bootstrap(d0, v)
    rows.append({"var": v, "pct": main.loc[v, "pct"], "score_t": t0, "p_wild_score": p, "wald_t": tw, "p_wild_unrestricted": pu,
                 "top2_province_score_share": top, "leave_one_province_out_min_pct": lo, "leave_one_province_out_max_pct": hi, "clusters": G,
                 "p_cluster": main.loc[v, "p_cluster"], "p_ri": main.loc[v, "p_ri"], "p_conservative": main.loc[v, "p_conservative"]})
R = pd.DataFrame(rows)
R.to_csv(T / "kh_wild_bootstrap.csv", index=False)
print(f"Webb weights, B = {B}, clusters = {R.clusters.iloc[0]}")
print(R.round(4).to_string(index=False))

# ---------- does any province carry the result? ----------
fitp = lambda dd: smf.glm("N ~ phys + lnpop + C(prov)", data=dd, family=sm.families.Poisson()).fit(
    cov_type="cluster", cov_kwds={"groups": pd.factorize(dd.prov)[0]}, maxiter=300)
top2 = [k.strip().split(":")[0] for k in R.loc[R["var"] == "phys", "top2_province_score_share"].iloc[0].split(",")]
rows = []
for lab, dd in [("all provinces", d0)] + [(f"without province {k}", d0[d0.prov != k]) for k in top2] + [("without both", d0[~d0.prov.isin(top2)])]:
    r = fitp(dd); rows.append({"check": lab, "pct": 100 * (np.exp(r.params["phys"]) - 1), "p_cluster": r.pvalues["phys"], "districts": len(dd)})
pos = tot = 0
for k, g in d0.groupby("prov"):
    if len(g) < 3 or g.N.sum() == 0:
        continue
    rho = spearmanr(g.phys, g.N / g.pop_2019).correlation
    if not np.isnan(rho):
        tot += 1; pos += rho > 0
rows.append({"check": f"provinces where better-connected districts get more projects per person: {pos} of {tot}",
             "pct": np.nan, "p_cluster": binomtest(pos, tot, 0.5).pvalue, "districts": np.nan})
PC = pd.DataFrame(rows); PC.to_csv(T / "kh_province_check.csv", index=False)
print(PC.round(4).to_string(index=False))
