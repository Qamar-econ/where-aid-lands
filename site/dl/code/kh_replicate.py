"""Stand-alone check of the headline results from two downloadable files. Needs only numpy, pandas, scipy, statsmodels.

Put kh_analysis.csv and kh_access.csv in the same folder as this script and run:  python kh_replicate.py  [--draws 2000]

1. Rebuilds the physical-access index from its ten components (kh_access.csv) and checks it against the published index.
2. Single-factor PPML with province fixed effects and ln population, for physical access, digital access (mobile speed),
   night lights and poverty: percent effect per SD and four p-values (province-clustered, Conley 50 km, Conley 100 km,
   Freedman-Lane randomisation inference); the reported p is the largest.
3. The four-factor model and its joint test (clustered Wald).
Randomisation inference uses random draws, so its p-values can differ from the published ones in the third decimal.
"""
import argparse, math
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy.stats import norm, chi2

ap = argparse.ArgumentParser(); ap.add_argument("--draws", type=int, default=2000); args = ap.parse_args()
HERE = Path(__file__).resolve().parent
d = pd.read_csv(HERE / "kh_analysis.csv"); d["prov"] = d.prov.astype(str)
z = lambda s: (s - s.mean()) / s.std()
rng = np.random.default_rng(1)

# ---------- 1. physical-access index from its components ----------
a = pd.read_csv(HERE / "kh_access.csv").set_index("k")
comp = pd.DataFrame({"neg_ln_tt_town_dry": -np.log(a.tt_town_dry), "neg_ln_tt_town_wet": -np.log(a.tt_town_wet), "neg_ln_tt_hq": -np.log(a.tt_hq_dry),
                     "neg_ln_tt_hosp": -np.log(a.tt_hosp_dry), "neg_ln_walk": -np.log(a.walk_min), "share_walk30": a.share_walk30,
                     "ln_road_density": np.log(a.road_density), "paved_share": a.paved_share, "neg_wet_penalty": -a.wet_penalty_share,
                     "neg_steep_share": -a.steep_share}).replace([np.inf, -np.inf], np.nan)
comp = comp.fillna(comp.median()); Zc = (comp - comp.mean()) / comp.std()
w, V = np.linalg.eigh(np.cov(Zc.T)); load = V[:, -1] * np.sign(V[:, -1].sum())
idx = pd.Series(Zc.values @ load, index=Zc.index); idx = (idx - idx.mean()) / idx.std()
print("1. Physical-access index rebuilt from 10 components: PC1 explains", f"{w[-1] / w.sum():.0%};",
      "correlation with published index =", round(np.corrcoef(idx.reindex(d.unit), d.access_pc1)[0, 1], 4))
print("   loadings:", dict(zip(comp.columns, load.round(3))))

# ---------- helpers ----------


def fit(f, data):
    return smf.glm(f, data=data, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(data.prov)[0]}, maxiter=300)


def conley_cov(r, data, cutoff):
    X = r.model.exog; mu = r.fittedvalues.values; S = X * (r.model.endog - mu)[:, None]
    Dm = np.hypot(data.x_km.values[:, None] - data.x_km.values[None, :], data.y_km.values[:, None] - data.y_km.values[None, :])
    K = np.clip(1 - Dm / cutoff, 0, None)
    bread = np.linalg.pinv(X.T @ (X * mu[:, None]))
    return bread @ (S.T @ K @ S) @ bread


def ri(data, var, others, f, b_obs):
    """Freedman-Lane: split X into its fit on the other regressors and the residual, permute the residual within province."""
    aux = smf.ols(f"{var} ~ " + " + ".join(others + ["lnpop", "C(prov)"]), data=data).fit()
    fv, e0, g = aux.fittedvalues.values, aux.resid.values, data.prov.values; bs = []
    for _ in range(args.draws):
        e = e0.copy()
        for k in np.unique(g):
            j = np.where(g == k)[0]; e[j] = rng.permutation(e0[j])
        bs.append(smf.glm(f, data=data.assign(**{var: fv + e}), family=sm.families.Poisson()).fit(maxiter=100).params[var])
    return float((1 + np.sum(np.abs(bs) >= abs(b_obs))) / (1 + len(bs)))   # observed draw included


def report(var, data, others=()):
    f = "N ~ " + " + ".join([var, *others, "lnpop", "C(prov)"])
    r = fit(f, data); b = r.params[var]; j = r.model.exog_names.index(var)
    p = {"cluster": r.pvalues[var]}
    for c in (50, 100):
        p[f"conley{c}"] = 2 * norm.sf(abs(b / math.sqrt(conley_cov(r, data, c)[j, j])))
    p["RI"] = ri(data, var, list(others), f, b)
    return {"var": var, "effect_%": round(100 * (math.exp(b) - 1), 1), **{k: round(v, 4) for k, v in p.items()}, "p_reported": round(max(p.values()), 4)}


# ---------- 2. single-factor models ----------
print(f"\n2. Single-factor PPML (province FE + ln population), randomisation inference with {args.draws} draws")
rows = [report(v, d) for v in ["phys", "conn", "lights"]]
dn = d.dropna(subset=["need"]).reset_index(drop=True)
rows.append(report("need", dn))
mine = pd.DataFrame(rows); print(mine.to_string(index=False))

# ---------- 3. four factors together ----------
X4 = ["lights", "phys", "conn", "prior_infra"]
r4 = fit("N ~ " + " + ".join(X4 + ["lnpop", "C(prov)"]), d)
j = [r4.model.exog_names.index(v) for v in X4]; b = r4.params.values[j]
W = float(b @ np.linalg.pinv(r4.cov_params().values[np.ix_(j, j)]) @ b)
print("\n3. Four-factor model:", {v: f"{100 * (math.exp(r4.params[v]) - 1):+.0f}% (clustered p {r4.pvalues[v]:.3f})" for v in X4})
print(f"   joint test, clustered Wald chi2(4) = {W:.1f}, p = {chi2.sf(W, 4):.2g}")

# ---------- 4. side by side with the published numbers (kh_published.csv, if present) ----------
pubf = HERE / "kh_published.csv"
if pubf.exists():
    pub = pd.read_csv(pubf).set_index("check")
    print("\n4. Your re-estimate against the published numbers")
    print("   Effects and clustered/Conley p are deterministic and should match to rounding. Randomisation inference is a")
    print(f"   simulation: with {args.draws} draws its p carries a Monte Carlo error of about sqrt(p(1-p)/draws), shown as 'RI_mc_se'.")
    cmp = []
    for r in rows:
        k = f"single:{r['var'] if r['var'] != 'phys' else 'phys'}"
        if k not in pub.index:
            continue
        q = pub.loc[k]; se = math.sqrt(max(q.p_ri * (1 - q.p_ri), 1e-6) / args.draws)
        cmp.append({"var": r["var"], "effect_yours": r["effect_%"], "effect_published": round(q.effect_pct, 1),
                    "p_cluster_yours": r["cluster"], "p_cluster_published": round(q.p_cluster, 4),
                    "p_RI_yours": r["RI"], "p_RI_published": round(q.p_ri, 4), "RI_mc_se": round(se, 4),
                    "p_reported_yours": r["p_reported"], "p_reported_published": round(q.p_reported, 4)})
    print(pd.DataFrame(cmp).to_string(index=False))
    if "joint:four_factors" in pub.index:
        print(f"   joint test: yours chi2 = {W:.1f} | published chi2 = {pub.loc['joint:four_factors', 'statistic']:.1f}")
