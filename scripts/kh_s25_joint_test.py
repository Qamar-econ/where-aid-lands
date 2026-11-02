"""Joint test of the four-factor model: do night lights, physical access, digital access and earlier infrastructure
together add explanatory power beyond population and province effects? (H0: all four coefficients are zero.)

Three versions, most conservative reported, as for the single-factor estimates:
  Wald chi2(4) with the province-clustered covariance;  Wald chi2(4) with the Conley 100 km covariance;
  randomisation inference: the four regressors are permuted together, as one block of rows, within province (their mutual
  correlation is kept, their link to the outcome is broken), 1,999 draws; statistic = deviance explained beyond the baseline.
Also: variance inflation factors of the four-factor model.
Output: outputs/tables/kh_joint_test.csv
"""
import numpy as np, pandas as pd, geopandas as gpd, statsmodels.api as sm, statsmodels.formula.api as smf
from scipy.stats import chi2
from pathlib import Path

SRC = Path(__file__).with_name("kh_s06_ladder.py")
ns = {"__file__": str(SRC)}
exec(SRC.read_text(encoding="utf-8").split("LADDER = [")[0], ns)
cur, outcome, ROOT = ns["cur"], ns["outcome"], ns["ROOT"]
RNG = np.random.default_rng(20261102)
X = ["lights", "access", "conn", "prior_infra"]
d = outcome(cur).reset_index(drop=True)

U = gpd.read_file(ROOT / "data/processed/cambodia/kh_units.gpkg")
cen = U.to_crs(32648).geometry.centroid
XY = pd.DataFrame({"x": cen.x.values / 1000, "y": cen.y.values / 1000}, index=U.GEOLEVEL2.astype(float).astype(int))
u = d.unit.astype(float).astype(int); d["x_km"], d["y_km"] = u.map(XY.x).values, u.map(XY.y).values

f_full = "N ~ " + " + ".join(X + ["lnpop", "C(prov)"]); f0 = "N ~ lnpop + C(prov)"
r = smf.glm(f_full, data=d, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.prov)[0]}, maxiter=300)
r0 = smf.glm(f0, data=d, family=sm.families.Poisson()).fit(maxiter=300)
j = [r.model.exog_names.index(v) for v in X]; b = r.params.values[j]
wald = lambda V: float(b @ np.linalg.pinv(V) @ b)
W_cl = wald(r.cov_params().values[np.ix_(j, j)])

# Conley 100 km covariance for the four coefficients
Xm = r.model.exog; mu = r.fittedvalues.values; S = Xm * (r.model.endog - mu)[:, None]
Dm = np.hypot(d.x_km.values[:, None] - d.x_km.values[None, :], d.y_km.values[:, None] - d.y_km.values[None, :])
K = np.clip(1 - Dm / 100, 0, None); bread = np.linalg.pinv(Xm.T @ (Xm * mu[:, None]))
W_co = wald((bread @ (S.T @ K @ S) @ bread)[np.ix_(j, j)])

dev_gain = r0.deviance - r.deviance
null_dev = smf.glm("N ~ 1", data=d, family=sm.families.Poisson()).fit().deviance
g = d.prov.values; gains = []
for _ in range(1999):
    perm = np.arange(len(d))
    for k in np.unique(g):
        ix = np.where(g == k)[0]; perm[ix] = RNG.permutation(ix)
    dp = d.copy(); dp[X] = d[X].values[perm]
    try:
        gains.append(r0.deviance - smf.glm(f_full, data=dp, family=sm.families.Poisson()).fit(maxiter=100).deviance)
    except Exception:
        pass
p_ri = (1 + np.sum(np.array(gains) >= dev_gain)) / (1 + len(gains))
p = {"p_wald_cluster": chi2.sf(W_cl, 4), "p_wald_conley100": chi2.sf(W_co, 4), "p_ri": p_ri}
vif = np.diag(np.linalg.inv(np.corrcoef(d[X + ["lnpop"]].values.T)))[:4]
R = pd.DataFrame([{"test": "four factors jointly zero", "wald_cluster": W_cl, "wald_conley100": W_co, **p, "p_conservative": max(p.values()),
                   "pseudoR2_base": 1 - r0.deviance / null_dev, "pseudoR2_full": 1 - r.deviance / null_dev, "ri_draws": len(gains),
                   **{f"vif_{v}": x for v, x in zip(X, vif)}}])
R.to_csv(ROOT / "outputs/tables/kh_joint_test.csv", index=False)
print(R.T.round(4).to_string())
