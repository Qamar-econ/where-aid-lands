"""Mobile speed corrected for measurement error with a split-sample instrument (exploratory).

District speed from Ookla is noisy (half-year reliability 0.52, kh_s20). With two independent halves of the tests,
one half instruments the other: noise in the two halves is unrelated, so the first half predicts only the part of the
second half that reflects the district's true speed. PPML with a control function: the first stage regresses
log speed (second half, 2019 Q3-2020 Q1) on log speed (first half, 2019 Q1-Q2), ln population and province effects;
its residual enters the PPML. Inference by a province-cluster bootstrap of both stages (999 draws).
Output: outputs/tables/kh_speed_iv.csv
"""
import math
import numpy as np, pandas as pd, geopandas as gpd, statsmodels.api as sm, statsmodels.formula.api as smf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "scripts/kh_s04_models.py").read_text(encoding="utf-8")
exec(src[:src.index("\nbase = A[A.level")])
RAW, PROCD = ROOT / "data/raw/cambodia", ROOT / "data/processed/cambodia"


def xy2ll(x, y, z=16):
    n = 2 ** z
    return (x + .5) / n * 360 - 180, math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + .5) / n))))


U = gpd.read_file(PROCD / "kh_units.gpkg").rename(columns={"GEOLEVEL2": "unit"})
parts = []
for f in sorted((RAW / "ookla").glob("ookla_mobile_*_KH.csv")):
    q = pd.read_csv(f); q["q"] = f.stem.split("_")[2]; parts.append(q)
O = pd.concat(parts)
ll = np.array([xy2ll(a, b) for a, b in zip(O.tx, O.ty)])
O = gpd.sjoin(gpd.GeoDataFrame(O, geometry=gpd.points_from_xy(ll[:, 0], ll[:, 1]), crs=4326), U[["unit", "geometry"]], predicate="within")
half = lambda m: O[m].groupby("unit").apply(lambda d: np.log(np.average(d.avg_d_kbps, weights=d.tests)), include_groups=False)
H = pd.concat([half(O.q.isin(["2019Q1", "2019Q2"])), half(~O.q.isin(["2019Q1", "2019Q2"]))], axis=1, keys=["s1", "s2"])
H.index = H.index.astype(int)
d0 = outcome(A[A.level == "adm2"])
d = d0.assign(s1=d0.unit.astype(int).map(H.s1), s2=d0.unit.astype(int).map(H.s2)).dropna(subset=["s1", "s2"]).reset_index(drop=True)
sd = d.s2.std(); d["s1z"], d["s2z"] = (d.s1 - d.s1.mean()) / sd, (d.s2 - d.s2.mean()) / sd   # one SD of second-half speed


def fit(dd):
    fs = smf.ols("s2z ~ s1z + lnpop + C(prov)", data=dd).fit()
    dd = dd.assign(v=fs.resid)
    naive = smf.glm("N ~ s2z + lnpop + C(prov)", data=dd, family=sm.families.Poisson()).fit(maxiter=300).params["s2z"]
    cf = smf.glm("N ~ s2z + v + lnpop + C(prov)", data=dd, family=sm.families.Poisson()).fit(maxiter=300).params["s2z"]
    return naive, cf, fs.params["s1z"], fs.tvalues["s1z"] ** 2


b_naive, b_cf, pi, F = fit(d)
rng = np.random.default_rng(20261102); provs = d.prov.unique(); bs = []
for _ in range(999):
    take = rng.choice(provs, len(provs), replace=True)
    dd = pd.concat([d[d.prov == p].assign(prov=f"{p}_{i}") for i, p in enumerate(take)], ignore_index=True)
    try:
        bs.append(fit(dd)[:2])
    except Exception:
        pass
bs = np.array(bs)
pct = lambda b: 100 * (np.exp(b) - 1)
rows = []
for k, (lab, b) in enumerate([("naive: second-half speed", b_naive), ("split-sample IV (control function)", b_cf)]):
    lo, hi = np.percentile(bs[:, k], [2.5, 97.5])
    p = 2 * min(np.mean(bs[:, k] <= 0), np.mean(bs[:, k] >= 0))
    rows.append({"model": lab, "pct": pct(b), "ci_lo": pct(lo), "ci_hi": pct(hi), "p_boot": p, "first_stage_coef": pi, "first_stage_F": F, "n": len(d), "draws": len(bs)})
R = pd.DataFrame(rows); R.to_csv(T / "kh_speed_iv.csv", index=False)
print(R.round(3).to_string(index=False))
