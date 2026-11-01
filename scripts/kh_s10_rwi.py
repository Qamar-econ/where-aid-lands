"""Need (relative poverty) for Cambodia: Meta Relative Wealth Index (Chi et al. 2022), 모형명세서 §9-4.

RWI points (~2.4 km) are weighted by WorldPop 2019 population at the point and averaged per GEO2_KH district.
need = -RWI (standardised; higher = poorer).  Models as before: PPML, province FE, ln pop, province-clustered SE;
plus correlations with the other constructs, a race model with all five factors, and a Shapley decomposition of explained
deviance over the five factors. Sector split (OECD categories as used throughout).
Outputs: data/processed/cambodia/kh_rwi.csv, outputs/tables/kh_need.csv, kh_shapley5.csv, outputs/results_kh_need.txt
"""
import io, itertools, warnings
import numpy as np, pandas as pd, geopandas as gpd, rasterio, requests, statsmodels.api as sm, statsmodels.formula.api as smf
from pathlib import Path

warnings.filterwarnings("ignore")
SRC = Path(__file__).with_name("kh_s06_ladder.py")
ns = {"__file__": str(SRC)}
exec(SRC.read_text(encoding="utf-8").split("LADDER = [")[0], ns)
DF, L, U, outcome, ROOT, RAW = ns["DF"], ns["L"], ns["U"], ns["outcome"], ns["ROOT"], ns["RAW"]
z = lambda s: (s - s.mean()) / s.std()
fitc = lambda f, d: smf.glm(f, data=d, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.prov)[0]}, maxiter=300)

p = RAW / "rwi" / "khm_relative_wealth_index.csv"
if not p.exists():
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(requests.get("https://data.humdata.org/dataset/76f2a2ea-ba50-40f5-b79c-db95d668b843/resource/73a569e7-2f22-4a16-917d-ecbabd693e4a/download/khm_relative_wealth_index.csv",
                               timeout=300).content)
R = pd.read_csv(p)
with rasterio.open(RAW / "worldpop/khm_ppp_2019_1km.tif") as s:
    R["pop"] = [max(v[0], 0) for v in s.sample(zip(R.longitude, R.latitude))]
g = gpd.sjoin(gpd.GeoDataFrame(R, geometry=gpd.points_from_xy(R.longitude, R.latitude), crs=4326), U[["unit", "geometry"]], predicate="within")
rw = g.groupby("unit").apply(lambda d: np.average(d.rwi, weights=d["pop"] + 1e-6), include_groups=False).rename("rwi")
rw.to_frame().to_csv(ROOT / "data/processed/cambodia/kh_rwi.csv")
DF["rwi"] = DF.unit.map(rw)
DF["need"] = z(-DF.rwi)
cur = L[L.start.between(2017, 2020) & (L.family != "ict")]
d = outcome(cur)
out = [f"RWI points {len(R)} | districts with RWI {d.need.notna().sum()} of {len(d)}",
       "correlations:\n" + d[["need", "access", "conn", "lights", "prior_infra", "lnpop"]].corr().round(2).to_string()]
rows = []
def run(tag, dd, f="N ~ need + lnpop + C(prov)", var="need"):
    m = fitc(f, dd.dropna(subset=["need"]))
    rows.append({"spec": tag, "need_pct": 100 * (np.exp(m.params[var]) - 1), "p": m.pvalues[var], "projects": round(dd.N.sum(), 1)})
run("need alone (count)", d)
run("need + physical access", d, "N ~ need + access + lnpop + C(prov)")
run("need + night lights", d, "N ~ need + lights + lnpop + C(prov)")
run("need, excluding Phnom Penh core", d[~d.capital])
run("need, amount track ($m)", d.assign(N=d.M))
for fam, lab in [("social", "Social"), ("infra", "Infrastructure (transport/energy)"), ("production", "Production")]:
    run(f"need, sector: {lab}", outcome(cur[cur.family == fam]))
NE = pd.DataFrame(rows); NE.to_csv(ROOT / "outputs/tables/kh_need.csv", index=False)
out.append("NEED (= -RWI; % change in projects per SD more poverty):\n" + NE.round(3).to_string(index=False))

# race + Shapley with five factors
X = ["need", "lights", "access", "conn", "prior_infra"]
dd = d.dropna(subset=["need"])
m = fitc("N ~ " + " + ".join(X) + " + lnpop + C(prov)", dd)
vif = np.diag(np.linalg.inv(np.corrcoef(dd[X + ["lnpop"]].values.T)))
out.append("RACE (all five): " + ", ".join(f"{x} {100 * (np.exp(m.params[x]) - 1):+.0f}% (p {m.pvalues[x]:.3f})" for x in X) +
           f" | VIF {dict(zip(X + ['lnpop'], vif.round(2)))}")
dev = lambda xs: smf.glm("N ~ " + " + ".join(xs + ["lnpop", "C(prov)"]), data=dd, family=sm.families.Poisson()).fit(maxiter=300).deviance
D0, Dn = dev([]), smf.glm("N ~ 1", data=dd, family=sm.families.Poisson()).fit().deviance
cache = {}
v = lambda s: cache.setdefault(tuple(sorted(s)), (D0 - dev(list(sorted(s)))) / Dn)
phi = {x: 0.0 for x in X}; perms = list(itertools.permutations(X))
for pm in perms:
    s = []
    for x in pm:
        phi[x] += (v(s + [x]) - v(s)) / len(perms); s.append(x)
tot = v(X)
SH = pd.DataFrame([{"factor": x, "share_pct": 100 * phi[x] / tot, "pp": 100 * phi[x]} for x in X]).sort_values("share_pct", ascending=False)
SH.to_csv(ROOT / "outputs/tables/kh_shapley5.csv", index=False)
out.append(f"SHAPLEY (five factors explain {100 * tot:.1f} pp beyond population + province; full pseudo-R2 {1 - D0 / Dn + tot:.3f}):\n" + SH.round(1).to_string(index=False))
(ROOT / "outputs/results_kh_need.txt").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
