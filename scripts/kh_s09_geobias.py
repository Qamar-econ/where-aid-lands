"""Geocoding-bias checks motivated by the GODAD authors' own validation (Bomprezzi et al., Appendix B):
errors and omissions come mainly from thin or poorly structured project text, and small localities are hard to geocode.

(1) Well-reported projects only: re-estimate on projects whose title + description is long (>= median, and >= 75th pct),
    where geocoding errors/omissions should be rarer. If the access effect holds, thin-text omissions are unlikely to drive it.
(2) Population-proportional spread: province-only located projects are distributed over the province's districts in proportion
    to population (a realistic middle scenario between the equal split and the extreme worst/best cases).
Constructs: physical access, connectivity index (connectivity-only PC1), night lights.  PPML, province FE, ln pop, clustered SE.
Output: outputs/tables/kh_geobias_checks.csv, outputs/results_kh_geobias.txt
"""
import math, warnings
import numpy as np, pandas as pd, geopandas as gpd, statsmodels.api as sm, statsmodels.formula.api as smf
from pathlib import Path

warnings.filterwarnings("ignore")
SRC = Path(__file__).with_name("kh_s06_ladder.py")
ns = {"__file__": str(SRC)}
exec(SRC.read_text(encoding="utf-8").split("LADDER = [")[0], ns)
DF, L, K, U, outcome, ROOT, RAW = ns["DF"], ns["L"], ns["K"], ns["U"], ns["outcome"], ns["ROOT"], ns["RAW"]
z = lambda s: (s - s.mean()) / s.std()
fitc = lambda f, d: smf.glm(f, data=d, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.prov)[0]}, maxiter=300)

# connectivity-only index (as in kh_s08)
o = pd.concat([pd.read_csv(f) for f in sorted((RAW / "ookla").glob("ookla_mobile_*_KH.csv"))]).drop_duplicates(["tx", "ty"])
n16 = 2 ** 16
o["lon"] = (o.tx + .5) / n16 * 360 - 180
o["lat"] = [math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + .5) / n16)))) for y in o.ty]
og = gpd.sjoin(gpd.GeoDataFrame(o, geometry=gpd.points_from_xy(o.lon, o.lat), crs=4326), U[["unit", "geometry"]], predicate="within")
area = DF.filter(like="area_km2").iloc[:, 0]
cc = pd.DataFrame({"s": z(np.log(DF.mob_mbps_shrunk)), "t": z(np.log1p(DF.mob_tests_pc)),
                   "f": z(np.log1p(DF.unit.map(og.groupby("unit").size()).fillna(0) / area)), "x": z(np.log(DF.fix_mbps_shrunk))})
w_, V_ = np.linalg.eigh(np.cov(cc.T)); pcv = V_[:, -1] * np.sign(V_[:, -1].sum())
DF["conn_index"] = z(cc.values @ pcv)

from kh_common import reclassify_geocodes
cur = reclassify_geocodes(L[L.start.between(2017, 2020) & (L.family != "ict")], DF, ROOT / "data/processed/cambodia")  # spec 9-26
txt = K.drop_duplicates("project_id").set_index("project_id")
tlen = (txt.title.fillna("").astype(str) + " " + txt.description.fillna("").astype(str)).str.len()
cur = cur.assign(tlen=cur.project_id.map(tlen))
plen = cur.drop_duplicates("project_id").tlen
q50, q75 = plen.quantile(0.5), plen.quantile(0.75)

rows = []
def run(tag, d):
    r = {"check": tag, "projects": round(d.N.sum(), 1)}
    for v in ["access", "conn_index", "lights"]:
        m = fitc(f"N ~ {v} + lnpop + C(prov)", d)
        r[v + "_pct"], r[v + "_p"] = 100 * (np.exp(m.params[v]) - 1), m.pvalues[v]
    rows.append(r)

base = outcome(cur)
run("all district-precise projects (baseline)", base)
run(f"well-reported: text >= median ({q50:.0f} chars)", outcome(cur[cur.tlen >= q50]))
run(f"well-reported: text >= 75th pct ({q75:.0f} chars)", outcome(cur[cur.tlen >= q75]))
run(f"thin text: < median", outcome(cur[cur.tlen < q50]))

# population-proportional spread of province-only projects
K["adm1"] = np.where(K.dgroup == "EU/US", K.precision_crs == 4, K.precision_code == 4)
K["fam"] = K["family"]
provs = U[["unit", "geometry"]].merge(DF[["unit", "PARENT"]], on="unit").dissolve("PARENT").reset_index()[["PARENT", "geometry"]]  # U already carries its own PARENT column
a1 = K[K.startyear.between(2017, 2020) & K.adm1 & (K.fam != "ict")]
a1 = gpd.sjoin(gpd.GeoDataFrame(a1, geometry=gpd.points_from_xy(a1.longitude, a1.latitude), crs=4326), provs, predicate="within").drop_duplicates(["project_id", "PARENT"])
a1["w"] = 1 / a1.groupby("project_id").PARENT.transform("nunique")
share = DF.assign(s=DF.pop_2019 / DF.groupby("PARENT").pop_2019.transform("sum"))[["unit", "PARENT", "s"]]
add = a1[["project_id", "PARENT", "w"]].merge(share, on="PARENT")
add = (add.w * add.s).groupby(add.unit).sum()
d = base.copy(); d["N"] = d.N + d.unit.map(add).fillna(0)
run("province-only projects spread by population share", d)
eq = a1[["project_id", "PARENT", "w"]].merge(DF[["unit", "PARENT"]], on="PARENT")
eq["w2"] = eq.w / eq.groupby(["project_id", "PARENT"]).unit.transform("size")
d2 = base.copy(); d2["N"] = d2.N + d2.unit.map(eq.groupby("unit").w2.sum()).fillna(0)
run("province-only projects spread equally (reference)", d2)

R = pd.DataFrame(rows); R.to_csv(ROOT / "outputs/tables/kh_geobias_checks.csv", index=False)
txt_out = f"province-only projects available: {a1.project_id.nunique()} | text length median {q50:.0f}, p75 {q75:.0f}\n" + R.round(3).to_string(index=False)
(ROOT / "outputs/results_kh_geobias.txt").write_text(txt_out, encoding="utf-8")
print(txt_out)
