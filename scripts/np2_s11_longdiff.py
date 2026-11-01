"""Long-difference design (설계_장기차분_2011_2021.md): build the district x period panel (P0 2006-08, P1 2011-13, P2 2021-23 starts),
IATI development projects re-geocoded with geocoder v2 in all periods; null-model power simulation.
The confirmatory beta is estimated only with --estimate (after the design is frozen).
Outputs: data/processed/nepal/np2_longdiff_projects.csv, np2_longdiff_panel.csv, outputs/results_np2_longdiff_power.txt
"""
import json, re, sys, warnings
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.mask import mask
import statsmodels.api as sm, statsmodels.formula.api as smf
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from np2_geo import geocode_text, best_by_district  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ESTIMATE = "--estimate" in sys.argv
WIN = {"P0": (2006, 2008), "P1": (2011, 2013), "P2": (2021, 2023)}
DEV = {"SOC_education", "SOC_health", "SOC_water", "SOC_other", "GOV", "PROD", "ENV_MULTI"}
PLACEBO = {"HUMAN": "hum", "ACCESS_energy": "infra", "ACCESS_transport": "infra"}
CAPITAL = {"kathmandu", "lalitpur", "bhaktapur"}
QUAKE14 = {"gorkha", "dhading", "rasuwa", "nuwakot", "sindhupalchok", "dolakha", "ramechhap", "kavre", "kathmandu", "lalitpur", "bhaktapur",
           "makwanpur", "okhaldhunga", "sindhuli"}

# ---------------- projects (IATI only, same geocoder in every period) ----------------
P = pd.read_csv(ROOT / "data/processed/nepal/oda_projects.csv")
P = P[(P.source == "IATI") & (P["cat"].isin(DEV | set(PLACEBO)))].copy()
P["period"] = None
for k, (a, b) in WIN.items():
    P.loc[P.start_year.between(a, b), "period"] = k
P = P[P.period.notna() & (P.ict != True)]
ft = pd.read_csv(ROOT / "data/interim/iati_fulltext_NP.csv").drop_duplicates("aid").set_index("aid").fillna("")
loc = pd.DataFrame(json.loads((ROOT / "data/raw/nepal/iati/dportal_loc_NP.json").read_text(encoding="utf-8")))
loc = loc[loc.location_longitude.notna()]
xy = loc.location_longitude.round(2).astype(str) + loc.location_latitude.round(2).astype(str)
loc = loc[xy.map(loc.assign(xy=xy).groupby("xy").aid.nunique()) < 40]
Dg = gpd.read_file(ROOT / "data/processed/nepal/districts.gpkg")
lg = gpd.GeoDataFrame(loc, geometry=gpd.points_from_xy(loc.location_longitude, loc.location_latitude), crs=4326)
lg = gpd.sjoin(lg, Dg[["k", "geometry"]], predicate="within")
COORD = lg.groupby("aid").k.apply(set).to_dict()

rows = []
for r in P.itertuples():
    aid = r.uid[5:]
    core = f"{r.title} {r.desc if isinstance(r.desc, str) else ''}"
    hits = geocode_text(core + " " + ft.text.get(aid, ""))
    cd = {h["district"] for h in geocode_text(core)}
    hits = [h for h in hits if h["district"] not in CAPITAL or h["district"] in cd]  # V3b
    ds = {h["district"] for h in best_by_district(hits) if h["precision"] <= 4}
    cds = COORD.get(aid, set())
    if cds and not cds <= CAPITAL:  # V3: capital-only coordinates are HQ defaults
        ds |= cds
    for d in ds:
        rows.append({"uid": r.uid, "k": d, "w": 1 / len(ds), "n_districts": len(ds)})
A = pd.DataFrame(rows).merge(P[["uid", "period", "cat", "usd", "donor", "start_year"]], on="uid")
A["grp"] = A["cat"].map(PLACEBO).fillna("dev")
A["size"] = pd.cut(A.usd, [-1, 1e6, 1e7, np.inf], labels=["small", "mid", "large"]).astype(str)
A.to_csv(ROOT / "data/processed/nepal/np2_longdiff_projects.csv", index=False, encoding="utf-8-sig")

# ---------------- covariates ----------------
D = pd.read_csv(ROOT / "data/processed/nepal/districts.csv")
H = pd.read_csv(ROOT / "data/processed/nepal/census2021_district_hh.csv")
ntl = {}
for y in [2005, 2010, 2020]:
    with rasterio.open(ROOT / f"data/raw/nepal/ntl/ntl_{y}_NPL.tif") as src:
        g = Dg.to_crs(src.crs)
        ntl[y] = {k: float(np.nansum(np.where(a := mask(src, [geom], crop=True)[0][0], a, 0)))
                  for k, geom in zip(g.k, g.geometry)}
base = D[["k", "province", "hh2011", "internet_pct", "mobile_pct", "computer_pct"]].merge(
    H[["k", "hh", "internet_pct21", "mobile_ord_pct21", "mobile_smart_pct21", "computer_pct21"]], on="k")
pan = []
for t in ["P0", "P1", "P2"]:
    b = base[["k", "province"]].copy(); b["period"] = t
    if t == "P2":
        b["Int"] = base.internet_pct21 / 10; b["Mob"] = np.minimum(100, base.mobile_ord_pct21 + base.mobile_smart_pct21) / 10
        b["lnHH"] = np.log(base.hh); b["lnNTL"] = np.log1p(base.k.map(ntl[2020])); b["Comp"] = base.computer_pct21 / 10
    else:
        b["Int"] = base.internet_pct / 10; b["Mob"] = base.mobile_pct / 10; b["lnHH"] = np.log(base.hh2011)
        b["lnNTL"] = np.log1p(base.k.map(ntl[2010 if t == "P1" else 2005])); b["Comp"] = base.computer_pct / 10
    b["quake_post"] = (b.k.isin(QUAKE14) & (t == "P2")).astype(int)
    pan.append(b)
pan = pd.concat(pan)
for g in ["dev", "hum", "infra"]:
    y = A[A.grp == g].groupby(["k", "period"]).w.sum().rename(f"N_{g}")
    pan = pan.merge(y.reset_index(), on=["k", "period"], how="left")
for s in ["small", "mid", "large"]:
    y = A[(A.grp == "dev") & (A["size"] == s)].groupby(["k", "period"]).w.sum().rename(f"N_{s}")
    pan = pan.merge(y.reset_index(), on=["k", "period"], how="left")
pan = pan.fillna({c: 0 for c in pan.columns if c.startswith("N_")})
pan["dInt"] = pan.k.map((base.set_index("k").internet_pct21 - base.set_index("k").internet_pct) / 10)
pan["pt"] = pan.province.astype(str) + "_" + pan.period
pan.to_csv(ROOT / "data/processed/nepal/np2_longdiff_panel.csv", index=False, encoding="utf-8-sig")

# ---------------- descriptives + null-model power ----------------
out = []
w = out.append
w("located projects by period x group: " + str(A.drop_duplicates("uid").groupby(["period", "grp"]).size().to_dict()))
w("all projects in scope by period: " + str(P.groupby("period").size().to_dict()))
w("located share by period (dev): " + str((A[A.grp == "dev"].drop_duplicates("uid").groupby("period").size()
                                           / P[P["cat"].isin(DEV)].groupby("period").size()).round(3).to_dict()))
q = pan[pan.period.isin(["P1", "P2"])]
q = q[q.groupby("k").N_dev.transform("sum") > 0]  # all-zero districts carry no information under district FE
w("dev projects per district: " + str(q.groupby("period").N_dev.agg(["mean", "std"]).round(2).to_dict()))
# located-share by district type (does detection change differently in remote districts?)
rem = base.set_index("k").internet_pct < base.internet_pct.median()
w("located dev projects, remote vs connected (by 2011 internet median): " +
  str(A[A.grp == "dev"].assign(remote=A.k.map(rem)).groupby(["period", "remote"]).w.sum().round(1).to_dict()))

# power: the SE of the main coefficient does not depend on the outcome-covariate association's sign, so the SE-based MDE is
# reported before the point estimate is looked at (a simulation was dropped: simulated all-zero districts break the FE fit)
se_obs = smf.glm("N_dev ~ Int + lnHH + C(k) + C(pt)", data=q, family=sm.families.Poisson()).fit(
    cov_type="cluster", cov_kwds={"groups": pd.factorize(q.k)[0]}).bse["Int"]
w(f"observed-data SE-based MDE (2.8 x SE): {100 * (np.exp(2.8 * se_obs) - 1):.0f}% per 10pp internet "
  f"(SD of 2011->21 change = {q.dInt.std():.2f} x 10pp)")

if ESTIMATE:
    def fit(tag, f, d, term="Int"):
        m = smf.glm(f, data=d, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.k)[0]}, maxiter=200)
        w(f"{tag:<40} {term} {m.params[term]:+.3f} (se {m.bse[term]:.3f}, p {m.pvalues[term]:.3f}) = {100 * (np.exp(m.params[term]) - 1):+.1f}% per 10pp")
    fit("MAIN", "N_dev ~ Int + lnHH + C(k) + C(pt)", q)
    fit("S1 mobile", "N_dev ~ Mob + lnHH + C(k) + C(pt)", q, "Mob")
    fit("S3 + NTL + quake", "N_dev ~ Int + lnHH + lnNTL + quake_post + C(k) + C(pt)", q)
    for g in ["hum", "infra"]:
        fit(f"placebo {g}", f"N_{g} ~ Int + lnHH + C(k) + C(pt)", q)
    for s in ["small", "mid", "large"]:
        fit(f"size {s}", f"N_{s} ~ Int + lnHH + C(k) + C(pt)", q)
    pre = pan[pan.period.isin(["P0", "P1"])].copy()
    pre["fut"] = pre.dInt * (pre.period == "P1")
    fit("PRE-TREND (future dInt on P0->P1)", "N_dev ~ fut + lnHH + C(k) + C(pt)", pre, "fut")
with open(ROOT / "outputs/results_np2_longdiff_power.txt", "w", encoding="utf-8") as fh:
    for s in out:
        print(s); fh.write(s + "\n")
