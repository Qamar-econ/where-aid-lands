"""Nepal palika (local level) auxiliary analysis with district fixed effects.

Outcome: non-infrastructure development projects with point locations (precision 1-2) inside each palika
(AidData AIMS + earthquake release + IATI non-generic coordinates). District FE absorb census network and deprivation;
within-district variation: Ookla network (tested tiles / tests per pop), travel time to Kathmandu, night lights, RWI.
Outputs: outputs/tables/nepal_palika_models.csv, appended to outputs/results_boost.txt
"""
import json, math, warnings
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.mask import mask
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data/raw/nepal", ROOT / "data/processed/nepal"
LOG = open(ROOT / "outputs/results_boost.txt", "a", encoding="utf-8")
say = lambda *a: (print(" ".join(map(str, a))), LOG.write(" ".join(map(str, a)) + "\n"))
z = lambda s: (s - s.mean()) / s.std()
DEV = {"SOC_education", "SOC_health", "SOC_water", "SOC_other", "GOV", "PROD", "ENV_MULTI"}

P3 = gpd.read_file(RAW / "cod-ab-npl/npl_admin_boundaries.shp/npl_admin3.shp")[["adm3_pcode", "adm3_name", "geometry"]]
D = gpd.read_file(OUT / "districts.gpkg")[["k", "capital", "geometry"]]
cent = P3.copy(); cent["geometry"] = P3.representative_point()
P3 = P3.merge(gpd.sjoin(cent, D, predicate="within")[["adm3_pcode", "k", "capital"]], on="adm3_pcode")
P3["area_km2"] = P3.to_crs(32645).area / 1e6

# ---- point-located projects ----
PR = pd.read_csv(OUT / "oda_projects.csv"); dev = set(PR.loc[PR.cat.isin(DEV) & PR.start_year.between(2000, 2026), "uid"])
pts = []
for rel, pref in [("NepalAIMS_GeocodedResearchRelease_Level1_v1.4.1/NepalAIMS_GeocodedResearchRelease_Level1_v1.4.1", "AIMS-"),
                  ("NepalEarthquake_GeocodedResearchRelease_Level1_v1.0/NepalEarthquake_GeocodedResearchRelease_Level1_v1.0", "EQ-")]:
    L = pd.read_csv(RAW / "aiddata" / rel / "data/locations.csv", encoding="latin1")
    L = L[L.precision_code.isin([1, 2])]
    pts.append(pd.DataFrame({"uid": pref + L.project_id.astype(str), "lon": L.longitude, "lat": L.latitude}))
loc = pd.DataFrame(json.loads((RAW / "iati/dportal_loc_NP.json").read_text(encoding="utf-8")))
loc = loc[loc.location_longitude.notna() & loc.location_precision.isin([1, 2])]
xy = loc.location_longitude.round(2).astype(str) + loc.location_latitude.round(2).astype(str)
loc = loc[xy.map(loc.assign(xy=xy).groupby("xy").aid.nunique()) < 40]
pts.append(pd.DataFrame({"uid": "IATI-" + loc.aid, "lon": loc.location_longitude, "lat": loc.location_latitude}))
pts = pd.concat(pts); pts = pts[pts.uid.isin(dev)]
pg = gpd.sjoin(gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts.lon, pts.lat), crs=4326), P3[["adm3_pcode", "geometry"]], predicate="within")
P3["y"] = P3.adm3_pcode.map(pg.groupby("adm3_pcode").uid.nunique()).fillna(0)

# ---- covariates ----
c = pd.read_csv(OUT / "cells_traveltime.csv")
cg = gpd.sjoin(gpd.GeoDataFrame(c, geometry=gpd.points_from_xy(c.lon, c.lat), crs=4326), P3[["adm3_pcode", "geometry"]], predicate="within")
agg = cg.groupby("adm3_pcode").apply(lambda d: pd.Series({"pop": d["pop"].sum(), "tt_wet": np.average(d.tt_wet, weights=d["pop"] + 1e-6)}), include_groups=False)
P3 = P3.merge(agg, on="adm3_pcode", how="left")


def qk_center(qk):
    x = y = 0; zz = len(qk)
    for i in range(zz):
        m = 1 << (zz - i - 1); v = int(qk[i])
        if v & 1: x |= m
        if v & 2: y |= m
    N = 2 ** zz
    return (x + .5) / N * 360 - 180, math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + .5) / N))))


oo = pd.concat([pd.read_csv(RAW / f"ookla/ookla_mobile_{yy}Q1_NP.csv", dtype={"quadkey": str}) for yy in ["2019", "2025"]])
cc = oo.quadkey.map(qk_center)
og = gpd.sjoin(gpd.GeoDataFrame(oo, geometry=gpd.points_from_xy([p[0] for p in cc], [p[1] for p in cc]), crs=4326), P3[["adm3_pcode", "geometry"]], predicate="within")
P3 = P3.merge(og.groupby("adm3_pcode").agg(tiles=("tests", "size"), tests=("tests", "sum")), on="adm3_pcode", how="left").fillna({"tiles": 0, "tests": 0})
for yy in [2010, 2015]:
    with rasterio.open(RAW / f"ntl/ntl_{yy}_NPL.tif") as s:
        P3[f"ntl{yy}"] = [mask(s, [g], crop=True, filled=True, nodata=0)[0][0].astype(float).sum() for g in P3.geometry]
rw = pd.read_csv(RAW / "rwi/npl_relative_wealth_index.csv")
rg = gpd.sjoin(gpd.GeoDataFrame(rw, geometry=gpd.points_from_xy(rw.longitude, rw.latitude), crs=4326), P3[["adm3_pcode", "geometry"]], predicate="within")
P3["rwi"] = P3.adm3_pcode.map(rg.groupby("adm3_pcode").rwi.mean())

S = P3[(P3.capital == 0) & (P3["pop"] > 100) & P3.rwi.notna()].copy()
S = S[S.groupby("k").y.transform("sum") > 0]                       # districts with no located project add nothing under FE
S["net"] = z(np.log1p(S.tiles / S["pop"] * 1e4) + np.log1p(S.tests / S["pop"] * 1e4))
S["road"] = z(-np.log(S.tt_wet))
S["ntl"] = z(np.log1p(S.ntl2015 / S.area_km2)); S["ntl10"] = z(np.log1p(S.ntl2010 / S.area_km2))
S["poor"] = z(-S.rwi); S["lpop"] = np.log(S["pop"])
say("\n" + "=" * 90 + f"\nNEPAL PALIKA (district FE, SE clustered by district): {len(S)} palikas in {S.k.nunique()} districts; point-located dev projects = {S.y.sum():.0f}")
rows = []
for nm, rhs in [("lights only", "ntl"), ("network only", "net"), ("road only", "road"), ("lights + network", "ntl + net"),
                ("lights + network + road", "ntl + net + road"), ("+ poverty (RWI)", "ntl + net + road + poor"), ("2010 lights + network + road", "ntl10 + net + road")]:
    m = smf.glm(f"y ~ {rhs} + lpop + C(k)", S, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(S.k)[0]})
    vs = [v for v in ["ntl", "ntl10", "net", "road", "poor"] if v in m.params]
    say(f"  {nm:<30} " + "  ".join(f"{v}={m.params[v]:+.3f}(p={m.pvalues[v]:.3f})" for v in vs))
    rows.append({"model": nm, **{f"b_{v}": m.params[v] for v in vs}, **{f"p_{v}": m.pvalues[v] for v in vs}, **{f"se_{v}": m.bse[v] for v in vs}})
S["ly"] = np.log1p(S.y); base = "lpop + C(k)"
b0 = smf.ols(f"ly ~ ntl + {base}", S).fit().params["ntl"]; full = smf.ols(f"ly ~ ntl + net + road + {base}", S).fit()
cn = full.params["net"] * smf.ols(f"net ~ ntl + {base}", S).fit().params["ntl"]
cr = full.params["road"] * smf.ols(f"road ~ ntl + {base}", S).fit().params["ntl"]
say(f"  Gelbach (OLS): lights {b0:+.3f} -> {full.params['ntl']:+.3f}; network explains {cn / b0:.0%}, road {cr / b0:.0%}")
rows.append({"model": "Gelbach", "b_ntl": b0, "share_net": cn / b0, "share_road": cr / b0})
pd.DataFrame(rows).to_csv(ROOT / "outputs/tables/nepal_palika_models.csv", index=False)
LOG.close()
