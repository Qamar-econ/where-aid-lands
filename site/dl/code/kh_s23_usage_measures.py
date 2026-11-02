"""Is the absence of Ookla tests informative? Usage-side measures of digital access (exploratory).

Where nobody runs a speed test, few people may own a smartphone or reach the network. Two usage measures:
  tests per 1,000 people (log)   and   footprint = share of district population in 1 km cells with any tested tile.
For each: reliability between half-years (2019 Q1-Q2 vs 2019 Q3-2020 Q1), the PPML effect alone, and the effect
holding physical access fixed, with the correlation between the two regressors.
Output: outputs/tables/kh_usage_measures.txt
"""
import math
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.mask import mask
from scipy.stats import norm
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "scripts/kh_s04_models.py").read_text(encoding="utf-8")
exec(src[:src.index("\nbase = A[A.level")])
RAW, PROCD = ROOT / "data/raw/cambodia", ROOT / "data/processed/cambodia"
log = []
say = lambda *a: (print(*a), log.append(" ".join(str(x) for x in a)))
zs = lambda s: (s - s.mean()) / s.std()


def xy2ll(x, y, z=16):
    n = 2 ** z
    return (x + .5) / n * 360 - 180, math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + .5) / n))))


U = gpd.read_file(PROCD / "kh_units.gpkg").rename(columns={"GEOLEVEL2": "unit"}); U["unit"] = U.unit.astype(int)
parts = []
for f in sorted((RAW / "ookla").glob("ookla_mobile_*_KH.csv")):
    q = pd.read_csv(f); q["h"] = 1 if f.stem.split("_")[2] in ("2019Q1", "2019Q2") else 2; parts.append(q)
O = pd.concat(parts)
ll = np.array([xy2ll(a, b) for a, b in zip(O.tx, O.ty)]); O["lon"], O["lat"] = ll[:, 0], ll[:, 1]
O = gpd.sjoin(gpd.GeoDataFrame(O, geometry=gpd.points_from_xy(O.lon, O.lat), crs=4326), U[["unit", "geometry"]], predicate="within")
src_pop = rasterio.open(RAW / "worldpop/khm_ppp_2019_1km.tif"); tr = src_pop.transform; W = src_pop.width
r_, c_ = rasterio.transform.rowcol(tr, O.lon.values, O.lat.values); O["cell"] = np.array(r_) * W + np.array(c_)
pop19 = DF.set_index(DF.unit.astype(int)).pop_2019


def measures(o):
    tpc = np.log1p(o.groupby("unit").tests.sum().reindex(U.unit).fillna(0) / pop19.reindex(U.unit) * 1000)
    cells = set(o.cell.unique()); fp = {}
    for u, geom in zip(U.unit, U.to_crs(src_pop.crs).geometry):
        a, t = mask(src_pop, [geom], crop=True, filled=True, nodata=0); a = a[0].astype(float); a[a < 0] = 0
        rr, cc = np.nonzero(a > 0); xs, ys = rasterio.transform.xy(t, rr, cc)
        R2, C2 = rasterio.transform.rowcol(tr, xs, ys)
        fp[u] = a[rr, cc][np.isin(np.array(R2) * W + np.array(C2), list(cells))].sum() / max(a.sum(), 1)
    return tpc, pd.Series(fp)


all_t, all_f = measures(O); h1_t, h1_f = measures(O[O.h == 1]); h2_t, h2_f = measures(O[O.h == 2])
d0 = outcome(A[A.level == "adm2"]); d0["u"] = d0.unit.astype(int)
keep = d0.u
say("== reliability between half-years (districts in the sample)")
for lab, a, b in [("log tests per 1,000 people", h1_t, h2_t), ("footprint share", h1_f, h2_f)]:
    r = np.corrcoef(a.reindex(keep), b.reindex(keep))[0, 1]
    say(f"  {lab:<28s} r = {r:.2f}, pooled reliability {2 * r / (1 + r):.2f}")
say("  (mobile speed, for comparison: r = 0.35, reliability 0.52; kh_s20)")


def fit(f, d):
    r = ppml(f, d); return r


say("\n== effects (PPML, province FE, ln pop); p_max = larger of clustered and Conley 50/100 km")
for lab, s in [("log tests per 1,000 people", all_t), ("footprint share", all_f)]:
    d = d0.assign(xv=zs(d0.u.map(s)))
    r = ppml("N ~ xv + lnpop + C(prov)", d); b = r.params["xv"]
    p1 = max([r.pvalues["xv"]] + [2 * norm.sf(abs(b / conley_se(r, d, "xv", c))) for c in (50, 100)])
    r2 = ppml("N ~ xv + phys + lnpop + C(prov)", d); b2, bp = r2.params["xv"], r2.params["phys"]
    p2 = max([r2.pvalues["xv"]] + [2 * norm.sf(abs(b2 / conley_se(r2, d, "xv", c))) for c in (50, 100)])
    pp = max([r2.pvalues["phys"]] + [2 * norm.sf(abs(bp / conley_se(r2, d, "phys", c))) for c in (50, 100)])
    say(f"  {lab}: alone {100 * (np.exp(b) - 1):+.0f}% (p {p1:.3f}) | with physical access {100 * (np.exp(b2) - 1):+.0f}% (p {p2:.3f}), "
        f"physical access {100 * (np.exp(bp) - 1):+.0f}% (p {pp:.3f}) | corr with physical access {d.xv.corr(d.phys):.2f}")
(ROOT / "outputs/tables/kh_usage_measures.txt").write_text("\n".join(log), encoding="utf-8")
