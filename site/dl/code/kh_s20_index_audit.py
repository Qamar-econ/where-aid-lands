"""Methodological audit of the two access measures (exploratory; the main specification is unchanged).

Physical access
  P1  surface-free index: speeds by road class only (kh_s03 KH_SURFACE=class), alone and with the kh_s19 corrections
  P2  components filled with the median (missing values) and the PC1 loadings of each version
  P3  influence of Phnom Penh: the effect without the capital's province
Digital access (Ookla mobile, 2019 Q1-2020 Q1)
  D1  reliability: district speed in the first half (2019 Q1-Q2) against the second (2019 Q3-2020 Q1)
  D2  aggregation: test-weighted (main) vs tile-equal vs population-weighted speed, and the share of tests in a
      district's single busiest tile (how far one town drives the district value)
  D3  test footprint: share of district population in 1 km cells that contain any tested tile
  D4  timing: speed measured from 2019 Q1 alone
  D5  correlations with the other constructs
Output: outputs/tables/kh_index_audit.txt
"""
import math
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from scipy.stats import norm
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "scripts/kh_s04_models.py").read_text(encoding="utf-8")
exec(src[:src.index("\nbase = A[A.level")])          # DF, A, outcome(), ppml(), conley_se(), T
RAW, PROCD = ROOT / "data/raw/cambodia", ROOT / "data/processed/cambodia"
log = []
say = lambda *a: (print(*a), log.append(" ".join(str(x) for x in a)))
zs = lambda s: (s - s.mean()) / s.std()
d0 = outcome(A[A.level == "adm2"])


def eff(series, d=None, label=""):
    d = (d0 if d is None else d).copy()
    series = series.copy(); series.index = series.index.astype(d.unit.dtype)      # gpkg codes come back as strings
    d["xv"] = zs(d.unit.map(series))
    d = d[d.xv.notna()]
    r = ppml("N ~ xv + lnpop + C(prov)", d); b = r.params["xv"]
    ps = [r.pvalues["xv"]] + [2 * norm.sf(abs(b / conley_se(r, d, "xv", c))) for c in (50, 100)]
    say(f"  {label:<58s} {100 * (np.exp(b) - 1):+7.0f}%  p_max {max(ps):.3f}  n {len(d)}")


main = pd.read_csv(PROCD / "kh_access.csv").set_index("k")
say("== P1. physical access without surface tags")
for tag, lab in [("", "main (surface tags, untagged minor roads unpaved)"), ("class", "class speeds only, 9 components"),
                 ("class_fix", "class speeds + provincial-capital towns + 185 m grade")]:
    a = main if not tag else pd.read_csv(PROCD / f"kh_access_{tag}.csv").set_index("k")
    say(f"  corr with main {a.access_pc1.corr(main.access_pc1):.3f}"); eff(a.access_pc1, label=lab)
    pca = pd.read_csv(T / ("kh_access_pca.csv" if not tag else f"kh_access_pca_{tag}.csv"))
    say("   loadings:", ", ".join(f"{c}={l:+.2f}" for c, l in zip(pca.component, pca.loading)), "| PC1 share", pca.pc1_var_share.iloc[0])

say("\n== P2. missing component values filled with the median")
comp = pd.DataFrame({"tt_town_dry": main.tt_town_dry, "tt_hq_dry": main.tt_hq_dry, "tt_hosp_dry": main.tt_hosp_dry, "walk_min": main.walk_min,
                     "road_density": main.road_density, "paved_share": main.paved_share, "steep_share": main.steep_share})
say(" ", comp.isna().sum().to_dict(), "| districts in index", len(main))

say("\n== P3. influence of Phnom Penh")
ppcode = DF.loc[DF.pop_2019.idxmax(), "prov"]
eff(main.access_pc1, d0[d0.prov != ppcode], f"physical access without province {ppcode} (largest district's province)")

# ---------------- digital access ----------------
def xy2ll(x, y, z=16):
    n = 2 ** z
    return (x + .5) / n * 360 - 180, math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + .5) / n))))


U = gpd.read_file(PROCD / "kh_units.gpkg").rename(columns={"GEOLEVEL2": "unit"})
parts = []
for f in sorted((RAW / "ookla").glob("ookla_mobile_*_KH.csv")):
    q = pd.read_csv(f); q["q"] = f.stem.split("_")[2]; parts.append(q)
O = pd.concat(parts)
ll = np.array([xy2ll(a, b) for a, b in zip(O.tx, O.ty)]); O["lon"], O["lat"] = ll[:, 0], ll[:, 1]
O = gpd.sjoin(gpd.GeoDataFrame(O, geometry=gpd.points_from_xy(O.lon, O.lat), crs=4326), U[["unit", "geometry"]], predicate="within").drop(columns="index_right")
with rasterio.open(RAW / "worldpop/khm_ppp_2019_1km.tif") as s:
    pop = s.read(1).astype(float); pop[pop < 0] = 0; tr = s.transform
    r_, c_ = rasterio.transform.rowcol(tr, O.lon.values, O.lat.values)
O["cell"] = np.array(r_) * pop.shape[1] + np.array(c_)
O["cellpop"] = pop.ravel()[O.cell.values]

lg = lambda s: np.log(s)
say("\n== D1. reliability of district mobile speed")
h1 = O[O.q.isin(["2019Q1", "2019Q2"])].groupby("unit").apply(lambda d: np.average(d.avg_d_kbps, weights=d.tests), include_groups=False)
h2 = O[~O.q.isin(["2019Q1", "2019Q2"])].groupby("unit").apply(lambda d: np.average(d.avg_d_kbps, weights=d.tests), include_groups=False)
j = pd.concat([h1, h2], axis=1, keys=["h1", "h2"]).dropna()
rr = np.corrcoef(lg(j.h1), lg(j.h2))[0, 1]
say(f"  corr(log speed, first half vs second half) = {rr:.2f} over {len(j)} districts; Spearman-Brown reliability of the pooled measure = {2 * rr / (1 + rr):.2f}")

say("\n== D2. how the district value is aggregated")
g = O.groupby("unit")
tile = O.groupby(["unit", "quadkey"]).apply(lambda d: pd.Series({"kbps": np.average(d.avg_d_kbps, weights=d.tests), "tests": d.tests.sum(), "cellpop": d.cellpop.iloc[0]}), include_groups=False).reset_index()
tw = g.apply(lambda d: np.average(d.avg_d_kbps, weights=d.tests), include_groups=False)
eq = tile.groupby("unit").kbps.apply(lambda s: np.exp(np.log(s).mean()))
pw = tile.groupby("unit").apply(lambda d: np.average(d.kbps, weights=d.cellpop + 1e-6), include_groups=False)
top = tile.groupby("unit").apply(lambda d: d.tests.max() / d.tests.sum(), include_groups=False)
say(f"  share of a district's tests in its busiest tile: median {top.median():.2f}, p75 {top.quantile(.75):.2f}, max {top.max():.2f}")
say(f"  corr(log test-weighted, log tile-equal) {np.corrcoef(lg(tw), lg(eq.loc[tw.index]))[0, 1]:.2f} | with population-weighted {np.corrcoef(lg(tw), lg(pw.loc[tw.index]))[0, 1]:.2f}")
mainconn = DF.set_index("unit").conn
eff(mainconn, label="main: log test-weighted speed, shrunk")
eff(lg(tw), label="log test-weighted speed, not shrunk")
eff(lg(eq), label="log tile-equal (geometric mean over tiles)")
eff(lg(pw), label="log population-weighted over tested tiles")

say("\n== D3. test footprint (people living in a 1 km cell with any tested tile)")
cells = set(O.cell.unique())
with rasterio.open(RAW / "worldpop/khm_ppp_2019_1km.tif") as s:
    from rasterio.mask import mask
    fp = {}
    for u, geom in zip(U.unit, U.to_crs(s.crs).geometry):
        a, t = mask(s, [geom], crop=True, filled=True, nodata=0); a = a[0].astype(float); a[a < 0] = 0
        rr_, cc_ = np.nonzero(a > 0)
        xs, ys = rasterio.transform.xy(t, rr_, cc_)
        R2, C2 = rasterio.transform.rowcol(tr, xs, ys)
        ids = np.array(R2) * pop.shape[1] + np.array(C2)
        fp[u] = a[rr_, cc_][np.isin(ids, list(cells))].sum() / max(a.sum(), 1)
fp = pd.Series(fp); fp.index = fp.index.astype(main.index.dtype)
say(f"  footprint share: median {fp.median():.2f}, min {fp.min():.2f}, max {fp.max():.2f}; corr with physical access {fp.corr(main.access_pc1.reindex(fp.index)):.2f}")
eff(fp, label="test footprint share")

say("\n== D4. timing: 2019 Q1 only")
q1 = O[O.q == "2019Q1"].groupby("unit").apply(lambda d: np.average(d.avg_d_kbps, weights=d.tests), include_groups=False)
eff(lg(q1), label=f"log speed 2019 Q1 only ({len(q1)} districts with tests)")

say("\n== D5. correlations (districts in the main sample)")
C = d0[["phys", "conn", "conn_tests", "conn_fixed", "ntl", "lnpop"]].copy()
rw = pd.read_csv(PROCD / "kh_rwi.csv").set_index("unit").rwi
C["poverty"] = -d0.unit.map(rw).values
say(C.corr().round(2).to_string())
(ROOT / "outputs/tables/kh_index_audit.txt").write_text("\n".join(log), encoding="utf-8")
