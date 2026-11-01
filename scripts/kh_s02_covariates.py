"""Cambodia district covariates (IPUMS GEO2_KH units): population (WorldPop), night lights (harmonized DMSP-VIIRS), Ookla.

pop_{y}       WorldPop 1 km UN-adjusted sum (2008, 2015, 2019)
ntl_{y}       sum of harmonized night-light DN (2008, 2015, 2019); ntl_pc = ntl / pop (lights per person)
Ookla (pooled 2019 Q1-Q4 + 2020 Q1, each tile-quarter weighted by its tests):
  mob_tests_pc, fix_tests_pc  tests per 1,000 people
  mob_mbps, fix_mbps          test-weighted mean download (Mbps), shrunk toward the province mean with a 30-test prior
                              (empirical-Bayes style: (n*x + 30*x_prov) / (n + 30)) to tame sparse districts
Lake unit (Tonle Sap) and units with < 1,000 people are flagged (excluded in the analysis).
Output: data/processed/cambodia/kh_covariates.csv
"""
import math
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.mask import mask
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data/raw/cambodia", ROOT / "data/processed/cambodia"
U = gpd.read_file(OUT / "kh_units.gpkg").rename(columns={"GEOLEVEL2": "unit"})
C = U[["unit", "ADMIN_NAME", "PARENT"]].copy()
C["area_km2"] = U.to_crs(32648).area / 1e6


def zsum(path):
    with rasterio.open(path) as s:
        g = U.to_crs(s.crs)
        out = []
        for geom in g.geometry:
            a = mask(s, [geom], crop=True, filled=True, nodata=0)[0][0].astype(float)
            a[a < 0] = 0; out.append(a.sum())
    return out


for y in [2008, 2015, 2019]:
    C[f"pop_{y}"] = zsum(RAW / f"worldpop/khm_ppp_{y}_1km.tif")
    p = RAW / f"ntl/ntl_{y}_KHM.tif"
    if p.exists():
        C[f"ntl_{y}"] = zsum(p)


def xy2ll(x, y, z=16):
    n = 2 ** z
    return (x + .5) / n * 360 - 180, math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + .5) / n))))


for typ, ab in [("mobile", "mob"), ("fixed", "fix")]:
    parts = [pd.read_csv(f) for f in sorted((RAW / "ookla").glob(f"ookla_{typ}_*_KH.csv"))]
    if not parts:
        continue
    o = pd.concat(parts)
    ll = [xy2ll(a, b) for a, b in zip(o.tx, o.ty)]
    g = gpd.sjoin(gpd.GeoDataFrame(o, geometry=gpd.points_from_xy([p[0] for p in ll], [p[1] for p in ll]), crs=4326), U[["unit", "geometry"]], predicate="within")
    a = g.groupby("unit").apply(lambda d: pd.Series({"tests": d.tests.sum(), "mbps": np.average(d.avg_d_kbps, weights=d.tests) / 1000}), include_groups=False)
    C = C.merge(a.add_prefix(ab + "_"), left_on="unit", right_index=True, how="left")
    C[ab + "_tests"] = C[ab + "_tests"].fillna(0)
    prov = C.groupby("PARENT").apply(lambda d: np.average(d[ab + "_mbps"].fillna(0), weights=d[ab + "_tests"]) if d[ab + "_tests"].sum() > 0 else np.nan,
                                     include_groups=False)
    natl = np.average(C[ab + "_mbps"].fillna(0), weights=C[ab + "_tests"])
    prior = C.PARENT.map(prov).fillna(natl)
    n = C[ab + "_tests"]
    C[ab + "_mbps_shrunk"] = (n * C[ab + "_mbps"].fillna(0) + 30 * prior) / (n + 30)
    C[ab + "_tests_pc"] = n / C.pop_2019 * 1000
C["exclude"] = C.ADMIN_NAME.str.contains("Lake", case=False) | (C.pop_2019 < 1000)
C["ntl_pc_2019"] = C.get("ntl_2019", np.nan) / C.pop_2019
C["ntl_pc_2015"] = C.get("ntl_2015", np.nan) / C.pop_2015   # main since spec 9-8: measured before the 2017-2020 projects
C.to_csv(OUT / "kh_covariates.csv", index=False)
print(C.drop(columns=["ADMIN_NAME"]).describe().round(2).T[["mean", "min", "50%", "max"]].to_string())
for ab in ["mob", "fix"]:
    if ab + "_tests" in C:
        k = C[~C.exclude]
        print(f"{ab}: units with >=30 tests {(k[ab + '_tests'] >= 30).mean():.2f} | zero-test units {(k[ab + '_tests'] == 0).sum()} | total tests {int(k[ab + '_tests'].sum())}")
