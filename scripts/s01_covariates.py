"""Posto-level covariates: population, density, urban share, RWI, terrain, rainfall, Ookla network.

Output: data/processed/posto_covariates.gpkg (65 postos, EPSG:4326) + .csv
"""
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.mask import mask
from rasterio.merge import merge
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)
UTM = 32751  # WGS84 / UTM 51S

po = gpd.read_file(RAW / "cod_ab/tls_admin_boundaries.shp/tls_admin2.shp")
po = po[["adm2_pcode", "adm2_name", "adm1_pcode", "adm1_name", "geometry"]].rename(
    columns={"adm2_pcode": "pcode", "adm2_name": "posto", "adm1_pcode": "mun_pcode", "adm1_name": "municipio"})
po["area_km2"] = po.to_crs(UTM).area / 1e6


def zonal(path, geoms, stat, nodata_to_nan=True):
    out = []
    with rasterio.open(path) as src:
        for g in geoms:
            arr, _ = mask(src, [g], crop=True, filled=True, nodata=np.nan if src.dtypes[0].startswith("float") else 0)
            a = arr[0].astype("float64")
            if src.nodata is not None:
                a[a == src.nodata] = np.nan
            if nodata_to_nan:
                a[a < -1e30] = np.nan
            out.append(stat(a))
    return np.array(out)


# --- population (WorldPop) ---
for y in [2005, 2010, 2015, 2020]:
    po[f"pop{y}"] = zonal(RAW / f"worldpop/tls_ppp_{y}_1km_UNadj.tif", po.geometry, np.nansum)
po["pop100m_2020"] = zonal(RAW / "worldpop/tls_ppp_2020_100m.tif", po.geometry, np.nansum)
# urban share: population living in 100 m cells with > 15 persons (= > 1,500 /km2)
po["urban_share"] = zonal(RAW / "worldpop/tls_ppp_2020_100m.tif", po.geometry,
                          lambda a: np.nansum(np.where(a > 15, a, 0)) / max(np.nansum(a), 1))
# rescale 2020 WorldPop to Census-based 2022 municipality totals (COD-PS) for levels
cps = pd.read_excel(RAW / "cod_ps/tls_admpop_2022.xlsx", sheet_name="tls_admpop_adm1_2022")[["ADM1_PCODE", "T_TL"]]
cps.columns = ["mun_pcode", "mun_pop2022"]
po = po.merge(cps, on="mun_pcode", how="left")
po["pop2022"] = po["pop100m_2020"] / po.groupby("mun_pcode")["pop100m_2020"].transform("sum") * po["mun_pop2022"]
po["density"] = po["pop2022"] / po["area_km2"]

# --- Relative Wealth Index (Meta), population-weighted by 1 km WorldPop at the RWI point ---
rwi = pd.read_csv(RAW / "rwi/tls_relative_wealth_index.csv")
rg = gpd.GeoDataFrame(rwi, geometry=gpd.points_from_xy(rwi.longitude, rwi.latitude), crs=4326)
with rasterio.open(RAW / "worldpop/tls_ppp_2020_1km_UNadj.tif") as src:
    v = np.array([x[0] for x in src.sample(list(zip(rg.longitude, rg.latitude)))], dtype=float)
    v[v < 0] = 0
rg["w"] = v + 1
rg = gpd.sjoin(rg, po[["pcode", "geometry"]], predicate="within")
r = rg.groupby("pcode").apply(lambda d: pd.Series({"rwi": np.average(d.rwi, weights=d.w), "rwi_n": len(d)}),
                              include_groups=False)
po = po.merge(r, on="pcode", how="left")

# --- terrain (Copernicus GLO-90) ---
srcs = [rasterio.open(f) for f in sorted((RAW / "dem").glob("*.tif"))]
dem, tr = merge(srcs)
meta = srcs[0].meta.copy(); meta.update(height=dem.shape[1], width=dem.shape[2], transform=tr, count=1)
[s.close() for s in srcs]
z = dem[0].astype("float64")
# slope in degrees using local metric pixel size (~ 3 arc-seconds)
lat = tr.f + tr.e * (np.arange(z.shape[0]) + 0.5)
dy = abs(tr.e) * 111320.0
dx = tr.a * 111320.0 * np.cos(np.radians(lat))[:, None]
gy, gx = np.gradient(z)
slope = np.degrees(np.arctan(np.hypot(gx / dx, gy / dy)))
meta.update(dtype="float32", nodata=-9999)
INT = ROOT / "data" / "interim"; INT.mkdir(exist_ok=True)
for name, arr in [("dem", z), ("slope", slope)]:
    with rasterio.open(INT / f"{name}.tif", "w", **meta) as dst:
        dst.write(arr.astype("float32"), 1)
po["elev_mean"] = zonal(INT / "dem.tif", po.geometry, np.nanmean)
po["slope_mean"] = zonal(INT / "slope.tif", po.geometry, np.nanmean)
po["steep_share"] = zonal(INT / "slope.tif", po.geometry, lambda a: np.nanmean(a > 25))  # landslide-prone terrain

# --- wet-season rainfall (WorldClim 2.1, Dec-Apr) ---
pr = []
for m in [12, 1, 2, 3, 4]:
    pr.append(zonal(RAW / f"climate/wc2.1_5m_prec/wc2.1_5m_prec_{m:02d}.tif", po.geometry.buffer(0.05), np.nanmean))
po["wet_rain_mm"] = np.sum(pr, axis=0)

# --- Ookla network (mobile + fixed, 2019Q1 / 2022Q1 / 2025Q1) ---
def qk_bounds(qk):
    x = y = 0; zz = len(qk)
    for i in range(zz):
        m = 1 << (zz - i - 1); d = int(qk[i])
        if d & 1: x |= m
        if d & 2: y |= m
    n = 2 ** zz
    lon = x / n * 360 - 180
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + 0.5) / n))))
    return lon + 180 / n, lat  # tile centre (approx.)

for typ in ["mobile", "fixed"]:
    for y in [2019, 2022, 2025]:
        f = RAW / f"ookla/ookla_{typ}_{y}Q1_TL.csv"
        o = pd.read_csv(f, dtype={"quadkey": str})
        c = o.quadkey.map(qk_bounds)
        g = gpd.GeoDataFrame(o, geometry=gpd.points_from_xy([a for a, b in c], [b for a, b in c]), crs=4326)
        g = gpd.sjoin(g, po[["pcode", "geometry"]], predicate="within")
        agg = g.groupby("pcode").apply(lambda d: pd.Series({
            f"ook_{typ}_{y}_tiles": len(d),
            f"ook_{typ}_{y}_tests": d.tests.sum(),
            f"ook_{typ}_{y}_dl_mbps": np.average(d.avg_d_kbps, weights=d.tests) / 1000,
            f"ook_{typ}_{y}_lat_ms": np.average(d.avg_lat_ms, weights=d.tests)}), include_groups=False)
        po = po.merge(agg, on="pcode", how="left")
        for col in [f"ook_{typ}_{y}_tiles", f"ook_{typ}_{y}_tests"]:
            po[col] = po[col].fillna(0)

# network composite: tested tiles per 10k pop (presence of usable mobile data) and pooled mobile speed
po["ook_mobile_tiles_all"] = po[[f"ook_mobile_{y}_tiles" for y in [2019, 2022, 2025]]].sum(axis=1)
po["ook_mobile_tests_all"] = po[[f"ook_mobile_{y}_tests" for y in [2019, 2022, 2025]]].sum(axis=1)
po["net_tiles_per10k"] = po["ook_mobile_tiles_all"] / po["pop2022"] * 1e4
po["net_tests_per1k"] = po["ook_mobile_tests_all"] / po["pop2022"] * 1e3

po.to_file(PROC / "posto_covariates.gpkg", driver="GPKG")
po.drop(columns="geometry").to_csv(PROC / "posto_covariates.csv", index=False, encoding="utf-8-sig")
print(po.drop(columns="geometry").describe().T[["mean", "min", "max"]].round(2).to_string())
print(po[["posto", "municipio", "pop2022", "density", "rwi", "slope_mean", "ook_mobile_2025_tests"]].sort_values("pop2022").to_string())
