"""High-resolution map backgrounds for the archive site (plate carree, fixed bbox so the page can overlay points exactly).
bbox lon 102.3-107.7, lat 10.35-14.75; image 1800 x 1503 px (aspect = 5.4*cos(12.5 deg) / 4.4). Transparent outside Cambodia.
site/img/bg_night.png   harmonized DMSP-VIIRS night lights 2015 on a night-blue country fill
site/img/bg_access.png  travel time to the nearest town >= 20k (1 km cells), lake in grey
site/img/bg_signal.png  Ookla mobile speed tiles (2019 Q1-2020 Q1) on a pale fill
"""
import math
import numpy as np, pandas as pd, geopandas as gpd, rasterio, matplotlib
from rasterio.mask import mask
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC, OUT = ROOT / "data/raw/cambodia", ROOT / "data/processed/cambodia", ROOT / "site/img"
OUT.mkdir(parents=True, exist_ok=True)
X0, X1, Y0, Y1 = 102.3, 107.7, 10.35, 14.75
W = 1800; H = int(round(W * 4.4 / (5.4 * math.cos(math.radians(12.55)))))
U = gpd.read_file(PROC / "kh_units.gpkg")
KH = U.union_all()
lake = U[U.ADMIN_NAME.str.contains("Lake", case=False)]


def canvas():
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1); ax.axis("off")
    fig.patch.set_alpha(0); ax.patch.set_alpha(0)
    return fig, ax


def save(fig, ax, name):
    # imshow and GeoDataFrame.plot set an equal aspect, which shrinks the axes inside the figure (1467 of 1503 px)
    # and misaligns the page's linear lon/lat overlay; force the data to fill the whole canvas
    ax.set_aspect("auto"); ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1)
    fig.savefig(OUT / name, transparent=True); plt.close(fig)


def grid_from_points(lon, lat, val, res):
    nx, ny = int((X1 - X0) / res), int((Y1 - Y0) / res)
    g = np.full((ny, nx), np.nan)
    ix = ((lon - X0) / res).astype(int); iy = ((Y1 - lat) / res).astype(int)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    g[iy[ok], ix[ok]] = val[ok]
    return g


# night lights
fig, ax = canvas()
gpd.GeoSeries([KH]).plot(ax=ax, color="#0e1527", edgecolor="none", zorder=0)
with rasterio.open(RAW / "ntl/ntl_2015_KHM.tif") as s:
    a, tr = mask(s, [KH], crop=True, filled=True, nodata=0); a = a[0].astype(float)
    ext = (tr.c, tr.c + tr.a * a.shape[1], tr.f + tr.e * a.shape[0], tr.f)
a[a <= 0] = np.nan
ax.imshow(a, extent=ext, cmap="magma", norm=LogNorm(vmin=1, vmax=np.nanpercentile(a, 99.5)), interpolation="nearest", zorder=1)
save(fig, ax, "bg_night.png")

# physical access: 1 km travel-time cells
c = pd.read_csv(PROC / "kh_cells_tt.csv")
g = grid_from_points(c.lon.values, c.lat.values, np.clip(c.tt_town_dry.values, 5, 600), 1 / 120)
fig, ax = canvas()
gpd.GeoSeries([KH]).plot(ax=ax, color="#eef2ee", edgecolor="none", zorder=0)
ax.imshow(g, extent=(X0, X0 + g.shape[1] / 120, Y1 - g.shape[0] / 120, Y1), cmap="YlGnBu_r", norm=LogNorm(vmin=5, vmax=600), interpolation="nearest", zorder=1)
if len(lake):
    lake.plot(ax=ax, color="#d9dee3", edgecolor="none", zorder=2)
save(fig, ax, "bg_access.png")

# signal: Ookla mobile tiles
o = pd.concat([pd.read_csv(f) for f in sorted((RAW / "ookla").glob("ookla_mobile_*_KH.csv"))])
o = o.groupby(["tx", "ty"]).apply(lambda d: pd.Series({"mbps": np.average(d.avg_d_kbps, weights=d.tests) / 1000}), include_groups=False).reset_index()
n16 = 2 ** 16
o["lon"] = (o.tx + .5) / n16 * 360 - 180
o["lat"] = [math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + .5) / n16)))) for y in o.ty]
o = o[gpd.GeoSeries(gpd.points_from_xy(o.lon, o.lat), crs=4326).within(KH).values]
g = grid_from_points(o.lon.values, o.lat.values, np.clip(o.mbps.values, 1, 60), 1 / 160)
fig, ax = canvas()
gpd.GeoSeries([KH]).plot(ax=ax, color="#f1effa", edgecolor="none", zorder=0)
ax.imshow(g, extent=(X0, X0 + g.shape[1] / 160, Y1 - g.shape[0] / 160, Y1), cmap="Purples", norm=LogNorm(vmin=1, vmax=60), interpolation="nearest", zorder=1)
save(fig, ax, "bg_signal.png")
print("image size", W, H, {p.name: round(p.stat().st_size / 1e6, 2) for p in OUT.glob("*.png")})
