"""Cambodia map figures: where do aid projects sit relative to (1) night lights, (2) physical access, (3) connectivity?
Left: the characteristic at fine resolution, clipped to Cambodia; dots = GODAD project locations (district-reliable, starts
2017-2020), sized by the number of projects at the same coordinate, coloured by donor group.
Right: projects per 100,000 people by district quintile of the characteristic (national rate dashed) + Spearman rho.
Outputs: outputs/figures/cambodia/M1_nightlights.png, M2_access.png, M3_connectivity.png
"""
import math
import numpy as np, pandas as pd, geopandas as gpd, rasterio, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from rasterio.mask import mask
from rasterio.plot import plotting_extent
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data/raw/cambodia", ROOT / "data/processed/cambodia"
OUT = ROOT / "outputs/figures/cambodia"; OUT.mkdir(parents=True, exist_ok=True)
INK, MUTED = "#1f2328", "#6e7781"
DONOR = {"EU/US": "#e8590c", "China": "#c2255c", "World Bank": "#1971c2"}
U = gpd.read_file(PROC / "kh_units.gpkg").rename(columns={"GEOLEVEL2": "unit"})
U["unit"] = U.unit.astype("int64")
KH = U.union_all()
PROV = U.dissolve("PARENT")
DF = pd.read_csv(PROC / "kh_covariates.csv").merge(pd.read_csv(PROC / "kh_access.csv").rename(columns={"k": "unit"}), on="unit", how="left")
A = pd.read_csv(PROC / "kh_assign.csv")
DF["N"] = DF.unit.map(A[A.level == "adm2"].groupby("unit").w.sum()).fillna(0)
DK = DF[~DF.exclude].copy()
K = pd.read_csv(RAW / "godad_khm.csv", low_memory=False)
K["dgroup"] = np.select([K.donor.eq("World Bank"), K.donor.str.contains("China", na=False)], ["World Bank", "China"], "EU/US")
K["adm2"] = np.where(K.dgroup == "EU/US", K.precision_crs == 3, K.precision_code <= 3)
pts = K[K.startyear.between(2017, 2020) & K.adm2].copy()
pts["xy"] = pts.longitude.round(3).astype(str) + "_" + pts.latitude.round(3).astype(str)
pts = pts.groupby(["xy", "dgroup"]).agg(lon=("longitude", "first"), lat=("latitude", "first"), n=("project_id", "nunique")).reset_index()


def layout(title, sub):
    fig = plt.figure(figsize=(8.2, 5.0), dpi=220)
    ax = fig.add_axes([0.01, 0.03, 0.66, 0.82]); axb = fig.add_axes([0.73, 0.42, 0.24, 0.36]); axn = fig.add_axes([0.71, 0.02, 0.27, 0.28])
    fig.text(0.015, 0.95, title, fontsize=11, fontweight="bold", color=INK)
    fig.text(0.015, 0.905, sub, fontsize=6.6, color=MUTED)
    axn.axis("off")
    return fig, ax, axb, axn


def base_map(ax, dark=False):
    PROV.boundary.plot(ax=ax, color="white", linewidth=0.6, alpha=0.75 if dark else 0.95, zorder=3)
    U.boundary.plot(ax=ax, color="white", linewidth=0.15, alpha=0.35 if dark else 0.6, zorder=3)
    for g, col in DONOR.items():
        d = pts[pts.dgroup == g]
        ax.scatter(d.lon, d.lat, s=7 + 10 * np.sqrt(d.n), color=col, edgecolor="#0b1020" if dark else "white", linewidth=0.5, alpha=0.92, zorder=5)
    ax.set_xlim(102.3, 107.7); ax.set_ylim(10.35, 14.75); ax.set_axis_off()


def quintiles(axb, axn, var, label, low="low", high="high", extra=""):
    d = DK.dropna(subset=[var]).copy()
    d["q"] = pd.qcut(d[var].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    s = d.groupby("q").agg(N=("N", "sum"), pop=("pop_2019", "sum"))
    pc = (s.N / s["pop"] * 1e5).values; natl = d.N.sum() / d.pop_2019.sum() * 1e5
    axb.bar(range(1, 6), pc, color=["#ced4da"] * 5, width=0.72)
    axb.bar(int(np.argmax(pc)) + 1, pc.max(), color=INK, width=0.72)
    axb.axhline(natl, color=INK, lw=0.8, ls="--"); axb.text(0.55, natl, " national rate", fontsize=5.5, va="bottom", color=INK)
    axb.set_xticks(range(1, 6)); axb.set_xticklabels([low, "2", "3", "4", high], fontsize=6)
    axb.tick_params(axis="y", labelsize=6, colors=MUTED); axb.tick_params(axis="x", length=0)
    for sp in ["top", "right"]:
        axb.spines[sp].set_visible(False)
    axb.set_title(f"Projects per 100,000 people\nby district quintile of {label}", fontsize=7, color=INK, loc="left")
    rho = d[var].corr(d.N / d.pop_2019, method="spearman")
    h = [plt.scatter([], [], s=28, color=c, edgecolor="white", linewidth=0.5) for c in DONOR.values()]
    axn.legend(h, [f"{k} project location" for k in DONOR], loc="upper left", fontsize=6.3, frameon=False, title="Dot size = projects at that place",
               title_fontsize=6.3, alignment="left")
    axn.text(0.0, 0.22, f"Spearman ρ = {rho:+.2f}\n(district {label} vs projects per person)\n{len(d)} districts · projects starting 2017–2020{extra}",
             fontsize=5.9, color=INK, transform=axn.transAxes, va="top")


# ---- M1 night lights
fig, ax, axb, axn = layout("Aid projects and night lights", "Harmonized DMSP–VIIRS night lights 2019 (clipped to Cambodia) · dots: GODAD project locations, district-precise")
with rasterio.open(RAW / "ntl/ntl_2019_KHM.tif") as s:
    a, tr = mask(s, [KH], crop=True, filled=True, nodata=0); a = a[0].astype(float)
    ext = (tr.c, tr.c + tr.a * a.shape[1], tr.f + tr.e * a.shape[0], tr.f)
a[a <= 0] = np.nan
U.plot(ax=ax, color="#10162a", edgecolor="none", zorder=1)
ax.imshow(a, extent=ext, cmap="magma", norm=LogNorm(vmin=1, vmax=np.nanpercentile(a, 99.5)), zorder=2, interpolation="nearest")
base_map(ax, dark=True)
quintiles(axb, axn, "ntl_pc_2019", "night lights per person")
fig.savefig(OUT / "M1_nightlights.png", facecolor="white"); plt.close(fig)

# ---- M2 physical access
cells = pd.read_csv(PROC / "kh_cells_tt.csv")
fig, ax, axb, axn = layout("Aid projects and physical access", "Travel time to the nearest town ≥20,000 people (OSM road network, dry season, 1 km cells) · dots: GODAD project locations")
U.plot(ax=ax, color="#f1f3f5", edgecolor="none", zorder=1)
sc = ax.scatter(cells.lon, cells.lat, c=np.clip(cells.tt_town_dry, 5, 600), s=0.4, marker="s", cmap="YlGnBu_r", norm=LogNorm(vmin=5, vmax=600),
                linewidths=0, zorder=2, rasterized=True)
base_map(ax)
cax = fig.add_axes([0.735, 0.345, 0.22, 0.014]); cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
cb.ax.set_title("minutes to nearest town", fontsize=5.8, loc="left"); cb.ax.tick_params(labelsize=5.5)
quintiles(axb, axn, "access_pc1", "physical access", low="far", high="near")
fig.savefig(OUT / "M2_access.png"); plt.close(fig)

# ---- M3 connectivity
o = pd.concat([pd.read_csv(f) for f in sorted((RAW / "ookla").glob("ookla_mobile_*_KH.csv"))])
o = o.groupby(["tx", "ty"]).apply(lambda d: pd.Series({"mbps": np.average(d.avg_d_kbps, weights=d.tests) / 1000, "tests": d.tests.sum()}),
                                  include_groups=False).reset_index()
n16 = 2 ** 16
o["lon"] = (o.tx + .5) / n16 * 360 - 180
o["lat"] = [math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + .5) / n16)))) for y in o.ty]
o = o[gpd.GeoSeries(gpd.points_from_xy(o.lon, o.lat), crs=4326).within(KH).values]
fig, ax, axb, axn = layout("Aid projects and digital connectivity", "Ookla mobile speed tests (≈600 m tiles, 2019 Q1–2020 Q1, clipped to Cambodia) · dots: GODAD project locations")
U.plot(ax=ax, color="#f3f0ff", edgecolor="none", zorder=1)
sc = ax.scatter(o.lon, o.lat, c=o.mbps.clip(1, 60), s=0.7 + 0.5 * np.log1p(o.tests), marker="s", cmap="Purples", norm=LogNorm(vmin=1, vmax=60),
                linewidths=0, zorder=2, rasterized=True)
base_map(ax)
cax = fig.add_axes([0.735, 0.345, 0.22, 0.014]); cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
cb.ax.set_title("mobile download, Mbps", fontsize=5.8, loc="left"); cb.ax.tick_params(labelsize=5.5)
quintiles(axb, axn, "mob_mbps_shrunk", "mobile speed", extra=f" · {int(o.tests.sum()):,} tests")
fig.savefig(OUT / "M3_connectivity.png"); plt.close(fig)
print("maps:", sorted(p.name for p in OUT.glob("M*.png")))
