"""Chambers' tarmac (roadside) bias, measured at project points.

For every precisely geocoded (precision 1) non-infrastructure development project location, compute the distance to the
nearest tarmac road (OSM paved trunk/primary/secondary/tertiary). Compare it with the population living in the SAME
upper unit (municipality / district): the roadside percentile = share of that unit's population living closer to tarmac
than the project point. No bias -> mean percentile 0.5; tarmac bias -> mean below 0.5.
Also: share of project points vs population within 1 / 2 / 5 km of tarmac, and a road-network-free comparison with
the nearest Ookla-tested tile (network proximity).
Outputs: outputs/tables/tarmac_bias.csv, outputs/results_tarmac.txt, figure outputs/figures/fig_tarmac.png
"""
import json, math
import numpy as np, pandas as pd, geopandas as gpd, rasterio, osmium
from scipy.spatial import cKDTree
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = open(ROOT / "outputs/results_tarmac.txt", "w", encoding="utf-8")
say = lambda *a: (print(" ".join(map(str, a))), LOG.write(" ".join(map(str, a)) + "\n"))
DEV = {"SOC_education", "SOC_health", "SOC_water", "SOC_other", "GOV", "PROD", "ENV_MULTI"}
MAJOR = {"trunk", "primary", "secondary", "tertiary", "trunk_link", "primary_link", "secondary_link", "tertiary_link"}
PAVED = {"asphalt", "concrete", "paved", "paving_stones", "chipseal", "concrete:plates"}


def densify(lines, step_km=0.1):
    pts = []
    for xs, ys in lines:
        for (x1, y1), (x2, y2) in zip(zip(xs[:-1], ys[:-1]), zip(xs[1:], ys[1:])):
            L = math.hypot((x2 - x1) * 111.32 * math.cos(math.radians(y1)), (y2 - y1) * 110.57)
            n = max(1, int(L / step_km))
            t = np.linspace(0, 1, n, endpoint=False)
            pts.append(np.c_[x1 + (x2 - x1) * t, y1 + (y2 - y1) * t])
    return np.vstack(pts)


def qk_centres(csvs):
    out = []
    for f in csvs:
        for qk in pd.read_csv(f, dtype={"quadkey": str}).quadkey:
            x = y = 0; zz = len(qk)
            for i in range(zz):
                m = 1 << (zz - i - 1); c = int(qk[i])
                if c & 1: x |= m
                if c & 2: y |= m
            n = 2 ** zz
            out.append(((x + .5) / n * 360 - 180, math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + .5) / n))))))
    return np.array(out)


def analyse(name, road_pts, ookla_pts, proj, pop_tif, units, ucol, lat0):
    kx = math.cos(math.radians(lat0))
    tree = cKDTree(np.c_[road_pts[:, 0] * kx, road_pts[:, 1]]); otree = cKDTree(np.c_[ookla_pts[:, 0] * kx, ookla_pts[:, 1]])
    with rasterio.open(pop_tif) as s:
        a = s.read(1); rr, cc = np.where(a > 0); xs, ys = rasterio.transform.xy(s.transform, rr, cc)
    C = gpd.GeoDataFrame({"pop": a[rr, cc]}, geometry=gpd.points_from_xy(xs, ys), crs=4326)
    C = gpd.sjoin(C, units[[ucol, "geometry"]], predicate="within").drop(columns="index_right")
    C["d"] = tree.query(np.c_[C.geometry.x * kx, C.geometry.y])[0] * 111.32
    C["do"] = otree.query(np.c_[C.geometry.x * kx, C.geometry.y])[0] * 111.32
    Pg = gpd.sjoin(gpd.GeoDataFrame(proj, geometry=gpd.points_from_xy(proj.lon, proj.lat), crs=4326), units[[ucol, "geometry"]], predicate="within")
    Pg["d"] = tree.query(np.c_[Pg.geometry.x * kx, Pg.geometry.y])[0] * 111.32
    Pg["do"] = otree.query(np.c_[Pg.geometry.x * kx, Pg.geometry.y])[0] * 111.32
    pct, pcto, puid = [], [], []
    for u, grp in Pg.groupby(ucol):
        cu = C[C[ucol] == u]
        if cu["pop"].sum() == 0:
            continue
        for dist, col, lst in [(grp.d, "d", pct), (grp.do, "do", pcto)]:
            o = np.argsort(cu[col].values); cd = cu[col].values[o]; cw = np.cumsum(cu["pop"].values[o]) / cu["pop"].sum()
            lst += list(np.interp(dist.values, cd, cw, left=0, right=1))
        puid += list(grp.uid.values)
    pct, pcto = np.array(pct), np.array(pcto)
    pw = pd.DataFrame({"uid": puid, "p": pct, "po": pcto}).groupby("uid")[["p", "po"]].mean()   # each project weighted equally
    se = pct.std() / math.sqrt(len(pct)); seo = pcto.std() / math.sqrt(len(pcto))
    say("=" * 90 + f"\n{name}: {len(Pg)} precise dev-project locations ({Pg.uid.nunique()} projects), tarmac network points {len(road_pts):,}")
    say(f"  roadside percentile (tarmac): mean {pct.mean():.3f} (SE {se:.3f}); median {np.median(pct):.3f}   [0.5 = no bias]")
    say(f"  network percentile (nearest Ookla-tested tile): mean {pcto.mean():.3f} (SE {seo:.3f})")
    say(f"  project-weighted ({len(pw)} projects): tarmac percentile {pw.p.mean():.3f} (SE {pw.p.std() / math.sqrt(len(pw)):.3f}), "
        f"network percentile {pw.po.mean():.3f} (SE {pw.po.std() / math.sqrt(len(pw)):.3f}); share of projects with mean tarmac percentile < 0.5: {(pw.p < 0.5).mean():.0%}")
    out = []
    for km in [1, 2, 5]:
        sp = (Pg.d <= km).mean(); sc = C.loc[C.d <= km, "pop"].sum() / C["pop"].sum()
        say(f"  within {km} km of tarmac: projects {sp:.0%} vs population {sc:.0%} (ratio {sp / sc:.2f})")
        out.append({"country": name, "km": km, "proj_share": sp, "pop_share": sc})
    return pct, pcto, out, Pg, C


# ---------- Timor-Leste ----------
rd = gpd.read_file(ROOT / "data/processed/roads.gpkg")
rd = rd[rd.highway.isin(MAJOR) & rd.paved]
tl_roads = densify([(list(g.xy[0]), list(g.xy[1])) for geom in rd.geometry for g in (geom.geoms if geom.geom_type == "MultiLineString" else [geom])])
PRt = pd.read_csv(ROOT / "data/processed/oda_projects.csv"); devt = set(PRt.loc[PRt.cat.isin(DEV), "uid"])
ad = ROOT / "data/raw/aiddata_tl/TimorLesteAIMS_GeocodedResearchRelease_Level1_v1.4.1/TimorLesteAIMS_GeocodedResearchRelease_Level1_v1.4.1/data"
L = pd.read_csv(ad / "locations.csv", encoding="latin1"); L = L[L.precision_code == 1]
lt = pd.DataFrame({"uid": "AMP-" + L.project_id.astype(str), "lon": L.longitude, "lat": L.latitude})
il = pd.DataFrame(json.loads((ROOT / "data/raw/iati/dportal_loc_TL.json").read_text(encoding="utf-8")))
il = il[(il.location_precision == 1) & il.location_longitude.notna()]
xy = il.location_longitude.round(2).astype(str) + il.location_latitude.round(2).astype(str)
il = il[xy.map(il.assign(xy=xy).groupby("xy").aid.nunique()) < 40]
lt = pd.concat([lt, pd.DataFrame({"uid": "IATI-" + il.aid, "lon": il.location_longitude, "lat": il.location_latitude})])
lt = lt[lt.uid.isin(devt)]
mun = gpd.read_file(ROOT / "data/raw/cod_ab/tls_admin_boundaries.shp/tls_admin1.shp")[["adm1_pcode", "geometry"]]
oo_tl = qk_centres([ROOT / f"data/raw/ookla/ookla_mobile_{y}Q1_TL.csv" for y in [2019, 2022, 2025]])
r_tl = analyse("Timor-Leste", tl_roads, oo_tl, lt, ROOT / "data/raw/worldpop/tls_ppp_2020_1km_UNadj.tif", mun, "adm1_pcode", -8.8)


# ---------- Nepal (parse tarmac from OSM) ----------
class H(osmium.SimpleHandler):
    def __init__(self):
        super().__init__(); self.lines = []

    def way(self, w):
        hw = w.tags.get("highway"); s = w.tags.get("surface")
        if hw in MAJOR and (s in PAVED or (s is None and hw in {"trunk", "primary", "trunk_link", "primary_link"})):
            try:
                self.lines.append(([n.lon for n in w.nodes], [n.lat for n in w.nodes]))
            except osmium.InvalidLocationError:
                pass


h = H(); h.apply_file(str(ROOT / "data/raw/nepal/osm/nepal-latest.osm.pbf"), locations=True)
np_roads = densify(h.lines)
PRn = pd.read_csv(ROOT / "data/processed/nepal/oda_projects.csv"); devn = set(PRn.loc[PRn.cat.isin(DEV), "uid"])
na = ROOT / "data/raw/nepal/aiddata/NepalAIMS_GeocodedResearchRelease_Level1_v1.4.1/NepalAIMS_GeocodedResearchRelease_Level1_v1.4.1/data"
L = pd.read_csv(na / "locations.csv", encoding="latin1"); L = L[L.precision_code == 1]
ln = pd.DataFrame({"uid": "AIMS-" + L.project_id.astype(str), "lon": L.longitude, "lat": L.latitude})
ln = ln[ln.uid.isin(devn)]
dist = gpd.read_file(ROOT / "data/processed/nepal/districts.gpkg")[["k", "geometry"]]
oo_np = qk_centres([ROOT / f"data/raw/nepal/ookla/ookla_mobile_{y}Q1_NP.csv" for y in [2019, 2025]])
r_np = analyse("Nepal", np_roads, oo_np, ln, ROOT / "data/raw/nepal/worldpop/npl_ppp_2020_1km_UNadj.tif", dist, "k", 28.3)
pd.DataFrame(r_tl[2] + r_np[2]).to_csv(ROOT / "outputs/tables/tarmac_bias.csv", index=False)

# ---------- figure: distance-to-tarmac CDFs, projects vs population ----------
import matplotlib as mpl, matplotlib.pyplot as plt
SURF, INK2, AXIS, GRID = "#fcfcfb", "#52514e", "#c3c2b7", "#e1e0d9"
mpl.rcParams.update({"font.family": ["Malgun Gothic", "Segoe UI", "sans-serif"], "axes.unicode_minus": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF, "axes.grid": True, "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False, "axes.edgecolor": AXIS})
fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
for ax, (nm, r) in zip(axs[:2], [("동티모르", r_tl), ("네팔", r_np)]):
    Pg, C = r[3], r[4]
    xs = np.logspace(-1.5, 1.8, 200)
    ax.plot(xs, [(Pg.d <= x).mean() for x in xs], color="#eb6834", lw=2.4, label="사업 지점 (정밀 좌표)")
    ax.plot(xs, [C.loc[C.d <= x, "pop"].sum() / C["pop"].sum() for x in xs], color="#898781", lw=2.4, label="인구 (WorldPop)")
    ax.set_xscale("log"); ax.set_xlabel("가장 가까운 포장도로까지 거리 (km, 로그)"); ax.set_ylabel("누적 비율")
    ax.set_title(f"{nm}: 포장도로까지 거리 분포"); ax.legend(loc="upper left", fontsize=9)
ax = axs[2]
vals = [r_tl[0].mean(), r_tl[1].mean(), r_np[0].mean(), r_np[1].mean()]
ax.bar(["동티모르\n포장도로", "동티모르\n통신 측정점", "네팔\n포장도로", "네팔\n통신 측정점"], vals, color=["#2a78d6", "#1baf7a", "#2a78d6", "#1baf7a"], width=0.6)
ax.axhline(0.5, color=INK2, ls="--", lw=1); ax.text(3.35, 0.51, "편향 없음 = 0.5", ha="right", fontsize=8.5, color=INK2)
for i, v in enumerate(vals):
    ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9, color=INK2)
ax.set_ylim(0, 0.6); ax.set_ylabel("같은 지자체·군 인구 대비 백분위 (평균)"); ax.grid(axis="x", visible=False)
ax.set_title("사업 지점의 '도로변·통신 근접' 백분위")
fig.tight_layout(); fig.savefig(ROOT / "outputs/figures/fig_tarmac.png", dpi=200, bbox_inches="tight")
LOG.close()
