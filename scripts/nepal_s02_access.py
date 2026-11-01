"""Nepal accessibility per district (protocol §5.1): OSM travel time to Kathmandu (dry / monsoon), paved share,
road density, steep-terrain share, monsoon penalty; Ookla (secondary), night lights 2010, RWI, WorldPop.
Sparse-graph Dijkstra (scipy) for the large Nepal network. Output: data/processed/nepal/district_access.csv
"""
import math
import numpy as np, pandas as pd, geopandas as gpd, rasterio, osmium
from rasterio.merge import merge
from rasterio.mask import mask
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra, connected_components
from scipy.spatial import cKDTree
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw" / "nepal", ROOT / "data" / "processed" / "nepal"
D = gpd.read_file(OUT / "districts.gpkg")
BASE = {"motorway": 60, "trunk": 50, "primary": 45, "secondary": 35, "tertiary": 25, "unclassified": 20, "residential": 15,
        "living_street": 10, "service": 12, "road": 15, "track": 10, "path": 3.5, "footway": 3.5, "bridleway": 3.5, "steps": 2}
for k in list(BASE):
    BASE[k + "_link"] = BASE[k]
PAVED = {"asphalt", "concrete", "paved", "paving_stones", "concrete:plates", "chipseal", "sett", "metal"}
UNPAVED = {"unpaved", "gravel", "dirt", "ground", "earth", "compacted", "mud", "sand", "fine_gravel", "grass", "rock", "pebblestone"}
DEF_PAVED = {"motorway", "trunk", "primary", "motorway_link", "trunk_link", "primary_link"}


class H(osmium.SimpleHandler):
    def __init__(self):
        super().__init__(); self.e = []; self.nodes = {}

    def way(self, w):
        hw = w.tags.get("highway")
        if hw not in BASE:
            return
        surf = w.tags.get("surface")
        paved = True if surf in PAVED else (False if surf in UNPAVED else hw in DEF_PAVED)
        try:
            ns = [(n.ref, n.lon, n.lat) for n in w.nodes]
        except osmium.InvalidLocationError:
            return
        for (a, x1, y1), (b, x2, y2) in zip(ns[:-1], ns[1:]):
            self.e.append((a, b, x1, y1, x2, y2, hw, paved))


h = H(); h.apply_file(str(RAW / "osm/nepal-latest.osm.pbf"), locations=True)
E = pd.DataFrame(h.e, columns=["a", "b", "x1", "y1", "x2", "y2", "hw", "paved"]); del h
print("edges", len(E))
lat_m = np.radians((E.y1 + E.y2) / 2)
E["km"] = np.hypot((E.x2 - E.x1) * 111.32 * np.cos(lat_m), (E.y2 - E.y1) * 110.57)
E = E[E.km > 0].reset_index(drop=True)

# DEM at 6 arc-seconds
srcs = [rasterio.open(f) for f in sorted((RAW / "dem").glob("*.tif"))]
dem, tr = merge(srcs, res=(1 / 600, 1 / 600)); [s.close() for s in srcs]
z = dem[0].astype("float32")
z1 = z[np.clip(((E.y1 - tr.f) / tr.e).astype(int), 0, z.shape[0] - 1), np.clip(((E.x1 - tr.c) / tr.a).astype(int), 0, z.shape[1] - 1)]
z2 = z[np.clip(((E.y2 - tr.f) / tr.e).astype(int), 0, z.shape[0] - 1), np.clip(((E.x2 - tr.c) / tr.a).astype(int), 0, z.shape[1] - 1)]
E["grade"] = np.clip(np.abs(z2 - z1) / (E.km * 1000), 0, 0.4)
latc = tr.f + tr.e * (np.arange(z.shape[0]) + 0.5)
gy, gx = np.gradient(z)
slope = np.degrees(np.arctan(np.hypot(gx / (tr.a * 111320 * np.cos(np.radians(latc)))[:, None], gy / (abs(tr.e) * 110570))))
del gx, gy
ls = slope[np.clip((((E.y1 + E.y2) / 2 - tr.f) / tr.e).astype(int), 0, z.shape[0] - 1), np.clip((((E.x1 + E.x2) / 2 - tr.c) / tr.a).astype(int), 0, z.shape[1] - 1)]
v = E.hw.map(BASE).astype(float).values
v = np.where(E.paved, v, np.minimum(v, 25) * 0.7)
v = v * np.clip(1 - 2.0 * E.grade.values, 0.4, 1)
E["v_dry"] = v
E["v_wet"] = v * np.where(E.paved, 0.85, 0.5) * np.where(ls > 25, 0.7, 1.0)          # monsoon

ids = pd.Index(pd.unique(np.r_[E.a.values, E.b.values]))
ai, bi = ids.get_indexer(E.a), ids.get_indexer(E.b)
n = len(ids)
coords = pd.concat([pd.DataFrame({"n": E.a, "x": E.x1, "y": E.y1}), pd.DataFrame({"n": E.b, "x": E.x2, "y": E.y2})]).drop_duplicates("n").set_index("n").loc[ids]


def graph(w):
    return coo_matrix((np.r_[w, w], (np.r_[ai, bi], np.r_[bi, ai])), shape=(n, n)).tocsr()


Gd, Gw = graph(E.km / E.v_dry * 60), graph(E.km / E.v_wet * 60)
_, lab = connected_components(Gd, directed=False)
kx = np.cos(np.radians(28.3))
tree_all = cKDTree(np.c_[coords.x.values * kx, coords.y.values])
src = tree_all.query([85.3240 * kx, 27.7172])[1]
main = lab == lab[src]
tree = cKDTree(np.c_[coords.x.values[main] * kx, coords.y.values[main]]); main_idx = np.where(main)[0]
td = dijkstra(Gd, indices=src, directed=False); tw = dijkstra(Gw, indices=src, directed=False)
print("main component nodes", main.sum(), "of", n)

with rasterio.open(RAW / "worldpop/npl_ppp_2020_1km_UNadj.tif") as s:
    a = s.read(1); t = s.transform
rr, cc = np.where(a > 0); xs, ys = rasterio.transform.xy(t, rr, cc)
cells = gpd.GeoDataFrame({"pop": a[rr, cc]}, geometry=gpd.points_from_xy(xs, ys), crs=4326)
cells = gpd.sjoin(cells, D[["k", "geometry"]], predicate="within").drop(columns="index_right")
dd, jj = tree.query(np.c_[cells.geometry.x * kx, cells.geometry.y])
node = main_idx[jj]
cells["walk"] = dd * 111.32 * 1.3 / 3.5 * 60                # off-network walking, 3.5 km/h (hills), 1.3 detour
cells["tt_dry"] = td[node] + cells.walk
cells["tt_wet"] = tw[node] + cells.walk
cells = cells[np.isfinite(cells.tt_wet)]
pd.DataFrame({"lon": cells.geometry.x, "lat": cells.geometry.y, "pop": cells["pop"], "k": cells.k, "tt_dry": cells.tt_dry,
              "tt_wet": cells.tt_wet, "walk": cells.walk}).to_csv(OUT / "cells_traveltime.csv", index=False)   # for palika-level analysis
g = cells.groupby("k")
A = pd.DataFrame({"tt_dry_min": g.apply(lambda d: np.average(d.tt_dry, weights=d["pop"]), include_groups=False),
                  "tt_wet_min": g.apply(lambda d: np.average(d.tt_wet, weights=d["pop"]), include_groups=False),
                  "walk_min": g.apply(lambda d: np.average(d.walk, weights=d["pop"]), include_groups=False),
                  "pop2020_wp": g["pop"].sum()})
A["wet_penalty_min"] = A.tt_wet_min - A.tt_dry_min

# road length / paved share / density (car roads)
E["mx"], E["my"] = (E.x1 + E.x2) / 2, (E.y1 + E.y2) / 2
car = E[E.hw.map(BASE) >= 10]
mp = gpd.sjoin(gpd.GeoDataFrame(car[["km", "paved"]], geometry=gpd.points_from_xy(car.mx, car.my), crs=4326), D[["k", "geometry"]], predicate="within")
R = mp.groupby("k").apply(lambda d: pd.Series({"road_km": d.km.sum(), "paved_km": d.km[d.paved].sum()}), include_groups=False)
A = A.join(R)
meta = {"driver": "GTiff", "dtype": "float32", "count": 1, "height": z.shape[0], "width": z.shape[1], "transform": tr, "crs": "EPSG:4326", "nodata": -9999}
with rasterio.io.MemoryFile() as mf:
    with mf.open(**meta) as ds:
        ds.write(slope.astype("float32"), 1)
        st, el = [], []
        for geom in D.geometry:
            s_, _ = mask(ds, [geom], crop=True, filled=True, nodata=-9999); s_ = s_[0]; s_ = s_[s_ > -9000]
            st.append(np.mean(s_ > 25)); el.append(np.mean(s_))
A = A.join(pd.DataFrame({"k": D.k, "steep_share": st, "slope_mean": el}).set_index("k"))
# night lights 2010, RWI, Ookla (secondary network)
with rasterio.open(RAW / "ntl/ntl_2010_NPL.tif") as s:
    A = A.join(pd.DataFrame({"k": D.k, "ntl2010": [mask(s, [gg], crop=True, filled=True, nodata=0)[0][0].astype(float).sum() for gg in D.geometry]}).set_index("k"))
rw = pd.read_csv(RAW / "rwi/npl_relative_wealth_index.csv")
rw = gpd.sjoin(gpd.GeoDataFrame(rw, geometry=gpd.points_from_xy(rw.longitude, rw.latitude), crs=4326), D[["k", "geometry"]], predicate="within")
A = A.join(rw.groupby("k").rwi.mean().rename("rwi"))


def qk_center(qk):
    x = y = 0; zz = len(qk)
    for i in range(zz):
        m = 1 << (zz - i - 1); c = int(qk[i])
        if c & 1: x |= m
        if c & 2: y |= m
    N = 2 ** zz
    return (x + .5) / N * 360 - 180, math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + .5) / N))))


for yy in ["2019", "2025"]:
    o = pd.read_csv(RAW / f"ookla/ookla_mobile_{yy}Q1_NP.csv", dtype={"quadkey": str})
    c = o.quadkey.map(qk_center)
    og = gpd.sjoin(gpd.GeoDataFrame(o, geometry=gpd.points_from_xy([p[0] for p in c], [p[1] for p in c]), crs=4326), D[["k", "geometry"]], predicate="within")
    A = A.join(og.groupby("k").agg(**{f"ook{yy}_tiles": ("tests", "size"), f"ook{yy}_tests": ("tests", "sum")}))
A = A.fillna({"ook2019_tiles": 0, "ook2019_tests": 0, "ook2025_tiles": 0, "ook2025_tests": 0})
A.reset_index().to_csv(OUT / "district_access.csv", index=False)
print(A.sort_values("tt_wet_min")[["tt_dry_min", "tt_wet_min", "road_km", "paved_km", "steep_share", "ook2019_tests"]].round(1).to_string())
