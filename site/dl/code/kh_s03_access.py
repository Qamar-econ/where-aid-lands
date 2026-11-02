"""Detailed physical accessibility for Cambodia's IPUMS GEO2_KH districts (adapted from np3_s02_access.py, Nepal).
Network: OSM roads + paths with surface- and gradient-dependent speeds, dry and monsoon (as nepal_s02_access.py).
Population-weighted (WorldPop 2020, 1 km) travel times from every populated cell to:
  Kathmandu (dry, monsoon) | nearest town >= 20,000 people (GeoNames) | nearest district headquarters (GeoNames PPLA2/PPLA3,
  one per district, nearest to the district centroid if several) | nearest hospital (OSM amenity=hospital)
plus walking time to the network, share of population within 30 min walk of a road, road density, paved share, paved density,
monsoon penalty share, steep-terrain share and mean slope.  Physical-access index = PC1 of the signed standardised components
(connectivity is deliberately NOT part of this index; it is a separate construct).
Outputs: data/processed/cambodia/kh_access.csv, outputs/tables/kh_access_pca.csv, data/processed/cambodia/kh_cells_tt.csv (1 km cells, for maps)
"""
import os
import numpy as np, pandas as pd, geopandas as gpd, rasterio, osmium
from rasterio.merge import merge
from rasterio.mask import mask
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra, connected_components
from scipy.spatial import cKDTree
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw" / "cambodia", ROOT / "data" / "processed" / "cambodia"
# KH_OSM_SNAPSHOT=latest builds the comparison index on the current extract; its outputs carry a _latest suffix
SNAP = os.environ.get("KH_OSM_SNAPSHOT", "170101")   # default since spec 9-6: the network as it stood before the 2017-2020 projects
SUF = "" if SNAP == "170101" else f"_{SNAP}"; OSM_FILE = f"cambodia-{SNAP}.osm.pbf"
# speed-assumption sensitivity (kh_s16): unpaved multiplier (default 0.7) and a scale on minor-road speeds (default 1)
UNPAVED_F = float(os.environ.get("KH_UNPAVED", "0.7")); MINOR_S = float(os.environ.get("KH_MINOR_SCALE", "1"))
# input-audit variants (kh_s19): towns = the 25 provincial capitals instead of GeoNames places >= 20k; untagged secondary
# roads treated as paved; grade measured over at least one DEM cell (185 m) so short segments get no spurious grade
# KH_SURFACE=class ignores OSM surface tags (present on ~20% of 2017 road length): dry speeds by road class only, wet-season
# factor by class group (secondary and above x0.85, below x0.5), and no paved-share component
CLASS_ONLY = os.environ.get("KH_SURFACE") == "class"
TOWNS = os.environ.get("KH_TOWNS", "pop20k"); SEC_PAVED = os.environ.get("KH_SEC_PAVED") == "1"; GRADE_MIN_KM = float(os.environ.get("KH_GRADE_MIN_KM", "0"))
if os.environ.get("KH_TAG"):
    SUF += "_" + os.environ["KH_TAG"]
D = gpd.read_file(OUT / "kh_units.gpkg").rename(columns={"GEOLEVEL2": "k"})
D["area_km2"] = D.to_crs(32648).area / 1e6
BASE = {"motorway": 60, "trunk": 50, "primary": 45, "secondary": 35, "tertiary": 25, "unclassified": 20, "residential": 15,
        "living_street": 10, "service": 12, "road": 15, "track": 10, "path": 3.5, "footway": 3.5, "bridleway": 3.5, "steps": 2}
for k in ["tertiary", "unclassified", "residential", "living_street", "service", "road", "track"]:
    BASE[k] *= MINOR_S
for k in list(BASE):
    BASE[k + "_link"] = BASE[k]
PAVED = {"asphalt", "concrete", "paved", "paving_stones", "concrete:plates", "chipseal", "sett", "metal"}
UNPAVED = {"unpaved", "gravel", "dirt", "ground", "earth", "compacted", "mud", "sand", "fine_gravel", "grass", "rock", "pebblestone"}
DEF_PAVED = {"motorway", "trunk", "primary", "motorway_link", "trunk_link", "primary_link"}
if SEC_PAVED:
    DEF_PAVED |= {"secondary", "secondary_link"}


class H(osmium.SimpleHandler):
    def __init__(self):
        super().__init__(); self.e = []; self.hosp = []

    def node(self, n):
        if n.tags.get("amenity") == "hospital" or n.tags.get("healthcare") == "hospital":
            self.hosp.append((n.location.lon, n.location.lat))

    def way(self, w):
        if w.tags.get("amenity") == "hospital":
            try:
                self.hosp.append((w.nodes[0].lon, w.nodes[0].lat))
            except Exception:
                pass
        hw = w.tags.get("highway")
        if hw not in BASE:
            return
        surf = w.tags.get("surface")
        paved = True if surf in PAVED else (False if surf in UNPAVED else hw in DEF_PAVED)
        if CLASS_ONLY:
            paved = hw in DEF_PAVED | {"secondary", "secondary_link"}
        try:
            ns = [(n.ref, n.lon, n.lat) for n in w.nodes]
        except osmium.InvalidLocationError:
            return
        for (a, x1, y1), (b, x2, y2) in zip(ns[:-1], ns[1:]):
            self.e.append((a, b, x1, y1, x2, y2, hw, paved))


h = H(); h.apply_file(str(RAW / "osm" / OSM_FILE), locations=True)
E = pd.DataFrame(h.e, columns=["a", "b", "x1", "y1", "x2", "y2", "hw", "paved"]); HOSP = np.array(h.hosp); del h
if os.environ.get("KH_HOSP") == "moh":   # input-audit variant: Ministry of Health referral hospitals (2010, HDX) instead of OSM tags
    _m = gpd.read_file(RAW / "moh_health/Health/khm_hltfacp_referral_gov.shp").to_crs(4326)
    HOSP = np.c_[_m.geometry.x, _m.geometry.y]
print("edges", len(E), "| hospitals", len(HOSP), flush=True)
lat_m = np.radians((E.y1 + E.y2) / 2)
E["km"] = np.hypot((E.x2 - E.x1) * 111.32 * np.cos(lat_m), (E.y2 - E.y1) * 110.57)
E = E[E.km > 0].reset_index(drop=True)
srcs = [rasterio.open(f) for f in sorted((RAW / "dem").glob("*.tif"))]
dem, tr = merge(srcs, res=(1 / 600, 1 / 600)); [s.close() for s in srcs]
z = dem[0].astype("float32")
ix = lambda x, y: (np.clip(((y - tr.f) / tr.e).astype(int), 0, z.shape[0] - 1), np.clip(((x - tr.c) / tr.a).astype(int), 0, z.shape[1] - 1))
E["grade"] = np.clip(np.abs(z[ix(E.x2, E.y2)] - z[ix(E.x1, E.y1)]) / (np.maximum(E.km, GRADE_MIN_KM) * 1000), 0, 0.4)
latc = tr.f + tr.e * (np.arange(z.shape[0]) + 0.5)
gy, gx = np.gradient(z)
slope = np.degrees(np.arctan(np.hypot(gx / (tr.a * 111320 * np.cos(np.radians(latc)))[:, None], gy / (abs(tr.e) * 110570)))); del gx, gy
ls = slope[ix((E.x1 + E.x2) / 2, (E.y1 + E.y2) / 2)]
v = E.hw.map(BASE).astype(float).values
v = (v if CLASS_ONLY else np.where(E.paved, v, np.minimum(v, 25) * UNPAVED_F)) * np.clip(1 - 2.0 * E.grade.values, 0.4, 1)
E["v_dry"], E["v_wet"] = v, v * np.where(E.paved, 0.85, 0.5) * np.where(ls > 25, 0.7, 1.0)
ids = pd.Index(pd.unique(np.r_[E.a.values, E.b.values])); ai, bi = ids.get_indexer(E.a), ids.get_indexer(E.b); n = len(ids)
coords = pd.concat([pd.DataFrame({"n": E.a, "x": E.x1, "y": E.y1}), pd.DataFrame({"n": E.b, "x": E.x2, "y": E.y2})]).drop_duplicates("n").set_index("n").loc[ids]
graph = lambda w: coo_matrix((np.r_[w, w], (np.r_[ai, bi], np.r_[bi, ai])), shape=(n, n)).tocsr()
Gd, Gw = graph(E.km / E.v_dry * 60), graph(E.km / E.v_wet * 60)
_, lab = connected_components(Gd, directed=False)
kx = np.cos(np.radians(12.5))
tree_all = cKDTree(np.c_[coords.x.values * kx, coords.y.values])
main = lab == np.bincount(lab).argmax(); main_idx = np.where(main)[0]  # largest connected component, not the capital's fragment
tree = cKDTree(np.c_[coords.x.values[main] * kx, coords.y.values[main]])
ktm = main_idx[tree.query([104.9282 * kx, 11.5564])[1]]  # Phnom Penh, snapped to the main network
print("main component nodes", main.sum(), "of", n, flush=True)
snap = lambda xy: np.unique(main_idx[tree.query(np.c_[xy[:, 0] * kx, xy[:, 1]])[1]])

# destinations
cols = ["geonameid", "name", "asciiname", "alt", "lat", "lon", "fclass", "fcode", "cc", "cc2", "a1", "a2", "a3", "a4", "pop", "elev", "dem", "tz", "mod"]
GN = pd.read_csv(RAW / "geonames/KH.txt", sep="\t", names=cols, dtype={"pop": float}, quoting=3, low_memory=False)
GN = GN[GN.fclass == "P"]
towns = (GN[GN.fcode.isin(["PPLA", "PPLC"])] if TOWNS == "capitals" else GN[GN["pop"] >= 20000])[["lon", "lat"]].values
hq = gpd.GeoDataFrame(GN[GN.fcode.isin(["PPLA2", "PPLA3", "PPLA", "PPLC"])], geometry=gpd.points_from_xy(GN[GN.fcode.isin(["PPLA2", "PPLA3", "PPLA", "PPLC"])].lon,
                      GN[GN.fcode.isin(["PPLA2", "PPLA3", "PPLA", "PPLC"])].lat), crs=4326)
hq = gpd.sjoin(hq, D[["k", "geometry"]], predicate="within")
cent = D.set_index("k").geometry.representative_point()
hq["dc"] = [p.distance(cent[k]) for p, k in zip(hq.geometry, hq.k)]
hq = hq.sort_values(["k", "fcode", "dc"]).groupby("k").head(1)  # one HQ per district (admin seat code first)
missing = sorted(set(D.k) - set(hq.k))
hq_xy = np.r_[hq[["lon", "lat"]].values, np.array([[cent[k].x, cent[k].y] for k in missing]).reshape(-1, 2)]
print("towns >= 20k:", len(towns), "| district HQ from GeoNames:", len(hq), "| HQ fallback to centroid:", len(missing), flush=True)
T = {}
T["ktm_dry"] = dijkstra(Gd, indices=ktm, directed=False); T["ktm_wet"] = dijkstra(Gw, indices=ktm, directed=False)
for nm, xy in [("town", towns), ("hq", hq_xy), ("hosp", HOSP)]:
    T[nm + "_dry"] = dijkstra(Gd, indices=snap(xy), directed=False, min_only=True)
T["town_wet"] = dijkstra(Gw, indices=snap(towns), directed=False, min_only=True)

POPYEAR = os.environ.get("KH_POPYEAR", "2020")   # weights for district means; 2015 = before the projects (audit variant)
with rasterio.open(RAW / f"worldpop/khm_ppp_{POPYEAR}_1km.tif") as s:
    a = s.read(1); t = s.transform
rr, cc = np.where(a > 0); xs, ys = rasterio.transform.xy(t, rr, cc)
cells = gpd.GeoDataFrame({"pop": a[rr, cc]}, geometry=gpd.points_from_xy(xs, ys), crs=4326)
cells = gpd.sjoin(cells, D[["k", "geometry"]], predicate="within").drop(columns="index_right")
dd, jj = tree.query(np.c_[cells.geometry.x * kx, cells.geometry.y]); node = main_idx[jj]
cells["walk"] = dd * 111.32 * 1.3 / 3.5 * 60
for nm, arr in T.items():
    cells[nm] = arr[node] + cells.walk
cells = cells[np.isfinite(cells.ktm_wet)]
pd.DataFrame({"lon": cells.geometry.x, "lat": cells.geometry.y, "pop": cells["pop"], "k": cells.k, "tt_town_dry": cells.town_dry,
              "tt_ktm_dry": cells.ktm_dry}).to_csv(OUT / f"kh_cells_tt{SUF}.csv", index=False)  # cell-level travel time for maps
wavg = lambda d, c: np.average(d[c], weights=d["pop"])
g = cells.groupby("k")
A = pd.DataFrame({f"tt_{nm}": g.apply(lambda d, c=nm: wavg(d, c), include_groups=False) for nm in T})
A["walk_min"] = g.apply(lambda d: wavg(d, "walk"), include_groups=False)
A["share_walk30"] = g.apply(lambda d: np.average(d.walk <= 30, weights=d["pop"]), include_groups=False)
A["wet_penalty_share"] = A.tt_ktm_wet / A.tt_ktm_dry - 1
E["mx"], E["my"] = (E.x1 + E.x2) / 2, (E.y1 + E.y2) / 2
car = E[E.hw.map(BASE) >= 10]
mp = gpd.sjoin(gpd.GeoDataFrame(car[["km", "paved"]], geometry=gpd.points_from_xy(car.mx, car.my), crs=4326), D[["k", "geometry"]], predicate="within")
A = A.join(mp.groupby("k").apply(lambda d: pd.Series({"road_km": d.km.sum(), "paved_km": d.km[d.paved].sum()}), include_groups=False))
meta = {"driver": "GTiff", "dtype": "float32", "count": 1, "height": z.shape[0], "width": z.shape[1], "transform": tr, "crs": "EPSG:4326", "nodata": -9999}
with rasterio.io.MemoryFile() as mf:
    with mf.open(**meta) as ds:
        ds.write(slope.astype("float32"), 1)
        st, el = [], []
        for geom in D.geometry:
            s_ = mask(ds, [geom], crop=True, filled=True, nodata=-9999)[0][0]; s_ = s_[s_ > -9000]
            st.append(np.mean(s_ > 25)); el.append(np.mean(s_))
A = A.join(pd.DataFrame({"k": D.k, "steep_share": st, "slope_mean": el, "area_km2": D.area_km2}).set_index("k"))
A["road_density"] = A.road_km / A.area_km2; A["paved_share"] = A.paved_km / A.road_km; A["paved_density"] = A.paved_km / A.area_km2

# physical-access index: PC1 of signed, standardised components (higher = better access)
comp = pd.DataFrame({"neg_ln_tt_town_dry": -np.log(A.tt_town_dry), "neg_ln_tt_town_wet": -np.log(A.tt_town_wet), "neg_ln_tt_hq": -np.log(A.tt_hq_dry),
                     "neg_ln_tt_hosp": -np.log(A.tt_hosp_dry), "neg_ln_walk": -np.log(A.walk_min), "share_walk30": A.share_walk30,
                     "ln_road_density": np.log(A.road_density), "paved_share": A.paved_share, "neg_wet_penalty": -A.wet_penalty_share,
                     "neg_steep_share": -A.steep_share}).replace([np.inf, -np.inf], np.nan)
if CLASS_ONLY:
    comp = comp.drop(columns="paved_share")
comp = comp.fillna(comp.median())
Zc = (comp - comp.mean()) / comp.std()
w, V = np.linalg.eigh(np.cov(Zc.T)); pc = V[:, -1] * np.sign(V[:, -1].sum())
A["access_pc1"] = Zc.values @ pc; A["access_pc1"] = (A.access_pc1 - A.access_pc1.mean()) / A.access_pc1.std()
A["neg_ln_tt_ktm"] = -np.log(A.tt_ktm_dry)
A.reset_index().to_csv(OUT / f"kh_access{SUF}.csv", index=False)
pd.DataFrame({"component": comp.columns, "loading": pc.round(3), "pc1_var_share": round(w[-1] / w.sum(), 3)}).to_csv(ROOT / f"outputs/tables/kh_access_pca{SUF}.csv", index=False)
print("PC1 var share", round(w[-1] / w.sum(), 2), dict(zip(comp.columns, pc.round(2))))
print(A[["tt_ktm_dry", "tt_town_dry", "tt_hq_dry", "tt_hosp_dry", "share_walk30", "paved_share", "access_pc1"]].describe().round(2).to_string())
