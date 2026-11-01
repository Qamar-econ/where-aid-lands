"""Physical accessibility: OSM road graph -> population-weighted travel time to Dili per posto.

* Speeds by road class x surface; terrain grade penalty; wet-season penalty for unpaved
  and steep (landslide-prone) segments.
* Off-network walking from each populated 1 km WorldPop cell to its nearest road node.
* Oecusse (enclave) and Atauro (island) reached by sea from Dili.
* Time-varying network 2005-2025: major donor-financed corridors run at "pre-upgrade"
  speed before their completion year.

Outputs: data/processed/posto_access.csv, posto_access_panel.csv, roads.gpkg
"""
import math
import numpy as np, pandas as pd, geopandas as gpd, rasterio, networkx as nx, osmium
from scipy.spatial import cKDTree
from shapely.geometry import LineString
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW, INT, PROC = ROOT / "data" / "raw", ROOT / "data" / "interim", ROOT / "data" / "processed"

BASE = {"motorway": 60, "trunk": 60, "primary": 50, "secondary": 40, "tertiary": 30, "unclassified": 25,
        "residential": 20, "living_street": 15, "service": 15, "road": 20, "track": 12,
        "path": 4, "footway": 4, "bridleway": 4, "steps": 3, "pedestrian": 4}
for k in list(BASE):
    BASE[k + "_link"] = BASE[k]
PAVED = {"asphalt", "concrete", "paved", "paving_stones", "concrete:plates", "chipseal", "sett", "metal"}
UNPAVED = {"unpaved", "gravel", "dirt", "ground", "earth", "compacted", "mud", "sand", "fine_gravel", "grass", "rock", "pebblestone"}
DEFAULT_PAVED = {"motorway", "trunk", "primary", "secondary", "motorway_link", "trunk_link", "primary_link", "secondary_link"}


class H(osmium.SimpleHandler):
    def __init__(self):
        super().__init__(); self.ways = []

    def way(self, w):
        hw = w.tags.get("highway")
        if hw not in BASE:
            return
        try:
            coords = [(n.lon, n.lat, n.ref) for n in w.nodes]
        except osmium.InvalidLocationError:
            return
        self.ways.append((w.id, hw, w.tags.get("surface"), w.tags.get("ref"), w.tags.get("name"), coords))


h = H()
h.apply_file(str(RAW / "osm/east-timor-latest.osm.pbf"), locations=True)
print("ways", len(h.ways))


def hav(lon1, lat1, lon2, lat2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


dem = rasterio.open(INT / "dem.tif"); slp = rasterio.open(INT / "slope.tif")

edges, geoms = [], []
for wid, hw, surf, ref, name, cs in h.ways:
    if surf in PAVED:
        paved = True
    elif surf in UNPAVED:
        paved = False
    else:
        paved = hw in DEFAULT_PAVED
    for (x1, y1, n1), (x2, y2, n2) in zip(cs[:-1], cs[1:]):
        edges.append((n1, n2, x1, y1, x2, y2, hw, paved, ref, wid))
    geoms.append({"way_id": wid, "highway": hw, "surface": surf, "paved": paved, "ref": ref, "name": name,
                  "geometry": LineString([(x, y) for x, y, _ in cs])})
E = pd.DataFrame(edges, columns=["u", "v", "x1", "y1", "x2", "y2", "hw", "paved", "ref", "wid"])
E["km"] = [hav(a, b, c, d) for a, b, c, d in zip(E.x1, E.y1, E.x2, E.y2)]
E = E[E.km > 0].reset_index(drop=True)
z1 = np.array([v[0] for v in dem.sample(list(zip(E.x1, E.y1)))], float)
z2 = np.array([v[0] for v in dem.sample(list(zip(E.x2, E.y2)))], float)
E["grade"] = np.clip(np.abs(z2 - z1) / (E.km * 1000), 0, 0.3)
E["local_slope"] = np.array([v[0] for v in slp.sample(list(zip((E.x1 + E.x2) / 2, (E.y1 + E.y2) / 2)))], float)

v = E.hw.map(BASE).astype(float)
v = np.where(E.paved, v, np.minimum(v, 25) * 0.7)            # unpaved: capped and slower
v = v * np.clip(1 - 2.0 * E.grade, 0.5, 1)                    # winding/steep alignment
E["v_dry"] = v
wet = np.where(E.paved, 0.85, 0.5)                              # rainy season: unpaved roads deteriorate
wet = wet * np.where(E.local_slope > 25, 0.7, 1.0)              # landslide-prone segments
E["v_wet"] = v * wet
gpd.GeoDataFrame(geoms, crs=4326).to_file(PROC / "roads.gpkg", driver="GPKG")

# ---- corridors upgraded by donor-financed projects (completion year) ----
CORR = [  # name, financier, year, waypoints (lon, lat)
    ("Dili-Tibar-Liquica", "ADB", 2016, [(125.578, -8.556), (125.470, -8.578), (125.341, -8.587)]),
    ("Tibar-Gleno", "ADB", 2016, [(125.470, -8.578), (125.437, -8.722)]),
    ("Manatuto-Natarbora", "ADB", 2018, [(126.012, -8.511), (126.020, -8.820), (126.040, -9.010)]),
    ("Dili-Manatuto-Baucau", "ADB/JICA", 2020, [(125.578, -8.556), (126.012, -8.511), (126.455, -8.468)]),
    ("Dili-Aileu-Maubisse-Ainaro", "World Bank", 2020, [(125.578, -8.556), (125.566, -8.727), (125.599, -8.839), (125.507, -8.993)]),
    ("Baucau-Lautem-Lospalos", "ADB", 2021, [(126.455, -8.468), (126.900, -8.360), (127.000, -8.520)]),
]

G = nx.Graph()
for r in E.itertuples():
    G.add_edge(r.u, r.v, km=r.km, i=r.Index, t_dry=r.km / r.v_dry * 60, t_wet=r.km / r.v_wet * 60,
               major=r.hw in DEFAULT_PAVED or r.hw == "tertiary")
nodes = pd.concat([E[["u", "x1", "y1"]].set_axis(["n", "x", "y"], axis=1), E[["v", "x2", "y2"]].set_axis(["n", "x", "y"], axis=1)]).drop_duplicates("n")
nodes = nodes.set_index("n")
tree = cKDTree(np.c_[nodes.x.values * np.cos(np.radians(-8.8)), nodes.y.values])
nid = nodes.index.values


def nearest(lon, lat):
    d, j = tree.query([lon * np.cos(np.radians(-8.8)), lat])
    return nid[j], d * 111.32


Gm = G.edge_subgraph([(a, b) for a, b, d in G.edges(data=True) if d["major"]]).copy()
corr_edges = {}
for name, fin, yr, wps in CORR:
    ids = set()
    for (a, b) in zip(wps[:-1], wps[1:]):
        na = min(Gm.nodes, key=lambda n: (nodes.x[n] - a[0]) ** 2 + (nodes.y[n] - a[1]) ** 2)
        nb = min(Gm.nodes, key=lambda n: (nodes.x[n] - b[0]) ** 2 + (nodes.y[n] - b[1]) ** 2)
        path = nx.shortest_path(Gm, na, nb, weight="km")
        ids |= {G[p][q]["i"] for p, q in zip(path[:-1], path[1:])}
    corr_edges[name] = (yr, fin, ids)
    print(f"corridor {name}: {sum(E.km[list(ids)]):.1f} km")
cg = []
for name, (yr, fin, ids) in corr_edges.items():
    sub = E.loc[list(ids)]
    cg.append({"corridor": name, "financier": fin, "year": yr,
               "geometry": gpd.GeoSeries([LineString([(a, b), (c, d)]) for a, b, c, d in zip(sub.x1, sub.y1, sub.x2, sub.y2)]).union_all()})
gpd.GeoDataFrame(cg, crs=4326).to_file(PROC / "corridors.gpkg", driver="GPKG")

# ---- origins: populated 1 km cells ----
po = gpd.read_file(PROC / "posto_covariates.gpkg")
with rasterio.open(RAW / "worldpop/tls_ppp_2020_1km_UNadj.tif") as src:
    a = src.read(1); tr = src.transform
rr, cc = np.where(a > 0)
xs, ys = rasterio.transform.xy(tr, rr, cc)
cells = gpd.GeoDataFrame({"pop": a[rr, cc]}, geometry=gpd.points_from_xy(xs, ys), crs=4326)
cells = gpd.sjoin(cells, po[["pcode", "posto", "municipio", "geometry"]], predicate="within").drop(columns="index_right")
# snap origins only to nodes of the three main connected networks (mainland, Oecusse, Atauro)
_main = set()
for lonlat in [(125.5789, -8.5569), (124.365, -9.198), (125.605, -8.268)]:
    _main |= nx.node_connected_component(G, nearest(*lonlat)[0])
nodes = nodes.loc[[n for n in nodes.index if n in _main]]
tree = cKDTree(np.c_[nodes.x.values * np.cos(np.radians(-8.8)), nodes.y.values])
nid = nodes.index.values
snap = [nearest(p.x, p.y) for p in cells.geometry]
cells["node"] = [s[0] for s in snap]
cells["walk_min"] = np.array([s[1] for s in snap]) * 1.3 / 4.0 * 60   # 4 km/h, 1.3 detour

DILI = nearest(125.5789, -8.5569)[0]
PORT_OE, PORT_AT, PORT_DILI = nearest(124.365, -9.198)[0], nearest(125.605, -8.268)[0], nearest(125.573, -8.553)[0]
SEA = {"Oecussi": (PORT_OE, 12 * 60), "Atauro": (PORT_AT, 2.5 * 60)}   # ferry (Berlin Nakroma) / Atauro boat


def run(weight, override=None):
    """Return minutes to Dili for each cell; override = {edge_index: factor}"""
    if override:
        for a, b, d in G.edges(data=True):
            f = override.get(d["i"])
            d["_w"] = d[weight] * f if f else d[weight]
        w = "_w"
    else:
        w = weight
    dist = nx.single_source_dijkstra_path_length(G, DILI, weight=w)
    side = {}
    for key, (port, sea) in SEA.items():
        dport = dist.get(PORT_DILI, 0)
        sub = nx.single_source_dijkstra_path_length(G, port, weight=w)
        side[key] = (sub, dport + sea)
    out = np.empty(len(cells))
    for k, (n, wm, mun, pst) in enumerate(zip(cells.node, cells.walk_min, cells.municipio, cells.posto)):
        key = "Oecussi" if mun == "Oecussi" else ("Atauro" if pst == "Atauro" else None)
        if key:
            sub, base = side[key]
            out[k] = base + sub.get(n, np.nan) + wm
        else:
            out[k] = dist.get(n, np.nan) + wm
    return out


cells["tt_dry"] = run("t_dry")
cells["tt_wet"] = run("t_wet")
print("unreached cells:", np.isnan(cells.tt_dry).sum(), "pop", cells.loc[np.isnan(cells.tt_dry), "pop"].sum())
cells = cells[~np.isnan(cells.tt_wet)]


def wmean(d, col):
    return np.average(d[col], weights=d["pop"])


acc = cells.groupby("pcode").apply(lambda d: pd.Series({
    "tt_dry_min": wmean(d, "tt_dry"), "tt_wet_min": wmean(d, "tt_wet"),
    "walk_min": wmean(d, "walk_min"),
    "share_pop_2km_road": np.average(d.walk_min <= 2 * 1.3 / 4 * 60, weights=d["pop"])}), include_groups=False).reset_index()
acc["wet_penalty_min"] = acc.tt_wet_min - acc.tt_dry_min

# road quality / density per posto
rd = gpd.read_file(PROC / "roads.gpkg").to_crs(32751)
pu = po.to_crs(32751)
inter = gpd.overlay(rd[["highway", "paved", "geometry"]], pu[["pcode", "geometry"]], how="intersection", keep_geom_type=True)
inter["km"] = inter.length / 1000
car = inter[inter.highway.isin([k for k in BASE if BASE[k] >= 12])]
rq = car.groupby("pcode").apply(lambda d: pd.Series({"road_km": d.km.sum(), "paved_km": d.km[d.paved].sum()}),
                                include_groups=False).reset_index()
acc = acc.merge(rq, on="pcode", how="left").merge(po[["pcode", "area_km2"]], on="pcode")
acc["road_density"] = acc.road_km / acc.area_km2
acc["paved_share"] = acc.paved_km / acc.road_km
acc.drop(columns="area_km2").to_csv(PROC / "posto_access.csv", index=False)

# ---- time-varying travel time panel 2005-2025 (pre-upgrade corridor speed = 22 km/h dry, 15 wet) ----
rows = []
years = list(range(2005, 2026))
cache = {}
for y in years:
    pending = frozenset(n for n, (yr, f, ids) in corr_edges.items() if y < yr)
    if pending not in cache:
        ov_d, ov_w = {}, {}
        for n in pending:
            for i in corr_edges[n][2]:
                ov_d[i] = max(1.0, E.v_dry[i] / 22.0)
                ov_w[i] = max(1.0, E.v_wet[i] / 15.0)
        td, tw = run("t_dry", ov_d), run("t_wet", ov_w)
        c2 = cells.assign(tdy=td, twy=tw)
        cache[pending] = c2.groupby("pcode").apply(lambda d: pd.Series({
            "tt_dry_min": np.average(d.tdy, weights=d["pop"]), "tt_wet_min": np.average(d.twy, weights=d["pop"])}),
            include_groups=False).reset_index()
    rows.append(cache[pending].assign(year=y))
panel = pd.concat(rows)
panel.to_csv(PROC / "posto_access_panel.csv", index=False)
chg = panel[panel.year == 2005].merge(panel[panel.year == 2025], on="pcode", suffixes=("_2005", "_2025"))
chg["gain_wet_min"] = chg.tt_wet_min_2005 - chg.tt_wet_min_2025
print(chg.merge(po[["pcode", "posto"]]).sort_values("gain_wet_min", ascending=False)[["posto", "tt_wet_min_2005", "tt_wet_min_2025", "gain_wet_min"]].head(25).round(0).to_string())
print(acc.merge(po[["pcode", "posto"]]).sort_values("tt_wet_min")[["posto", "tt_dry_min", "tt_wet_min", "paved_share", "road_density"]].round(2).to_string())
