"""Audit of the physical-access inputs (2017 network): destinations, surface tags, grades and slope.

Checks, with the reason each matters:
  1. Towns >= 20k (GeoNames): list them, flag duplicates and sub-city sections.
  2. District seats: how many come from GeoNames and how far the centroid fallback would be from a real seat.
  3. Hospitals (OSM 2017): count by province, provinces without one.
  4. Surface tags: share of road length tagged by class, and what the untagged default does to each class.
  5. Road grade: grade is |dz| / length between consecutive nodes; short segments that cross a DEM cell edge can get
     spurious grades. Share of road length slowed by the grade factor, in flat and hilly districts.
  6. Terrain slope: distribution and steep share by district.
Output: outputs/tables/kh_access_audit.txt
"""
import numpy as np, pandas as pd, geopandas as gpd, osmium, rasterio
from rasterio.merge import merge
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data/raw/cambodia", ROOT / "data/processed/cambodia"
D = gpd.read_file(OUT / "kh_units.gpkg").rename(columns={"GEOLEVEL2": "k"})
log = []
say = lambda *a: (print(*a), log.append(" ".join(str(x) for x in a)))
CLASSES = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "track", "service", "path"]


class H(osmium.SimpleHandler):
    def __init__(self):
        super().__init__(); self.e = []; self.hosp = []

    def node(self, n):
        if n.tags.get("amenity") == "hospital" or n.tags.get("healthcare") == "hospital":
            self.hosp.append((n.location.lon, n.location.lat, n.tags.get("name", "")))

    def way(self, w):
        if w.tags.get("amenity") == "hospital":
            try:
                self.hosp.append((w.nodes[0].lon, w.nodes[0].lat, w.tags.get("name", "")))
            except Exception:
                pass
        hw = w.tags.get("highway")
        if hw is None:
            return
        try:
            ns = [(n.lon, n.lat) for n in w.nodes]
        except osmium.InvalidLocationError:
            return
        for (x1, y1), (x2, y2) in zip(ns[:-1], ns[1:]):
            self.e.append((x1, y1, x2, y2, hw.replace("_link", ""), w.tags.get("surface")))


h = H(); h.apply_file(str(RAW / "osm/cambodia-170101.osm.pbf"), locations=True)
E = pd.DataFrame(h.e, columns=["x1", "y1", "x2", "y2", "hw", "surf"]); HO = pd.DataFrame(h.hosp, columns=["lon", "lat", "name"])
E["km"] = np.hypot((E.x2 - E.x1) * 111.32 * np.cos(np.radians((E.y1 + E.y2) / 2)), (E.y2 - E.y1) * 110.57)
E = E[(E.km > 0) & E.hw.isin(CLASSES)].reset_index(drop=True)

# ---- 1. towns
cols = ["geonameid", "name", "asciiname", "alt", "lat", "lon", "fclass", "fcode", "cc", "cc2", "a1", "a2", "a3", "a4", "pop", "elev", "dem", "tz", "mod"]
GN = pd.read_csv(RAW / "geonames/KH.txt", sep="\t", names=cols, dtype={"pop": float}, quoting=3, low_memory=False)
GN = GN[GN.fclass == "P"]
T = GN[GN["pop"] >= 20000].sort_values("pop", ascending=False)
say("== 1. towns >= 20k:", len(T)); say(T[["name", "fcode", "pop", "lat", "lon"]].to_string(index=False))
xy = T[["lon", "lat"]].values
dmin = [np.sort(np.hypot((xy[:, 0] - x) * 109, (xy[:, 1] - y) * 110.6))[1] for x, y in xy]
say("towns within 3 km of another town (sections of one city):", int(np.sum(np.array(dmin) < 3)))

# ---- 2. district seats
codes = ["PPLA2", "PPLA3", "PPLA", "PPLC"]
hq = gpd.GeoDataFrame(GN[GN.fcode.isin(codes)], geometry=gpd.points_from_xy(GN[GN.fcode.isin(codes)].lon, GN[GN.fcode.isin(codes)].lat), crs=4326)
hq = gpd.sjoin(hq, D[["k", "geometry"]], predicate="within")
cent = D.set_index("k").geometry.representative_point()
hq["dc_km"] = [p.distance(cent[k]) * 110 for p, k in zip(hq.geometry, hq.k)]
first = hq.sort_values(["k", "fcode", "dc_km"]).groupby("k").head(1)
say("\n== 2. districts with a GeoNames seat:", len(first), "of", len(D), "| fallback to centroid:", len(D) - len(first))
say("distance from real seat to the district's representative point, km (where both exist): median",
    round(first.dc_km.median(), 1), "| p75", round(first.dc_km.quantile(.75), 1), "| max", round(first.dc_km.max(), 1))
# could a populated place named after the district fill the gap?
miss = D[~D.k.isin(first.k)]
gp = gpd.GeoDataFrame(GN, geometry=gpd.points_from_xy(GN.lon, GN.lat), crs=4326)
hits = 0
for _, r in miss.iterrows():
    parts = [p.strip().lower() for p in str(r.get("ADMIN_NAME", "")).replace("-", ",").replace("–", ",").split(",") if p.strip()]
    inside = gp[gp.within(r.geometry)]
    hits += int(inside.asciiname.str.lower().isin(parts).any())
say("missing districts with a populated place whose name matches the district name:", hits, "of", len(miss))

# ---- 3. hospitals
hg = gpd.sjoin(gpd.GeoDataFrame(HO, geometry=gpd.points_from_xy(HO.lon, HO.lat), crs=4326), D[["k", "PARENT", "geometry"]] if "PARENT" in D else D[["k", "geometry"]], predicate="within")
say("\n== 3. hospital features:", len(HO), "| distinct within 200 m:", len(HO.round(3).drop_duplicates(["lon", "lat"])),
    "| districts with >= 1:", hg.k.nunique(), "of", len(D))
if "PARENT" in hg:
    pc = hg.groupby("PARENT").size(); allp = D.PARENT.unique()
    say("provinces without a hospital in OSM 2017:", sorted(set(allp) - set(pc.index)))

# ---- 4. surface tags
E["tagged"] = E.surf.notna()
s = E.groupby("hw").apply(lambda d: pd.Series({"km": d.km.sum(), "tagged_share": d.km[d.tagged].sum() / d.km.sum(),
                                               "paved_if_tagged": d.km[d.surf.isin(["asphalt", "concrete", "paved", "chipseal", "paving_stones"])].sum() / max(d.km[d.tagged].sum(), 1e-9)}),
                          include_groups=False).loc[[c for c in CLASSES if c in E.hw.unique()]]
say("\n== 4. surface tags by class (untagged default: paved for motorway/trunk/primary, unpaved otherwise)"); say(s.round(3).to_string())

# ---- 5/6. grade and slope
srcs = [rasterio.open(f) for f in sorted((RAW / "dem").glob("*.tif"))]
dem, tr = merge(srcs, res=(1 / 600, 1 / 600)); [q.close() for q in srcs]
z = dem[0].astype("float32")
ix = lambda x, y: (np.clip(((y - tr.f) / tr.e).astype(int), 0, z.shape[0] - 1), np.clip(((x - tr.c) / tr.a).astype(int), 0, z.shape[1] - 1))
E["grade"] = np.clip(np.abs(z[ix(E.x2, E.y2)] - z[ix(E.x1, E.y1)]) / (E.km * 1000), 0, 0.4)
E["slow"] = np.clip(1 - 2 * E.grade, 0.4, 1)
latc = tr.f + tr.e * (np.arange(z.shape[0]) + 0.5)
gy, gx = np.gradient(z)
slope = np.degrees(np.arctan(np.hypot(gx / (tr.a * 111320 * np.cos(np.radians(latc)))[:, None], gy / (abs(tr.e) * 110570))))
E["terrain"] = slope[ix((E.x1 + E.x2) / 2, (E.y1 + E.y2) / 2)]
w = E.km
say("\n== 5. road grade (DEM cell ~185 m)")
say("segment length, m: median", round(E.km.median() * 1000), "| share of length in segments < 185 m:", round(w[E.km < .185].sum() / w.sum(), 3))
say("share of road length slowed by >10% (grade > 5%):", round(w[E.slow < .9].sum() / w.sum(), 3))
flat = E.terrain < 2
say("  on flat terrain (slope < 2 deg):", round(w[flat & (E.slow < .9)].sum() / w[flat].sum(), 3),
    "| on segments < 185 m:", round(w[flat & (E.slow < .9) & (E.km < .185)].sum() / max(w[flat & (E.slow < .9)].sum(), 1e-9), 3), "of that length")
say("length-weighted mean speed factor from grade: all", round(np.average(E.slow, weights=w), 3), "| flat", round(np.average(E.slow[flat], weights=w[flat]), 3))
say("\n== 6. terrain slope: share of land > 25 deg", round(float(np.mean(slope[z > -100] > 25)), 4),
    "| p99 slope", round(float(np.percentile(slope[z > -100], 99)), 1), "deg")
Path(ROOT / "outputs/tables/kh_access_audit.txt").write_text("\n".join(log), encoding="utf-8")
