"""Major national-road projects of donors missing from GODAD (ADB, Japan), coded by hand from official documents.

Each project is a road number and a section given by two end towns. The section is traced on the 2017 OSM network as the
shortest path along ways carrying that road number (ref tag), and every district the path crosses is treated. The year is
the loan approval or grant/loan agreement year, matching how GODAD dates activities by first commitment.
Sources (all checked 2026-11-03):
  ADB 1659  Phnom Penh-Ho Chi Minh City Highway, approved 15 Dec 1998 (ADB approved-loans list; IED evaluation). Cambodian
            section coded as NR1 Neak Loeung-Bavet, the part east of the Mekong not covered by Japan's grant (assumption).
  ADB 1697  Primary Roads Restoration, approved Sep 1999 (IED PPER 2009): NR5 Pursat-Battambang, NR6 Kampong Thom-Siem Reap
            provincial border (coded to Stoung), NR7 Tonle Bet-Kratie.
  ADB 1945  Road Improvement, approved Nov 2002 (ADB approved-loans list): NR5/NR6 Siem Reap-Poipet.
  ADB 2539  GMS Northwest? (PCR 42358-013), approved 29 Aug 2009: NR56 Sisophon-Samraong.
  Japan     NR1 Phnom Penh-Neak Loeung grant, first agreement 2005 (Embassy of Japan ODA map; JICA ex-post evaluation 2020).
  Japan     Neak Loeung Bridge grant, agreement June 2010 (JICA ex-post evaluation 2020): districts within 3 km of Neak Loeung.
  JICA      NR5 Battambang-Sri Sophorn (Sisophon) loan, phase I May 2013 (JICA ex-ante evaluation 2017).
  JICA      NR5 Prek Kdam-Thlea Ma'am loan, phase I 10 Jul 2014 (JICA loan agreement).
  JICA      NR5 Thlea Ma'am-Battambang and Sri Sophorn-Poipet loan, phase I 30 Mar 2015 (JICA ex-ante evaluation).
Not coded: ADB 2373 (15 km of NR33), ADB 3576 (2017, sections not identified), smaller rural-road programmes.
Output: data/processed/cambodia/kh_major_roads_units.csv
"""
import re, numpy as np, pandas as pd, geopandas as gpd, osmium
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from shapely.geometry import LineString
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW, PROCD = ROOT / "data/raw/cambodia", ROOT / "data/processed/cambodia"
PROJECTS = [  # (project, donor, year, [(road number, from town, to town), ...])
    ("ADB 1659 Phnom Penh-HCMC Highway", "ADB", 1998, [("1", "Neak Loeung", "Bavet")]),
    ("ADB 1697 Primary Roads Restoration", "ADB", 1999, [("5", "Pursat", "Battambang"), ("6", "Kampong Thom", "Stoung"), ("7", "Tonle Bet", "Kratie")]),
    ("ADB 1945 Road Improvement", "ADB", 2002, [("6", "Siem Reap", "Sisophon"), ("5", "Sisophon", "Poipet")]),
    ("Japan NR1 Phnom Penh-Neak Loeung", "Japan", 2005, [("1", "Phnom Penh", "Neak Loeung")]),
    ("ADB 2539 NR56", "ADB", 2009, [("56", "Sisophon", "Samraong")]),
    ("Japan Neak Loeung Bridge", "Japan", 2010, [("bridge", "Neak Loeung", None)]),
    ("JICA NR5 Battambang-Sisophon", "Japan", 2013, [("5", "Battambang", "Sisophon")]),
    ("JICA NR5 Prek Kdam-Thlea Ma'am", "Japan", 2014, [("5", "Prek Kdam", "Thlea Ma'am")]),
    ("JICA NR5 Thlea Ma'am-Battambang, Sisophon-Poipet", "Japan", 2015, [("5", "Thlea Ma'am", "Battambang"), ("5", "Sisophon", "Poipet")]),
]
TOWNS = {"Neak Loeung": ["Neak Loeung", "Neak Luong", "Neak Leung", "Nak Loeung"], "Bavet": ["Bavet"], "Pursat": ["Pursat"],
         "Battambang": ["Battambang"], "Kampong Thom": ["Kampong Thom"], "Stoung": ["Stoung", "Stung"], "Tonle Bet": ["Tonle Bet", "Tonlé Bet"],
         "Kratie": ["Kratie", "Kratié"], "Siem Reap": ["Siem Reap"], "Sisophon": ["Sisophon", "Serei Saophoan"], "Poipet": ["Paoy Paet", "Poipet"],
         "Phnom Penh": ["Phnom Penh"], "Samraong": ["Samraong"]}

cols = ["geonameid", "name", "asciiname", "alt", "lat", "lon", "fclass", "fcode", "cc", "cc2", "a1", "a2", "a3", "a4", "pop", "elev", "dem", "tz", "mod"]
GN = pd.read_csv(RAW / "geonames/KH.txt", sep="\t", names=cols, quoting=3, low_memory=False)
GN = GN[GN.fclass == "P"].copy(); GN["pop"] = pd.to_numeric(GN["pop"], errors="coerce").fillna(0)
LOC = {}
for t, names in TOWNS.items():
    pat = "|".join(re.escape(n.lower()) for n in names)
    hit = GN[GN.name.str.lower().str.fullmatch(pat) | GN.asciiname.str.lower().str.fullmatch(pat)]
    if hit.empty:
        hit = GN[GN.alt.fillna("").str.lower().str.contains(pat)]
    hit = hit.sort_values(["pop", "fcode"], ascending=[False, True])
    LOC[t] = (float(hit.iloc[0].lon), float(hit.iloc[0].lat)) if len(hit) else None
    print(f"{t:<13s} {LOC[t]}  ({hit.iloc[0]['name'] if len(hit) else 'NOT FOUND'})")
# not in GeoNames: Prek Kdam is the NR5 crossing of the Tonle Sap north of Phnom Penh (approximate point, snapped to NR5);
# Thlea Ma'am is placed 118.7 km along NR5 from Prek Kdam, the section length in JICA's ex-ante evaluation (2016)
LOC["Prek Kdam"] = (104.80, 11.82)


class H(osmium.SimpleHandler):
    def __init__(self):
        super().__init__(); self.e = []

    def way(self, w):
        ref, hw = w.tags.get("ref"), w.tags.get("highway")
        if hw not in {"motorway", "trunk", "primary", "secondary", "tertiary", "trunk_link", "primary_link", "secondary_link"}:
            return
        nums = set(re.findall(r"\d+[A-Z]?", (ref or "").upper()))   # untagged major roads are kept to bridge gaps
        try:
            ns = [(n.ref, n.lon, n.lat) for n in w.nodes]
        except osmium.InvalidLocationError:
            return
        for (a, x1, y1), (b, x2, y2) in zip(ns[:-1], ns[1:]):
            self.e.append((a, b, x1, y1, x2, y2, ",".join(sorted(nums))))


h = H(); h.apply_file(str(RAW / "osm/cambodia-170101.osm.pbf"), locations=True)
E = pd.DataFrame(h.e, columns=["a", "b", "x1", "y1", "x2", "y2", "refs"])
E["km"] = np.hypot((E.x2 - E.x1) * 109.0, (E.y2 - E.y1) * 110.6)
U = gpd.read_file(PROCD / "kh_units.gpkg"); U["unit"] = U.GEOLEVEL2.astype("int64")


def trace(num, t0, t1, cut_km=None):
    """Shortest path that follows the road number; untagged or differently numbered major-road stretches cost 15 times
    their length, so short gaps in the ref tags are bridged but other roads are not used as shortcuts."""
    sub = E.copy(); on = sub.refs.str.split(",").apply(lambda r: num in r)
    sub["cost"] = np.where(on, sub.km, sub.km * 15)
    ids = pd.Index(pd.unique(np.r_[sub.a, sub.b])); ai, bi = ids.get_indexer(sub.a), ids.get_indexer(sub.b)
    G = coo_matrix((np.r_[sub.cost, sub.cost], (np.r_[ai, bi], np.r_[bi, ai])), shape=(len(ids), len(ids))).tocsr()
    G.sum_duplicates()
    xy = pd.concat([pd.DataFrame({"n": sub.a, "x": sub.x1, "y": sub.y1}), pd.DataFrame({"n": sub.b, "x": sub.x2, "y": sub.y2})]).drop_duplicates("n").set_index("n").loc[ids]
    tree = cKDTree(np.c_[xy.x * 109.0, xy.y * 110.6])
    (d0, i0), (d1, i1) = tree.query([LOC[t0][0] * 109.0, LOC[t0][1] * 110.6]), tree.query([LOC[t1][0] * 109.0, LOC[t1][1] * 110.6])
    dist, pred = dijkstra(G, indices=i0, return_predecessors=True, directed=False)
    if not np.isfinite(dist[i1]):
        return None, f"NR{num} {t0}-{t1}: no connected path", None
    path = [i1]
    while path[-1] != i0:
        path.append(pred[path[-1]])
    pts = xy.iloc[path[::-1]][["x", "y"]].values
    seg = np.r_[0, np.cumsum(np.hypot(np.diff(pts[:, 0]) * 109.0, np.diff(pts[:, 1]) * 110.6))]
    if cut_km is not None and seg[-1] > cut_km:
        pts = pts[seg <= cut_km]; seg = seg[seg <= cut_km]
    return LineString(pts), f"NR{num} {t0}-{t1}: {seg[-1]:.0f} km traced (ends snapped {d0:.1f} / {d1:.1f} km from towns)", tuple(pts[-1])


_, msg, LOC["Thlea Ma'am"] = trace("5", "Prek Kdam", "Pursat", cut_km=118.7)
print("Thlea Ma'am placed at", tuple(round(v, 4) for v in LOC["Thlea Ma'am"]), "|", msg)
rows = []
for proj, donor, year, sections in PROJECTS:
    for num, t0, t1 in sections:
        if num == "bridge":
            geom = gpd.GeoSeries(gpd.points_from_xy([LOC[t0][0]], [LOC[t0][1]]), crs=4326).to_crs(32648).buffer(3000).to_crs(4326).iloc[0]; msg = f"bridge at {t0} (3 km)"
        else:
            geom, msg, _ = trace(num, t0, t1)
        print(f"{proj:<52s} {msg}")
        if geom is None:
            continue
        hit = U[U.intersects(geom)]
        for u in hit.unit:
            rows.append({"project": proj, "donor": donor, "start_year": year, "section": f"{num}:{t0}-{t1}", "unit": u})
R = pd.DataFrame(rows).drop_duplicates(["project", "unit"])
R.to_csv(PROCD / "kh_major_roads_units.csv", index=False)
print(R.groupby("project").unit.nunique().to_string(), "\ndistrict-project pairs:", len(R), "| distinct districts:", R.unit.nunique())
