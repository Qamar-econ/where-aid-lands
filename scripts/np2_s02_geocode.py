"""Apply geocoder v2 to Nepal development projects starting 2017-2025 (IATI + World Bank), combining
(1) IATI coordinates (precision 1/2, non-generic), (2) text of title/description/full IATI record, (3) harvested documents
(data/interim/np2_doctext/<uid>.txt, if present). Output: data/processed/nepal/np2_geocodes.csv (project x unit rows)
and np2_project_status.csv (best precision per project).
"""
import json, re, sys
import numpy as np, pandas as pd, geopandas as gpd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from np2_geo import geocode_text, best_by_district  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INV = pd.read_csv(ROOT / "data/interim/np2_inventory.csv")
INV = INV[INV.dev]
ft = pd.read_csv(ROOT / "data/interim/iati_fulltext_NP.csv").drop_duplicates("aid").set_index("aid").fillna("")
DOCS = ROOT / "data/interim/np2_doctext"
loc = pd.DataFrame(json.loads((ROOT / "data/raw/nepal/iati/dportal_loc_NP.json").read_text(encoding="utf-8")))
loc = loc[loc.location_longitude.notna()]
xy = loc.location_longitude.round(2).astype(str) + loc.location_latitude.round(2).astype(str)
loc = loc[xy.map(loc.assign(xy=xy).groupby("xy").aid.nunique()) < 40]
P3 = gpd.read_file(ROOT / "data/raw/nepal/cod-ab-npl/npl_admin_boundaries.shp/npl_admin3.shp")[["adm3_pcode", "geometry"]]
Dk = gpd.read_file(ROOT / "data/processed/nepal/districts.gpkg")[["k", "geometry"]]
lg = gpd.GeoDataFrame(loc, geometry=gpd.points_from_xy(loc.location_longitude, loc.location_latitude), crs=4326)
lg = gpd.sjoin(gpd.sjoin(lg, P3, predicate="within").drop(columns="index_right"), Dk, predicate="within")
COORD = {a: d for a, d in lg.groupby("aid")}
CAPITAL = {"kathmandu", "lalitpur", "bhaktapur"}
ADB = json.loads((ROOT / "data/interim/adb_pds_milestones.json").read_text(encoding="utf-8"))

rows, status = [], []
for r in INV.itertuples():
    aid = r.uid[5:] if r.uid.startswith("IATI-") else None
    hits = []
    if aid in COORD:
        for c in COORD[aid].itertuples():
            p = int(c.location_precision) if pd.notna(c.location_precision) else 2
            hits.append({"level": "point", "unit": c.adm3_pcode, "district": c.k, "precision": 1 if p == 1 else 2,
                         "term": str(c.location_name), "snippet": "IATI coordinate", "source": "iati_coord"})
    text = f"{r.title} {r.desc if isinstance(r.desc, str) else ''}"
    if aid:
        text += f" {ft.text.get(aid, '')}"
    th = geocode_text(text)
    # V3b: capital-valley names that appear only in IATI structured fields (location/recipient-region/contact) and not in
    # the title or description are treated as HQ defaults
    core = {h["district"] for h in geocode_text(f"{r.title} {r.desc if isinstance(r.desc, str) else ''}")}
    th = [h for h in th if h["district"] not in CAPITAL or h["district"] in core]
    for h in th:
        h["source"] = "iati_text" if aid else "wb_text"
    hits += th
    m = re.search(r"46004-(\d{5}-\d{3})", r.uid)
    if m and ADB.get(m.group(1)) and ADB[m.group(1)][3]:
        ah = geocode_text(ADB[m.group(1)][3])  # ADB Project Data Sheet "Geographical Location" field
        for h in ah:
            h["source"] = "adb_pds"; h["snippet"] = "PDS: " + ADB[m.group(1)][3][:150]
        hits += ah
    dt = DOCS / f"{re.sub(r'[^A-Za-z0-9_-]', '_', r.uid)}.txt"
    if dt.exists():
        dh = geocode_text(dt.read_text(encoding="utf-8", errors="ignore")[:400000])
        for h in dh:
            h["source"] = "document"
        hits += dh
    # V3: capital-only evidence coming solely from IATI location fields is an HQ default
    if hits and {h["district"] for h in hits} <= CAPITAL and all(h["source"] == "iati_coord" for h in hits):
        hits = []
    best = best_by_district(hits)
    for b in best:
        rows.append({"uid": r.uid, **b})
    status.append({"uid": r.uid, "best_precision": min([b["precision"] for b in best], default=8), "n_districts": len({b["district"] for b in best}),
                   "sources": ",".join(sorted({b.get("source", "") for b in best})), "has_doc": dt.exists()})
G = pd.DataFrame(rows); S = pd.DataFrame(status)
G.to_csv(ROOT / "data/processed/nepal/np2_geocodes.csv", index=False, encoding="utf-8-sig")
S.to_csv(ROOT / "data/processed/nepal/np2_project_status.csv", index=False, encoding="utf-8-sig")
print("projects:", len(S), "| best precision distribution:", S.best_precision.value_counts().sort_index().to_dict())
print("located at district or finer (<=4):", (S.best_precision <= 4).mean().round(3), "| palika or finer (<=3):", (S.best_precision <= 3).mean().round(3))
print("multi-district projects (>5 districts):", (S.n_districts > 5).mean().round(3))
