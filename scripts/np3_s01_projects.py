"""Integrated Nepal project database 2010-2025 (paper version): every project in the harmonised inventory (AidData-AIMS,
AidData earthquake, IATI, World Bank), all sectors, with one location rule set.

Location sources (union, most precise per district kept):
  AidData-AIMS / AidData-EQ : expert geocodes from AidData (existing assignments, district / split levels)
  IATI / World Bank         : geocoder v2 (np2_geo: V1-V5) on title + description + full IATI record, IATI coordinates (V3 HQ rule),
                              ADB Project Data Sheet 'geographical location', harvested project documents
Status: located (>= 1 district at precision <= 4) | central_institution (central-government target, not located) | national / unresolved
Planning year: start year - 1 (observed median lag 0.86 y) for every project; documented dates kept for robustness.
Outputs: data/processed/nepal/np3_projects.csv, np3_assign.csv
"""
import json, re, sys
import numpy as np, pandas as pd, geopandas as gpd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from np2_geo import geocode_text, best_by_district  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = pd.read_csv(ROOT / "data/processed/nepal/oda_projects.csv")
P = P[P.start_year.between(2010, 2025)].copy()
A0 = pd.read_csv(ROOT / "data/processed/nepal/oda_assignments.csv")
ft = pd.read_csv(ROOT / "data/interim/iati_fulltext_NP.csv").drop_duplicates("aid").set_index("aid").fillna("")
ADB = json.loads((ROOT / "data/interim/adb_pds_milestones.json").read_text(encoding="utf-8"))
DOCS = ROOT / "data/interim/np2_doctext"
CAPITAL = {"kathmandu", "lalitpur", "bhaktapur"}
Dg = gpd.read_file(ROOT / "data/processed/nepal/districts.gpkg")
loc = pd.DataFrame(json.loads((ROOT / "data/raw/nepal/iati/dportal_loc_NP.json").read_text(encoding="utf-8")))
loc = loc[loc.location_longitude.notna()]
xy = loc.location_longitude.round(2).astype(str) + loc.location_latitude.round(2).astype(str)
loc = loc[xy.map(loc.assign(xy=xy).groupby("xy").aid.nunique()) < 40]
lg = gpd.sjoin(gpd.GeoDataFrame(loc, geometry=gpd.points_from_xy(loc.location_longitude, loc.location_latitude), crs=4326), Dg[["k", "geometry"]], predicate="within")
COORD = lg.groupby("aid").k.apply(set).to_dict()

rows, status = [], []
for r in P.itertuples():
    ds, src = {}, set()
    if r.source.startswith("AidData"):
        a = A0[(A0.uid == r.uid) & A0.level.isin(["district", "split"])]
        for d in a.unit:
            ds[d] = 1; src.add("aiddata")
    else:
        aid = r.uid[5:] if r.uid.startswith("IATI-") else None
        core = f"{r.title} {r.desc if isinstance(r.desc, str) else ''}"
        text = core + (" " + ft.text.get(aid, "") if aid else "")
        hits = geocode_text(text)
        cd = {h["district"] for h in geocode_text(core)}
        hits = [h for h in hits if h["district"] not in CAPITAL or h["district"] in cd]  # V3b
        m = re.search(r"46004-(\d{5}-\d{3})", r.uid)
        if m and ADB.get(m.group(1)) and ADB[m.group(1)][3]:
            hits += geocode_text(ADB[m.group(1)][3]); src.add("adb_pds")
        dt = DOCS / f"{re.sub(r'[^A-Za-z0-9_-]', '_', r.uid)}.txt"
        if dt.exists():
            hits += geocode_text(dt.read_text(encoding="utf-8", errors="ignore")[:400000]); src.add("document")
        for h in best_by_district(hits):
            if h["precision"] <= 4:
                ds[h["district"]] = min(ds.get(h["district"], 9), h["precision"]); src.add("text")
        cds = COORD.get(aid, set()) if aid else set()
        if cds and not cds <= CAPITAL:  # V3: capital-only coordinates are HQ defaults
            for d in cds:
                ds[d] = 1; src.add("coord")
    n = len(ds)
    for d, p in ds.items():
        rows.append({"uid": r.uid, "k": d, "w": 1 / n, "precision": p})
    stt = "located" if n else ("central_institution" if r.target_type == "central_institution" else "national_unresolved")
    status.append({"uid": r.uid, "status": stt, "n_districts": n, "best_precision": min(ds.values()) if n else 8, "loc_sources": ",".join(sorted(src))})
S = pd.DataFrame(status)
P = P.merge(S, on="uid")
GRP = {"SOC_education": "Education", "SOC_health": "Health", "SOC_water": "Water & sanitation", "GOV": "Governance", "PROD": "Agriculture & production",
       "ENV_MULTI": "Environment & multisector", "SOC_other": "Other social", "HUMAN": "Humanitarian", "ACCESS_energy": "Infrastructure",
       "ACCESS_transport": "Infrastructure", "ACCESS_comms": "Infrastructure", "OTHER": "Other/unclassified"}
P["category"] = P["cat"].map(GRP).fillna("Other/unclassified")
P["family"] = np.select([P.category.isin(["Humanitarian"]), P.category.eq("Infrastructure"), P.category.eq("Other/unclassified")],
                        ["humanitarian", "infrastructure", "other"], "development")
P["size"] = pd.cut(P.usd, [-1, 1e6, 1e7, np.inf], labels=["< $1M", "$1-10M", "> $10M"]).astype(str).replace("nan", "unknown")
MDB = r"Asian Development Bank|World Bank|International Development Association|International Finance Corporation|IFAD|International Fund for Agricultural|Asian Infrastructure|European Investment"
UN = r"United Nations|UNICEF|UNDP|UNFPA|World Health|WFP|Food and Agriculture|UN Women|UNOPS|ILO|IOM|UNESCO|Gavi|Global Fund|Green Climate|UN Pooled"
P["dclass"] = np.select([P.donor.str.contains(MDB, na=False), P.donor.str.contains(UN, na=False) | P.donor_group.isin(["UN agencies"]),
                         P.donor_group.eq("NGO/foundation")], ["MDB", "UN", "NGO/foundation"], "Bilateral")
P["planning_year"] = (P.start_year - 1).astype(int)
pl = pd.read_csv(ROOT / "data/processed/nepal/np2_planning.csv")
obs = pl[pl.method.isin(["O1", "O1a", "O2"])].set_index("uid").planning_year
P["planning_year_doc"] = P.uid.map(obs)
P["quake"] = P.source.eq("AidData-EQ") | P.title.str.contains(r"(?i)earthquake|reconstruction|post-disaster|rebuild", na=False) & P.start_year.between(2015, 2018)
P["covid"] = P.title.str.contains(r"(?i)covid|coronavirus|pandemic", na=False)
P.drop(columns=["desc"]).to_csv(ROOT / "data/processed/nepal/np3_projects.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(rows).to_csv(ROOT / "data/processed/nepal/np3_assign.csv", index=False, encoding="utf-8-sig")
print("projects 2010-2025:", len(P), "| USD total (bn):", round(P.usd.sum() / 1e9, 1))
print(pd.crosstab(P.family, P.status, margins=True).to_string())
print("located share by source:", P.groupby("source").status.apply(lambda s: round((s == "located").mean(), 3)).to_dict())
print("located share of USD:", round(P.loc[P.status == "located", "usd"].sum() / P.usd.sum(), 3),
      "| central share of USD:", round(P.loc[P.status == "central_institution", "usd"].sum() / P.usd.sum(), 3))
