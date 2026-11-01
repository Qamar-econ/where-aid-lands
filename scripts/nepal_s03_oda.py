"""Nepal unified ODA project DB (protocol v1): AidData Nepal AIMS (geocoded, ~2000-2014), AidData earthquake release (2015),
IATI (d-portal), World Bank API. District assignment with tiers A/B/C, HQ rules, target typology, sector, ICT, size, duration.
Outputs: data/processed/nepal/oda_projects.csv, oda_assignments.csv
"""
import json, re, sys
import numpy as np, pandas as pd, geopandas as gpd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from oda_common import code_cat, kw_cat, donor_group, norm_title, dup_of, ICT  # noqa: E402
from gazetteer_nepal import geoparse, UPPER_DISTRICTS, NATIONWIDE, REGIONAL, CENTRAL  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw" / "nepal", ROOT / "data" / "processed" / "nepal"
DIST = gpd.read_file(OUT / "districts.gpkg")[["k", "province", "geometry"]]
PROV_D = DIST.groupby("province").k.apply(list).to_dict()
CAPITAL = {"kathmandu", "lalitpur", "bhaktapur"}
projects, assigns, FULL = [], [], {}


def add(uid, source, donor, title, desc, y0, y1, usd, code, cat=None):
    projects.append({"uid": uid, "source": source, "donor": donor, "donor_group": donor_group(donor), "title": title,
                     "desc": str(desc)[:600] if isinstance(desc, str) else "", "start_year": y0, "end_year": y1, "usd": usd,
                     "sector_code": code, "cat": cat or code_cat(code, f"{title} {desc}")})


def from_points(pts, prec_col):
    out = []
    if not len(pts):
        return out
    j = gpd.sjoin(pts, DIST, predicate="within", how="inner")
    for r in j.itertuples():
        p = getattr(r, prec_col)
        if p in (1, 2, 3):
            out.append(("district", r.k, "A" if p == 1 else "B"))
        elif p in (4, 5):
            out += [("district_split", d, "C") for d in PROV_D[r.province]]
    return out


def from_text(t):
    u, up = geoparse(t)
    out = [("district", d, "B") for d in u]
    for c in up:
        out += [("district_split", d, "C") for d in UPPER_DISTRICTS[c]]
    return out


def push(uid, units, method):
    best = {}
    for lvl, d, tier in units:
        if (lvl == "district_split") and any(x[1] == d and x[0] == "district" for x in units):
            continue
        best[d] = min(best.get(d, ("Z", lvl))[0], tier), lvl
    if not best:
        assigns.append({"uid": uid, "level": "national", "unit": "NP", "weight": 1.0, "method": method, "tier": None}); return
    w = 1 / len(best)
    for d, (tier, lvl) in best.items():
        assigns.append({"uid": uid, "level": "district" if lvl == "district" else "split", "unit": d, "weight": w, "method": method, "tier": tier})


def yr(s):
    return pd.to_datetime(s, errors="coerce").year


# ---------- 1. AidData Nepal AIMS ----------
ad = RAW / "aiddata/NepalAIMS_GeocodedResearchRelease_Level1_v1.4.1/NepalAIMS_GeocodedResearchRelease_Level1_v1.4.1/data"
P = pd.read_csv(ad / "projects.csv", encoding="latin1"); L = pd.read_csv(ad / "locations.csv", encoding="latin1")
Lg = gpd.GeoDataFrame(L, geometry=gpd.points_from_xy(L.longitude, L.latitude), crs=4326)
for r in P.itertuples():
    uid = f"AIMS-{r.project_id}"
    usd = r.total_commitments if pd.notna(r.total_commitments) and r.total_commitments > 0 else r.total_disbursements
    y0 = yr(r.start_actual_isodate)
    if pd.isna(y0):
        y0 = pd.to_numeric(str(r.transactions_start_year).split("|")[0], errors="coerce")
    y1 = yr(r.end_actual_isodate)
    if pd.isna(y1):
        y1 = pd.to_numeric(str(r.transactions_end_year).split("|")[-1], errors="coerce")
    add(uid, "AidData-AIMS", r.donors, r.project_title, "", y0, y1, usd, pd.to_numeric(str(r.ad_sector_codes).split("|")[0], errors="coerce"))
    FULL[uid] = str(r.project_title)
    units = from_points(Lg[Lg.project_id == r.project_id], "precision_code"); m = "coord"
    if not units:
        units = from_text(FULL[uid]); m = "text"
    push(uid, units, m if units else "national")
print("AIMS", len(P))

# ---------- 2. AidData earthquake release (2015 reconstruction) ----------
eq = RAW / "aiddata/NepalEarthquake_GeocodedResearchRelease_Level1_v1.0/NepalEarthquake_GeocodedResearchRelease_Level1_v1.0/data"
EP = pd.read_csv(eq / "projects.csv", encoding="latin1"); EL = pd.read_csv(eq / "locations.csv", encoding="latin1")
EA = pd.read_csv(eq / "projects_ancillary.csv", encoding="latin1")
desc_col = next((c for c in EA.columns if "descr" in c.lower()), None)
ed = EA.set_index(EA.columns[0])[desc_col].to_dict() if desc_col else {}
ELg = gpd.GeoDataFrame(EL, geometry=gpd.points_from_xy(EL.longitude, EL.latitude), crs=4326)
for r in EP.itertuples():
    uid = f"EQ-{r.project_id}"
    usd = getattr(r, "total_commitments", np.nan)
    add(uid, "AidData-EQ", r.donors, r.project_title, ed.get(r.project_id, ""), yr(getattr(r, "start_actual_isodate", None)) or 2015, np.nan,
        usd, pd.to_numeric(str(getattr(r, "ad_sector_codes", "")).split("|")[0], errors="coerce"))
    FULL[uid] = f"{r.project_title} {ed.get(r.project_id, '')}"
    units = from_points(ELg[ELg.project_id == r.project_id], "precision_code"); m = "coord"
    if not units:
        units = from_text(FULL[uid]); m = "text"
    push(uid, units, m if units else "national")
print("EQ", len(EP))

# ---------- 3. IATI ----------
acts = pd.DataFrame(json.loads((RAW / "iati/dportal_act_NP.json").read_text(encoding="utf-8")))
locs = pd.DataFrame(json.loads((RAW / "iati/dportal_loc_NP.json").read_text(encoding="utf-8")))
secs = pd.DataFrame(json.loads((RAW / "iati/dportal_sector_NP.json").read_text(encoding="utf-8")))
acts = acts[(acts.reporting != "AidData") & (acts.country_percent.fillna(100) >= 50)].drop_duplicates("aid")
acts["y0"] = pd.to_datetime(acts.day_start, unit="D", errors="coerce").dt.year
acts["y1"] = pd.to_datetime(acts.day_end, unit="D", errors="coerce").dt.year
iati_ft = pd.read_csv(ROOT / "data" / "interim" / "iati_fulltext_NP.csv").drop_duplicates("aid").set_index("aid")[["location_names", "text"]].fillna("").to_dict("index")
secs["sector_percent"] = secs.sector_percent.fillna(100)
top_sec = secs.sort_values("sector_percent", ascending=False).drop_duplicates("aid").set_index("aid").sector_group
locs = locs[locs.aid.isin(acts.aid) & locs.location_longitude.notna()].copy()
locs["xy"] = locs.location_longitude.round(2).astype(str) + "," + locs.location_latitude.round(2).astype(str)
rep = locs.groupby("xy").aid.nunique()
gen_name = locs.location_name.fillna("").str.strip().str.lower().isin(["nepal", "kathmandu", "federal democratic republic of nepal"])
locs["generic"] = locs.xy.map(rep).ge(40) | (gen_name & (locs.location_precision != 1))
LGi = gpd.GeoDataFrame(locs, geometry=gpd.points_from_xy(locs.location_longitude, locs.location_latitude), crs=4326)
LGi["prec"] = LGi.location_precision.fillna(2).astype(int).clip(1, 2)
LGi = gpd.sjoin(LGi[~LGi.generic], DIST, predicate="within", how="inner")
loc_by = LGi.groupby("aid").apply(lambda d: list(zip(["district"] * len(d), d.k, np.where(d.prec == 1, "A", "B"))), include_groups=False).to_dict()
pool = pd.DataFrame(projects); pool["nt"] = pool.title.map(norm_title)
pool_aims = pool[pool.source == "AidData-AIMS"]
n_dup = 0
for r in acts.itertuples():
    dg = donor_group(r.reporting)
    if (pd.isna(r.y0) or r.y0 <= 2015) and dup_of(r.title, dg, r.y0, pool_aims):
        n_dup += 1; continue
    uid = f"IATI-{r.aid}"
    usd = r.commitment if r.commitment and r.commitment > 0 else r.spend
    add(uid, "IATI", r.reporting, r.title, r.description, r.y0, r.y1, usd, top_sec.get(r.aid))
    ft = iati_ft.get(r.aid, {})
    body = f"{r.title} {r.description} {ft.get('text', '')}"
    for ln in str(ft.get("location_names", "")).split(" ; "):
        if ln.strip():
            body = body.replace(ln.strip(), " ")                   # body text without IATI location-field strings
    FULL[uid] = f"{r.title} {r.description} {ft.get('location_names', '')} {ft.get('text', '')}"
    units = loc_by.get(r.aid, []); m = "coord"
    if units and all(u[1] in CAPITAL for u in units) and not (geoparse(body)[0] & CAPITAL):
        units, m = [], "hq_default"                               # R4: capital only in location field
    if not units and m != "hq_default":
        units = from_text(FULL[uid]); m = "text"
        if units and all(u[1] in CAPITAL for u in units) and not (geoparse(body)[0] & CAPITAL):
            units, m = [], "hq_default"
    push(uid, units, m if units else ("hq_default" if m == "hq_default" else "national"))
print("IATI used", len(acts) - n_dup, "dropped as AIMS duplicates", n_dup)

# ---------- 4. World Bank ----------
wb = json.loads((RAW / "worldbank/wb_projects_NP.json").read_text(encoding="utf-8"))["projects"]
pool = pd.DataFrame(projects); pool["nt"] = pool.title.map(norm_title)
WB_MJ = {"Transportation": 210, "Energy & Extractives": 230, "Information and Communications Technologies": 220, "Health": 120, "Education": 110,
         "Social Protection": 160, "Water, Sanitation and Waste Management": 140, "Agriculture, Fishing and Forestry": 311,
         "Public Administration": 150, "Financial Sector": 240, "Industry, Trade and Services": 250}
for pid, p in wb.items():
    title = p.get("project_name", ""); y0 = pd.to_numeric(p.get("board_approval_year"), errors="coerce")
    if dup_of(title, "World Bank", y0, pool):
        continue
    ab = p.get("project_abstract", {}); ab = ab.get("cdata", "") if isinstance(ab, dict) else str(ab or "")
    mj = p.get("mjsector1name_mdk") if isinstance(p.get("mjsector1name_mdk"), str) else ""
    uid = f"WB-{pid}"
    add(uid, "WorldBank-API", "World Bank", title, ab + " " + str(p.get("pdo", "")), y0, pd.to_numeric(str(p.get("closingdate", ""))[:4], errors="coerce"),
        pd.to_numeric(str(p.get("totalcommamt", "")).replace(",", ""), errors="coerce"), WB_MJ.get(mj))
    FULL[uid] = f"{title} {ab} {p.get('pdo', '')}"
    units = from_text(FULL[uid])
    push(uid, units, "text" if units else "national")

PR, AS = pd.DataFrame(projects), pd.DataFrame(assigns)
located = set(AS.loc[AS.level != "national", "uid"]); meth = AS.groupby("uid").method.first().to_dict()


def ttype(u):
    if u in located: return "located"
    if meth.get(u) == "hq_default": return "central_institution"
    t = FULL.get(u, "")
    if REGIONAL.search(t): return "regional_multicountry"
    if NATIONWIDE.search(t): return "nationwide"
    if CENTRAL.search(t): return "central_institution"
    return "unresolved"


PR["target_type"] = PR.uid.map(ttype)
PR["ict"] = PR.uid.map(lambda u: bool(ICT.search(FULL.get(u, ""))))
PR["duration"] = (PR.end_year - PR.start_year).where(lambda s: (s >= 0) & (s <= 30))
PR["era"] = np.where(PR.start_year <= 2014, "e1", "e2")
PR["eq"] = PR.source.eq("AidData-EQ") | PR.uid.map(lambda u: bool(re.search(r"earthquake|reconstruction|post[- ]disaster|Gorkha", FULL.get(u, ""), re.I)))
PR.to_csv(OUT / "oda_projects.csv", index=False, encoding="utf-8-sig"); AS.to_csv(OUT / "oda_assignments.csv", index=False, encoding="utf-8-sig")
P2 = PR[PR.start_year.between(2000, 2026)]
print("projects 2000-2026:", len(P2)); print(P2.source.value_counts().to_dict())
print(P2.target_type.value_counts().to_dict())
print("USD share by target type:", (P2.groupby("target_type").usd.sum() / P2.usd.sum()).round(3).to_dict())
print("tiers:", AS.groupby(["level", "tier"], dropna=False).uid.nunique().to_dict())
print("era x located:", pd.crosstab(P2.era, P2.target_type).to_dict())
