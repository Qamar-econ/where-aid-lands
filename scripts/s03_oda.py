"""Unified ODA project database for Timor-Leste with posto assignment and sector categories.

Sources (priority order for de-duplication):
  1. AidData Timor-Leste AIMS geocoded v1.4.1 (Government AMP, 1998-2015)
  2. IATI via d-portal (all publishers except the AidData re-publication)
  3. World Bank Projects API (+ AidData WB geocoded locations)
  4. Budget Book 5 project tables 2013-2025 (Ministry of Finance)
Outputs:
  data/processed/oda_projects.csv      one row per unique project
  data/processed/oda_assignments.csv   project x unit rows (level posto / municipio / national, weight)
"""
import json, re, sys
import numpy as np, pandas as pd, geopandas as gpd
from rapidfuzz import fuzz
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gazetteer import geoparse, NATIONWIDE, REGIONAL, CENTRAL  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW, INT, PROC = ROOT / "data" / "raw", ROOT / "data" / "interim", ROOT / "data" / "processed"
po = gpd.read_file(PROC / "posto_covariates.gpkg")[["pcode", "posto", "mun_pcode", "municipio", "geometry"]]
mun_name = po.drop_duplicates("mun_pcode").set_index("mun_pcode").municipio.to_dict()

# ---------------- sector categories ----------------
CAT = {}
for c in [210]: CAT[c] = "ACCESS_transport"
for c in [220]: CAT[c] = "ACCESS_comms"
for c in [230, 231, 232, 233, 234, 235, 236]: CAT[c] = "ACCESS_energy"
for c in [110, 111, 112, 113, 114]: CAT[c] = "SOC_education"
for c in [120, 121, 122, 123, 130]: CAT[c] = "SOC_health"
for c in [140]: CAT[c] = "SOC_water"
for c in [160]: CAT[c] = "SOC_other"
for c in [150, 151, 152]: CAT[c] = "GOV"
for c in [240, 250, 310, 311, 312, 313, 320, 321, 322, 323, 331, 332]: CAT[c] = "PROD"
for c in [410, 430]: CAT[c] = "ENV_MULTI"
for c in [700, 720, 730, 740, 520]: CAT[c] = "HUMAN"
for c in [510, 530, 600, 910, 920, 930, 998]: CAT[c] = "OTHER"
AD_NAME = {"Government and civil society": 150, "Education": 110, "Support to non-governmental organisations (NGOs) and government organizations": 920,
           "Agriculture": 311, "Health": 120, "Multisector/cross-cutting": 430, "Other social infrastructure and services": 160,
           "Water supply and sanitation": 140, "Humanitarian aid (Including reconstruction and recovery)": 700,
           "Transport and storage": 210, "Communications": 220, "Unallocated/unspecified": 998, "Banking and financial services": 240,
           "Energy generation and supply": 230, "Business and other services": 250, "Industry": 321, "Trade policy and regulations": 331,
           "Population policies/programmes and reproductive health": 130, "Fishing": 313, "Forestry": 312, "Administrative costs of donors": 910,
           "Construction": 323, "Mineral resources and mining": 322, "Tourism": 332, "Commodity aid and general programme assistance": 530,
           "Action relating to debt": 600}
KW = [  # keyword classifier (title + description), checked in order
    ("ACCESS_transport", r"\broads?\b|\bbridges?\b|highway|\btransport(ation)?\b|\bports?\b|seaport|airport|ferry|maritime|aviation|runway"),
    ("ACCESS_energy", r"electric|power plant|\benergy|solar|hydro ?power|grid\b|electrification|generator"),
    ("ACCESS_comms", r"telecom|internet|broadband|\bICT\b|digital|submarine cable|fib(re|er) optic|e-government|connectivity"),
    ("HUMAN", r"humanitarian|emergency|disaster response|flood response|relief|refugee|covid|pandemic response"),
    ("SOC_water", r"\bwater\b|sanitation|\bWASH\b|hygiene|irrigation"),
    ("SOC_health", r"health|hospital|clinic|malaria|tuberculosis|\bHIV|nutrition|maternal|immuni[sz]|vaccin|medical|disease"),
    ("SOC_education", r"educat|school|teacher|university|scholarship|training|TVET|vocational|literacy|curricul"),
    ("PROD", r"agricult|farm|coffee|livestock|fisher|forest|rural development|market|business|enterprise|private sector|tourism|trade|employment|microfinance|value chain|food security"),
    ("GOV", r"governance|justice|police|parliament|elect|public financ|decentrali|civil service|capacity building|anti-corruption|audit|statistic|security sector|rule of law|human rights|gender"),
    ("ENV_MULTI", r"climate|environment|biodiversity|resilien|mangrove|disaster risk"),
]
KWX = [(c, re.compile(p, re.I)) for c, p in KW]


def kw_cat(text):
    for c, p in KWX:
        if isinstance(text, str) and p.search(text):
            return c
    return "OTHER"


def code_cat(code, text):
    c = CAT.get(int(code) if pd.notna(code) else -1)
    if c is None or c in ("OTHER", "ENV_MULTI"):
        k = kw_cat(text)
        # keyword can only re-route unspecific codes, or detect access infrastructure inside multisector
        if c is None or k.startswith("ACCESS") or c == "OTHER":
            return k
    return c


DONOR_GROUP = [("Korea", r"Korea|KOICA"), ("Japan", r"Japan|JICA"), ("Australia", r"Australia|DFAT|AusAID"),
               ("USA", r"United States|USAID|\bUS\b|USA|Millennium Challenge|Department of State|Peace Corps|Department of Defense|Department of Agriculture"),
               ("EU", r"European|\bEU\b"), ("Portugal", r"Portugal|Cam[õo]es|Camões|CamÃµes"), ("New Zealand", r"New Zealand"),
               ("Germany", r"German|BMZ|GIZ|KfW"), ("China", r"China|Chinese"), ("World Bank", r"World Bank|IDA\b|IFC|International Finance Corporation"),
               ("ADB", r"Asian Development Bank|ADB"), ("UN agencies", r"United Nations|UNDP|UNICEF|WFP|World Food|WHO|World Health|FAO|Food and Agriculture|UNFPA|ILO|International Labour|UN Women|UNESCO|IOM|International Organization for Migration|UNHCR|UNOPS|UNIDO|OCHA|CERF|UN Pooled"),
               ("Global funds", r"Global Fund|Gavi|GAVI|Global Environment|GEF|IFAD|CGIAR"),
               ("NGO/foundation", r"Oxfam|CARE|Plan |World Vision|Caritas|CAFOD|GlobalGiving|Foundation|Asia Foundation|Hivos|Water Aid|WaterAid|Mercy|Red Cross|Catholic")]
DGX = [(g, re.compile(p)) for g, p in DONOR_GROUP]


def donor_group(d):
    for g, p in DGX:
        if isinstance(d, str) and p.search(d):
            return g
    return "Other bilateral/other"


def norm_title(t):
    t = re.sub(r"[^a-z0-9 ]", " ", str(t).lower())
    return re.sub(r"\s+", " ", t).strip()


projects, assigns = [], []


FULLTEXT = {}


def add_project(uid, source, donor, title, desc, year, usd, code, cat=None, end_year=np.nan):
    projects.append({"uid": uid, "source": source, "donor": donor, "donor_group": donor_group(donor), "title": title,
                     "desc": (desc or "")[:600] if isinstance(desc, str) else "", "start_year": year, "end_year": end_year,
                     "usd": usd, "sector_code": code, "cat": cat or code_cat(code, f"{title} {desc}")})


def assign_points(uid, pts, source):
    """pts: GeoDataFrame of points with 'level' (posto/municipio/national)."""
    out = []
    if len(pts):
        j = gpd.sjoin(pts, po[["pcode", "mun_pcode", "geometry"]], predicate="within", how="left")
        for r in j.itertuples():
            if pd.isna(r.pcode):
                continue
            prec = getattr(r, "precision_code", None)
            if prec is None:
                prec = getattr(r, "location_precision", None)
            if r.level == "posto":
                out.append(("posto", r.pcode, "A" if prec == 1 else "B"))      # tier A: exact point; B: near / ADM2
            elif r.level == "municipio":
                out.append(("municipio", r.mun_pcode, "C"))
    return out


def assign_text(text):
    p, m = geoparse(text)
    return [("posto", x, "B") for x in p] + [("municipio", x, "C") for x in m]


def push(uid, units, method):
    best = {}
    for lvl, u, tier in units:
        k = (lvl, u)
        best[k] = min(best.get(k, "Z"), tier)
    if not best:
        assigns.append({"uid": uid, "level": "national", "unit": "TL", "weight": 1.0, "method": method, "tier": None})
        return
    w = 1.0 / len(best)
    for (lvl, u), tier in sorted(best.items()):
        assigns.append({"uid": uid, "level": lvl, "unit": u, "weight": w, "method": method, "tier": tier})


# ================= 1. AidData AMP =================
ad = RAW / "aiddata_tl/TimorLesteAIMS_GeocodedResearchRelease_Level1_v1.4.1/TimorLesteAIMS_GeocodedResearchRelease_Level1_v1.4.1/data"
P = pd.read_csv(ad / "projects.csv", encoding="latin1")
L = pd.read_csv(ad / "locations.csv", encoding="latin1")
A = pd.read_csv(ad / "projects_ancillary.csv", encoding="latin1").rename(columns={"project_ID": "project_id"})
AMP_TEXT = ["Project Description", "Expected Outcomes", "Final Project Results", "District", "SubDistrict", "Suco",
            "RDTL Beneficiary Institutions", "Implementing Agency", "Executing Agency"]
amp_extra = A.drop_duplicates("project_id").set_index("project_id")[AMP_TEXT + ["Implementation Level"]].to_dict("index")
L["level"] = np.select([L.precision_code.isin([1, 2, 3]), L.precision_code.isin([4, 5])], ["posto", "municipio"], "national")
Lg = gpd.GeoDataFrame(L, geometry=gpd.points_from_xy(L.longitude, L.latitude), crs=4326)
for r in P.itertuples():
    uid = f"AMP-{r.project_id}"
    yr = pd.to_datetime(r.start_actual_isodate, errors="coerce").year
    if pd.isna(yr):
        yr = pd.to_numeric(str(r.transactions_start_year).split("|")[0], errors="coerce")
    ey = pd.to_datetime(r.end_actual_isodate, errors="coerce").year
    if pd.isna(ey):
        ey = pd.to_numeric(str(r.transactions_end_year).split("|")[-1], errors="coerce")
    code = pd.to_numeric(str(r.ad_sector_codes).split("|")[0], errors="coerce")
    usd = r.total_commitments if pd.notna(r.total_commitments) and r.total_commitments > 0 else r.total_disbursements
    rec = amp_extra.get(r.project_id, {})
    full = " ".join(str(rec.get(c, "")) for c in AMP_TEXT if isinstance(rec.get(c), str))
    add_project(uid, "AidData-AMP", r.donors, r.project_title, rec.get("Project Description"), yr, usd, code, end_year=ey)
    FULLTEXT[uid] = f"{r.project_title} {full}"
    pts = Lg[Lg.project_id == r.project_id]
    units = assign_points(uid, pts, "AMP")
    method = "coord"
    if not units:
        units = assign_text(FULLTEXT[uid])
        method = "text"
        # R5 (codebook): AMP national-level projects whose only place is Dili are central-institution projects
        if units and rec.get("Implementation Level") == "National" and all(u[1].startswith("TL06") for u in units):
            units, method = [], "central_dili_field"
    push(uid, units, method if units else "national")

amp_titles = {u: norm_title(t) for u, t in ((p["uid"], p["title"]) for p in projects)}

# ================= 2. IATI =================
acts = pd.DataFrame(json.loads((RAW / "iati/dportal_act_TL.json").read_text(encoding="utf-8")))
locs = pd.DataFrame(json.loads((RAW / "iati/dportal_loc_TL.json").read_text(encoding="utf-8")))
secs = pd.DataFrame(json.loads((RAW / "iati/dportal_sector_TL.json").read_text(encoding="utf-8")))
acts = acts[acts.reporting != "AidData"].drop_duplicates("aid")
acts["start_year"] = pd.to_datetime(acts.day_start, unit="D", errors="coerce").dt.year
acts["end_year"] = pd.to_datetime(acts.day_end, unit="D", errors="coerce").dt.year
iati_ft = pd.read_csv(INT / "iati_fulltext.csv").drop_duplicates("aid").set_index("aid")[["location_names", "text"]].fillna("").to_dict("index")
acts = acts[(acts.country_percent.fillna(100) >= 50)]            # TL-focused activities only
secs["sector_percent"] = secs.sector_percent.fillna(100)
top_sec = secs.sort_values("sector_percent", ascending=False).drop_duplicates("aid").set_index("aid").sector_group
locs = locs[locs.aid.isin(acts.aid)].copy()
lg = gpd.GeoDataFrame(locs, geometry=gpd.points_from_xy(locs.location_longitude, locs.location_latitude), crs=4326)
# generic / default coordinates: national centroids and points repeated across >= 40 activities
lg["xy"] = lg.location_longitude.round(2).astype(str) + "," + lg.location_latitude.round(2).astype(str)
rep = lg.groupby("xy").aid.nunique()
generic_name = lg.location_name.fillna("").str.strip().str.lower().isin(["timor-leste", "east timor", "timor leste", "dili", "díli"])
lg["generic"] = lg.xy.map(rep).ge(40) | (generic_name & (lg.location_precision != 1))
lg["level"] = np.where(lg.generic, "national_or_hq", "posto")

# fuzzy de-dup against AMP (same donor group, similar title, start year +-2)
amp_df = pd.DataFrame(projects)
amp_df["nt"] = amp_df.title.map(norm_title)


def dup_of(title, dgroup, year, pool):
    nt = norm_title(title)
    cand = pool[(pool.donor_group == dgroup)]
    if pd.notna(year):
        cand = cand[(cand.start_year.isna()) | ((cand.start_year - year).abs() <= 2)]
    for t in cand.nt:
        if fuzz.token_set_ratio(nt, t) >= 88 and len(nt) > 12:
            return True
    return False


n_dup = 0
for r in acts.itertuples():
    dg = donor_group(r.reporting)
    if (pd.isna(r.start_year) or r.start_year <= 2016) and dup_of(r.title, dg, r.start_year, amp_df):
        n_dup += 1
        continue
    uid = f"IATI-{r.aid}"
    usd = r.commitment if r.commitment and r.commitment > 0 else r.spend
    add_project(uid, "IATI", r.reporting, r.title, r.description, r.start_year, usd, top_sec.get(r.aid), end_year=r.end_year)
    ft = iati_ft.get(r.aid, {})
    FULLTEXT[uid] = f"{r.title} {r.description} {ft.get('location_names', '')} {ft.get('text', '')}"
    pts = lg[lg.aid == r.aid]
    real = pts[~pts.generic]
    units = assign_points(uid, real.assign(level="posto"), "IATI")
    method = "coord"
    if not units:
        units = assign_text(FULLTEXT[uid])
        method = "text"
        # R4 (codebook): a capital mention found only in IATI location fields (not in title/description/results text)
        # is an HQ default ("Dili", "capital", "Location of central agreement") -> not a target
        if units and all(u[1].startswith("TL06") for u in units):
            body = f"{r.title} {r.description} {ft.get('text', '')}"
            for ln in str(ft.get("location_names", "")).split(" ; "):
                if ln.strip():
                    body = body.replace(ln.strip(), " ")
            body = re.sub(r"Location of central agreement[^|]*|capital of a political entity|\|\s*capital\s*\||"
                          r"\|\s*(Dili|Díli)\s*(;[^|]*)?(?=\|)|Port Moresby", " ", body)
            if not assign_text(body) or "Location of central agreement" in ft.get("text", ""):
                units, method = [], "hq_default"
    if not units and len(pts):
        method = "hq_default"
    push(uid, units, method if units else ("hq_default" if method == "hq_default" else "national"))
print("IATI dropped as AMP duplicates:", n_dup)

# ================= 3. World Bank API =================
wb = json.loads((RAW / "worldbank/wb_projects_TL.json").read_text(encoding="utf-8"))["projects"]
pool = pd.DataFrame(projects); pool["nt"] = pool.title.map(norm_title)
WB_MJ = {"Transportation": 210, "Transport": 210, "Energy & Extractives": 230, "Energy and mining": 230,
         "Information and Communications Technologies": 220, "Information and communications": 220,
         "Health": 120, "Education": 110, "Social Protection": 160, "Water, Sanitation and Waste Management": 140,
         "Water/Sanit/Flood Prot": 140, "Agriculture, Fishing and Forestry": 311, "Agriculture": 311,
         "Public Administration": 150, "Public Admin, Law, and Justice": 150, "Financial Sector": 240,
         "Industry, Trade and Services": 250, "Health and other social services": 120}
for pid, p in wb.items():
    title = p.get("project_name", "")
    yr = pd.to_numeric(p.get("board_approval_year") or str(p.get("boardapprovaldate", ""))[:4], errors="coerce")
    if dup_of(title, "World Bank", yr, pool):
        continue
    mj = p.get("mjsector1name_mdk") or (p.get("mjsector") or [""])[0] if isinstance(p.get("mjsector"), list) else p.get("mjsector1name_mdk", "")
    code = WB_MJ.get(mj if isinstance(mj, str) else "", None)
    abstract = (p.get("project_abstract") or {}).get("cdata", "") if isinstance(p.get("project_abstract"), dict) else str(p.get("project_abstract") or "")
    usd = pd.to_numeric(str(p.get("totalcommamt", "0")).replace(",", ""), errors="coerce")
    uid = f"WB-{pid}"
    ey = pd.to_numeric(str(p.get("closingdate", ""))[:4], errors="coerce")
    add_project(uid, "WorldBank-API", "World Bank", title, abstract + " " + str(p.get("pdo", "")), yr, usd, code, end_year=ey)
    FULLTEXT[uid] = f"{title} {abstract} {p.get('pdo', '')} {p.get('impagency', '')}"
    units = assign_text(FULLTEXT[uid])
    push(uid, units, "text" if units else "national")

# ================= 4. Budget Book 5 rows =================
bb = pd.read_csv(INT / "bb5_rows.csv")
bb = bb[bb.donors.fillna("") != ""].copy()
# drop aggregate rows (donor totals / percentage tables) and implausible amounts (> USD 100 m in one year)
bb = bb[~bb.text.str.contains(r"\d+(\.\d+)?\s?%", regex=True) & ~bb.title.str.match(r"^(TOTAL|Total|Subtotal|UN Family|Others?)\b")]
bb.loc[bb.amount_k > 100000, "amount_k"] = np.nan
bb["dg"] = bb.donors.str.split("; ").str[0].map(donor_group)
bb["nt"] = bb.title.map(norm_title)
# de-dup within BB5 across editions: keep first appearance
bb = bb.sort_values("book_year")
keep, seen, last_ed = [], {}, {}
for r in bb.itertuples():
    key = r.dg
    lst = seen.setdefault(key, [])
    hit = next((i for t, i in lst if fuzz.token_set_ratio(r.nt, t) >= 90), None)
    if hit is not None:
        last_ed[hit] = max(last_ed[hit], r.book_year)       # project still listed in a later edition -> still active
        continue
    lst.append((r.nt, r.Index)); keep.append(r.Index); last_ed[r.Index] = r.book_year
bb = bb.loc[keep]
bb["last_edition"] = bb.index.map(last_ed)
pool = pd.DataFrame(projects); pool["nt"] = pool.title.map(norm_title)
n_bdup = 0
for r in bb.itertuples():
    if dup_of(r.title, r.dg, r.book_year, pool):
        n_bdup += 1
        continue
    uid = f"BB5-{r.book_year}-{r.Index}"
    add_project(uid, "BudgetBook5", r.donors.split("; ")[0], r.title, r.text, r.book_year,
                r.amount_k * 1000 if pd.notna(r.amount_k) else np.nan, None, kw_cat(r.text), end_year=r.last_edition)
    FULLTEXT[uid] = r.text
    units = [] if NATIONWIDE.search(r.text) else assign_text(r.text)
    push(uid, units, "text" if units else "national")
print("BB5 unique rows used:", len(bb) - n_bdup, "dropped as duplicates:", n_bdup)

PR = pd.DataFrame(projects)
AS = pd.DataFrame(assigns)
# ---- target type for projects without a sub-national location ----
located = set(AS.loc[AS.level != "national", "uid"])


meth_of = AS.groupby("uid").method.first().to_dict()


def target_type(uid):
    if uid in located:
        return "located"
    if meth_of.get(uid) in ("hq_default", "central_dili_field"):
        return "central_institution"
    t = FULLTEXT.get(uid, "")
    if REGIONAL.search(t):
        return "regional_multicountry"
    if NATIONWIDE.search(t):
        return "nationwide"
    if CENTRAL.search(t):
        return "central_institution"
    return "unresolved"


PR["target_type"] = PR.uid.map(target_type)
# projects with an ICT component (digital delivery, information systems, e-services, connectivity)
ICT = re.compile(r"\bICTs?\b|information and communications? technolog|\bdigital|e-?learning|online (learning|platform|system|service)|"
                 r"\binternet\b|computer|e-?government|e-?governance|e-?justice|e-?health|e-?procurement|information (management )?systems?\b|"
                 r"\bdatabase|software|mobile (phone|money|app|application|banking|network)|\bSMS\b|tele-?medicine|tele-?health|"
                 r"broadband|connectivity|\bEMIS\b|\bHMIS\b|DHIS ?2|\bIFMIS\b|\bGIS\b|web[- ]based|digitali[sz]|telecommunication", re.I)
PR["ict"] = PR.uid.map(lambda u: bool(ICT.search(FULLTEXT.get(u, ""))))
PR["text_len"] = PR.uid.map(lambda u: len(FULLTEXT.get(u, "")))
PR["duration"] = (PR.end_year - PR.start_year).where(lambda s: (s >= 0) & (s <= 30))
print("\ntarget types:\n", PR.target_type.value_counts())
print(pd.crosstab(PR.source, PR.target_type))
PR["cat_group"] = PR.cat.str.split("_").str[0].replace({"SOC": "SOCIAL"})
PR["access_infra"] = PR.cat.str.startswith("ACCESS")
PR.to_csv(PROC / "oda_projects.csv", index=False, encoding="utf-8-sig")
AS.to_csv(PROC / "oda_assignments.csv", index=False, encoding="utf-8-sig")

print("\nprojects by source:\n", PR.source.value_counts())
print("\ncategories:\n", PR.cat.value_counts())
print("\nassignment method by source:\n", AS.drop_duplicates("uid").merge(PR[["uid", "source"]]).groupby(["source", "method"]).size())
lv = AS.groupby("uid")["level"].agg(lambda s: "posto" if (s == "posto").any() else ("municipio" if (s == "municipio").any() else "national"))
print("\nbest level per project:\n", lv.value_counts())
print("\ndonor groups:\n", PR.donor_group.value_counts())
print("\nstart years:\n", PR.start_year.value_counts().sort_index().to_dict())
