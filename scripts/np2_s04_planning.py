"""Code the planning-start date of each Nepal development project (2017-2025 starts), observed first, estimated last.

Hierarchy (method codes, best first):
  O1  ADB Concept Clearance (Project Data Sheet). If concept clearance precedes approval by > 4 years the clearance belongs
      to the parent series (phase / additional financing) -> O1a: fact-finding date minus the median ADB (fact-finding -
      concept clearance) gap
  O2  World Bank: date of the first Concept-stage PID (WDS); if absent, first PID / PIN of any stage
  O3  IATI document dated before the start in a planning category (A01 appraisal, A02 objectives, A09 MoU) -> earliest such date
      (an upper bound on planning start: the document exists by then)
  S1  first IATI commitment (transaction type 2/11) minus median (approval - planning start) from O1/O2 projects of the same
      donor class
  E1  project start minus median (start - planning start) from observed projects of the same donor class
      (MDB / bilateral / UN-other); literature default if fewer than 10 observed: 2.0 years (MDB) / 1.5 (bilateral) / 1.0 (UN)
Output: data/processed/nepal/np2_planning.csv  (uid, start_date, approval_date, planning_date, planning_year, method, evidence)
"""
import json, re
import numpy as np, pandas as pd, requests
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
I = pd.read_csv(ROOT / "data/interim/np2_inventory.csv"); I = I[I.dev].copy()
M = pd.read_csv(ROOT / "data/interim/np2_iati_meta.csv").drop_duplicates("aid").set_index("aid")
L = pd.read_csv(ROOT / "data/interim/np2_doclinks.csv")
W = pd.read_csv(ROOT / "data/interim/np2_wb_docs.csv")
A = json.loads((ROOT / "data/interim/adb_pds_milestones.json").read_text(encoding="utf-8"))
dt = lambda s: pd.to_datetime(s, errors="coerce", dayfirst=True)


def donor_class(d):
    d = str(d)
    if re.search(r"Asian Development Bank|World Bank|International Development Association|IFC|International Finance Corporation|IFAD|"
                 r"International Fund for Agricultural|Asian Infrastructure|European Investment", d):
        return "MDB"
    if re.search(r"United Nations|UNICEF|UNDP|UNFPA|World Health|WFP|Food and Agriculture|UN Women|UNOPS|ILO|IOM|UNESCO|Gavi|Global Fund|"
                 r"Green Climate|GlobalGiving|Foundation", d, re.I):
        return "UN-other"
    return "bilateral"


I["dclass"] = I.donor.map(donor_class)
aid = I.uid.str.replace("^IATI-", "", regex=True)
meta = M.reindex(aid)
meta.index = I.index
I["start_date"] = dt(meta.get("date_2")).fillna(dt(meta.get("date_1")))
I["start_date"] = I.start_date.fillna(pd.to_datetime(I.start_year.astype("Int64").astype(str) + "-07-01", errors="coerce"))
I["commit_date"] = dt(meta.get("first_tx_2")).fillna(dt(meta.get("first_tx_11")))

# ---- O1 ADB
adb = {}
for k, v in A.items():
    if k.startswith("_"):
        continue
    cc, ff, ap = dt(v[0]), dt(str(v[1])[:11]), dt(v[2])
    adb[k] = (cc, ff, ap)
gap_ff = pd.Series([(f - c).days for c, f, a in adb.values() if pd.notna(c) and pd.notna(f) and 0 < (a - c).days <= 1461]).median()
rows = {}
for i, r in I.iterrows():
    m = re.search(r"46004-(\d{5}-\d{3})", r.uid)
    if m and m.group(1) in adb:
        cc, ff, ap = adb[m.group(1)]
        if pd.notna(cc) and pd.notna(ap) and (ap - cc).days <= 1461:
            rows[i] = (cc, ap, "O1", f"ADB PDS concept clearance {cc.date()}; approval {ap.date()}")
        elif pd.notna(ff):
            rows[i] = (ff - pd.Timedelta(days=gap_ff), ap, "O1a", f"ADB PDS fact-finding {ff.date()} minus median CC->FF {gap_ff:.0f} d (series CC {cc.date() if pd.notna(cc) else 'NA'})")

# ---- O2 World Bank
W["docdt"] = pd.to_datetime(W.docdt, errors="coerce")
wbappr = {}
for p in W.pnum.unique():
    try:
        j = requests.get("https://search.worldbank.org/api/v2/projects", params={"format": "json", "id": p, "fl": "boardapprovaldate"}, timeout=60).json()
        wbappr[p] = pd.to_datetime(list(j["projects"].values())[0].get("boardapprovaldate"), errors="coerce").tz_localize(None)
    except Exception:
        wbappr[p] = pd.NaT
for i, r in I.iterrows():
    m = re.search(r"(P\d{6})", r.uid)
    if not m or i in rows:
        continue
    d = W[W.pnum == m.group(1)]
    pid = d[d.docty.fillna("").str.contains("Information Document|Concept", case=False)]
    concept = pid[pid.title.fillna("").str.contains("Concept", case=False)]
    pick = concept if len(concept) else pid
    if len(pick):
        x = pick.sort_values("docdt").iloc[0]
        rows[i] = (x.docdt, wbappr.get(m.group(1)), "O2", f"WB {x.docty} dated {x.docdt.date()}: {str(x.title)[:90]}")

# ---- O3 IATI planning documents
L["doc_date"] = pd.to_datetime(L.doc_date, errors="coerce")
GENERIC = r"(?i)undaf|cooperation framework|country programme|country program|strategy|strategic plan|annual report|policy|framework|guideline"
plan_docs = L[L.category.fillna("").str.contains(r"A01|A02|A09") & L.doc_date.notna() & ~L.title.fillna("").str.contains(GENERIC)
              & ~L.url.fillna("").str.contains(GENERIC)]
first_doc = plan_docs.sort_values("doc_date").groupby("aid").first()
for i, r in I.iterrows():
    a = r.uid[5:]
    if i in rows or a not in first_doc.index:
        continue
    x = first_doc.loc[a]
    if pd.notna(r.start_date) and r.start_date - pd.Timedelta(days=1826) <= x.doc_date <= r.start_date + pd.Timedelta(days=90):
        rows[i] = (x.doc_date, r.commit_date, "O3", f"IATI doc [{x.category}] '{str(x.title)[:70]}' dated {x.doc_date.date()}")

P = pd.DataFrame.from_dict(rows, orient="index", columns=["planning_date", "approval_date", "method", "evidence"])
I = I.join(P)
obs = I[I.method.isin(["O1", "O2"])]
lag_appr = (obs.approval_date - obs.planning_date).dt.days.groupby(obs.dclass).median()
obs_all = I[I.method.isin(["O1", "O1a", "O2"])]  # O3 is only an upper bound -> not used for lag estimation
lag_start = ((obs_all.start_date - obs_all.planning_date).dt.days.groupby(obs_all.dclass).agg(["median", "count"]))
DEFAULT = {"MDB": 730, "bilateral": 548, "UN-other": 365}
for c in DEFAULT:
    if c not in lag_start.index or lag_start.loc[c, "count"] < 10:
        lag_start.loc[c, ["median", "count"]] = [DEFAULT[c], 0]
lag_appr = lag_appr.reindex(list(DEFAULT)).fillna(pd.Series(DEFAULT) * 0.6).round()
lag_start["median"] = lag_start["median"].round()
I["planning_date"] = pd.to_datetime(I.planning_date).astype("datetime64[ns]")
I["evidence"] = I.evidence.astype(object); I["method"] = I.method.astype(object)

for i, r in I[I.method.isna()].iterrows():
    if pd.notna(r.commit_date) and (pd.isna(r.start_date) or r.commit_date <= r.start_date + pd.Timedelta(days=365)):
        I.loc[i, ["planning_date", "method", "evidence"]] = [r.commit_date - pd.Timedelta(days=lag_appr[r.dclass]), "S1",
                                                            f"first IATI commitment {r.commit_date.date()} minus median approval lag {lag_appr[r.dclass]:.0f} d ({r.dclass})"]
    elif pd.notna(r.start_date):
        L0 = lag_start.loc[r.dclass, "median"]
        I.loc[i, ["planning_date", "method", "evidence"]] = [r.start_date - pd.Timedelta(days=L0), "E1",
                                                            f"start {r.start_date.date()} minus median start lag {L0:.0f} d ({r.dclass}, n_obs={int(lag_start.loc[r.dclass, 'count'])})"]
I["planning_date"] = pd.to_datetime(I.planning_date)
I["planning_year"] = I.planning_date.dt.year
I["lag_years"] = ((I.start_date - I.planning_date).dt.days / 365.25).round(2)
out = I[["uid", "donor", "dclass", "title", "usd", "start_year", "start_date", "commit_date", "approval_date", "planning_date", "planning_year",
         "lag_years", "method", "evidence"]]
out.to_csv(ROOT / "data/processed/nepal/np2_planning.csv", index=False, encoding="utf-8-sig")
print("method counts:", out.method.value_counts().to_dict())
print("observed (O*) share by USD:", round(out.loc[out.method.str.startswith("O"), "usd"].sum() / out.usd.sum(), 3))
print("median start lag by class (observed):", lag_start.to_dict())
print("median approval lag by class (O1/O2):", lag_appr.round(0).to_dict())
print("observed planning->start lag, years (O1/O2):", out[out.method.isin(["O1", "O2"])].lag_years.describe().round(2).to_dict())
print("planning year distribution:", out.planning_year.value_counts().sort_index().to_dict())
