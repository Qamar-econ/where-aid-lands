"""Harvest project documents and dated milestones for Nepal development projects starting 2017-2025.

A  World Bank: WDS document list per P-number (all WB projects): document type + date; text of PID / PAD / latest ISR / ICR
B  IATI: activity dates (1 planned start, 2 actual start, 3 planned end, 4 actual end), first commitment/disbursement dates,
   activity status, document links with document-date and category (all dev projects)
C  Other documents (top projects by USD, non-WB): IATI document links -> text (pdf / odt / docx / html)
Outputs: data/interim/np2_wb_docs.csv, np2_iati_meta.csv, np2_doclinks.csv, np2_doctext/<uid>.txt (concatenated, with headers)
"""
import io, json, re, sys, zipfile, time
import pandas as pd, requests
from concurrent.futures import ThreadPoolExecutor
from lxml import etree
from pathlib import Path
from pdfminer.high_level import extract_text

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/interim/np2_doctext"; OUT.mkdir(exist_ok=True)
I = pd.read_csv(ROOT / "data/interim/np2_inventory.csv"); I = I[I.dev].sort_values("usd", ascending=False)
H = {"User-Agent": "Mozilla/5.0 (research; aid-geocoding)"}
TOPN = int(sys.argv[1]) if len(sys.argv) > 1 else 400


def safe(uid):
    return re.sub(r"[^A-Za-z0-9_-]", "_", uid)


def pnum(uid):
    m = re.search(r"(P\d{6})", uid)
    return m.group(1) if m else None


def to_text(content, url, ctype=""):
    u = url.lower()
    try:
        if content[:4] == b"%PDF" or u.endswith(".pdf"):
            return extract_text(io.BytesIO(content), maxpages=40)
        if content[:2] == b"PK":
            z = zipfile.ZipFile(io.BytesIO(content))
            name = "content.xml" if "content.xml" in z.namelist() else ("word/document.xml" if "word/document.xml" in z.namelist() else None)
            if name:
                return re.sub(r"<[^>]+>", " ", z.read(name).decode("utf-8", "ignore"))
            return ""
        t = content.decode("utf-8", "ignore")
        t = re.sub(r"(?is)<(script|style).*?</\1>", " ", t)
        return re.sub(r"<[^>]+>", " ", t)
    except Exception:
        return ""


def get(url, timeout=90):
    for k in range(2):
        try:
            r = requests.get(url, headers=H, timeout=timeout)
            if r.status_code == 200:
                return r.content
            return None
        except Exception:
            time.sleep(2)
    return None


# ---------------- A  World Bank ----------------
def wb_docs(p):
    try:
        r = requests.get("https://search.worldbank.org/api/v2/wds", params={"format": "json", "proid": p, "rows": 200,
                         "fl": "docty,docdt,txturl,pdfurl,display_title"}, timeout=90).json()
        return [{"pnum": p, "docty": v.get("docty"), "docdt": (v.get("docdt") or "")[:10], "txturl": v.get("txturl"),
                 "title": v.get("display_title")} for v in r.get("documents", {}).values() if isinstance(v, dict) and v.get("docdt")]
    except Exception:
        return []


wbp = sorted({pnum(u) for u in I.uid if pnum(u)})
with ThreadPoolExecutor(8) as ex:
    W = pd.DataFrame([d for ds in ex.map(wb_docs, wbp) for d in ds])
W.to_csv(ROOT / "data/interim/np2_wb_docs.csv", index=False, encoding="utf-8-sig")
print("WB projects:", len(wbp), "docs:", len(W))

KEEP = re.compile(r"Program Information Document|Project Information Document|Project Appraisal Document|Program Appraisal|"
                  r"Implementation Status and Results|Implementation Completion|Project Paper|Program Document|Concept", re.I)


def wb_text(p):
    d = W[(W.pnum == p) & W.docty.fillna("").str.contains(KEEP)].sort_values("docdt")
    if d.empty:
        return p, ""
    pick = pd.concat([d.head(2), d[d.docty.str.contains("Appraisal|Program Document|Project Paper", case=False)].tail(1),
                      d[d.docty.str.contains("Implementation Status|Completion", case=False)].tail(1)]).drop_duplicates("txturl")
    parts = []
    for r in pick.itertuples():
        c = get(r.txturl) if isinstance(r.txturl, str) else None
        if c:
            parts.append(f"\n\n##### DOC {r.docty} | {r.docdt} | {r.txturl}\n" + c.decode("utf-8", "ignore")[:300000])
    return p, "".join(parts)


with ThreadPoolExecutor(6) as ex:
    WT = dict(ex.map(wb_text, wbp))

# ---------------- B  IATI metadata ----------------
aids = set(I.loc[I.source == "IATI", "uid"].str[5:])
meta, links = [], []
for _, el in etree.iterparse(str(ROOT / "data/raw/nepal/iati/dportal_full_NP.xml"), events=("end",), tag="iati-activity", huge_tree=True, recover=True):
    aid = (el.findtext("iati-identifier") or "").strip()
    if aid in aids:
        m = {"aid": aid, "status": (el.find("activity-status").get("code") if el.find("activity-status") is not None else None),
             "last_updated": el.get("last-updated-datetime")}
        for ad in el.iter("activity-date"):
            m[f"date_{ad.get('type')}"] = (ad.get("iso-date") or "")[:10]
        tx = {}
        for t in el.iter("transaction"):
            tt = t.find("transaction-type"); td = t.find("transaction-date")
            if tt is not None and td is not None and td.get("iso-date"):
                k = tt.get("code"); tx[k] = min(tx.get(k, "9999"), td.get("iso-date")[:10])
        m.update({f"first_tx_{k}": v for k, v in tx.items()})
        m["participating"] = " ; ".join(sorted({" ".join(n.text or "" for n in p.iter("narrative")).strip() for p in el.iter("participating-org")}))[:400]
        meta.append(m)
        for d in el.iter("document-link"):
            dd = d.find("document-date")
            links.append({"aid": aid, "url": d.get("url"), "format": d.get("format"),
                          "title": " ".join(n.text or "" for n in d.find("title").iter("narrative")) if d.find("title") is not None else "",
                          "category": ",".join(c.get("code") or "" for c in d.iter("category")),
                          "doc_date": dd.get("iso-date") if dd is not None else None})
    el.clear()
    while el.getprevious() is not None:
        del el.getparent()[0]
M = pd.DataFrame(meta); L = pd.DataFrame(links)
M.to_csv(ROOT / "data/interim/np2_iati_meta.csv", index=False, encoding="utf-8-sig")
L.to_csv(ROOT / "data/interim/np2_doclinks.csv", index=False, encoding="utf-8-sig")
print("IATI meta:", len(M), "| doc links:", len(L))

# ---------------- C  other documents for top projects ----------------
SKIP = re.compile(r"undaf|cooperation-framework|uninfo|programming approach|humanitarianresponse|#project-tenders|/media/\d+/file/covid|"
                  r"transparency/documents-register|cambodia|\.xlsx?$", re.I)
top = I.head(TOPN)


def other_text(uid):
    aid = uid[5:]
    d = L[(L.aid == aid) & L.url.notna()].drop_duplicates("url")
    d = d[~d.url.str.contains(SKIP)].head(6)
    parts = []
    for r in d.itertuples():
        c = get(r.url)
        if c:
            t = to_text(c, r.url)
            if len(t.strip()) > 200:
                parts.append(f"\n\n##### DOC {r.title} | cat {r.category} | {r.doc_date} | {r.url}\n" + t[:300000])
    return uid, "".join(parts)


with ThreadPoolExecutor(8) as ex:
    OT = dict(ex.map(other_text, [u for u in top.uid if not pnum(u)]))

n = 0
for uid in I.uid:
    txt = (WT.get(pnum(uid), "") if pnum(uid) else "") + OT.get(uid, "")
    if txt.strip():
        (OUT / f"{safe(uid)}.txt").write_text(txt, encoding="utf-8"); n += 1
print("projects with harvested document text:", n)
