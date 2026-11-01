"""One-page project summaries (PDF) for human coding: the most recent information per project from IATI, World Bank WDS
documents, ADB Project Data Sheets and harvested documents; with the machine codes (planning start, locations) and blank
boxes for the human coder.  Output: outputs/project_summaries/np2_project_summaries.pdf (one page per project, bookmarked)
and np2_summary_index.csv (page -> uid).  Usage: np2_s05_summaries.py [TOPN=400]
"""
import json, re, sys, html
import pandas as pd
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak, KeepInFrame
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parents[1]
TOPN = int(sys.argv[1]) if len(sys.argv) > 1 else 400
OUT = ROOT / "outputs/project_summaries"; OUT.mkdir(parents=True, exist_ok=True)
pdfmetrics.registerFont(TTFont("KR", "C:/Windows/Fonts/malgun.ttf"))
pdfmetrics.registerFont(TTFont("KRB", "C:/Windows/Fonts/malgunbd.ttf"))
st = lambda size=7.6, bold=False, color=colors.black, lead=None: ParagraphStyle("s", fontName="KRB" if bold else "KR", fontSize=size,
                                                                                leading=lead or size * 1.28, textColor=color)
S_T, S_H, S_B, S_S = st(12, True), st(8.6, True, colors.HexColor("#1f4e79")), st(7.6), st(6.6, color=colors.HexColor("#444444"))

I = pd.read_csv(ROOT / "data/interim/np2_inventory.csv"); I = I[I.dev].sort_values("usd", ascending=False).head(TOPN)
PL = pd.read_csv(ROOT / "data/processed/nepal/np2_planning.csv").set_index("uid")
G = pd.read_csv(ROOT / "data/processed/nepal/np2_geocodes.csv")
ST = pd.read_csv(ROOT / "data/processed/nepal/np2_project_status.csv").set_index("uid")
M = pd.read_csv(ROOT / "data/interim/np2_iati_meta.csv").drop_duplicates("aid").set_index("aid")
L = pd.read_csv(ROOT / "data/interim/np2_doclinks.csv")
W = pd.read_csv(ROOT / "data/interim/np2_wb_docs.csv")
A = json.loads((ROOT / "data/interim/adb_pds_milestones.json").read_text(encoding="utf-8"))
DOCS = ROOT / "data/interim/np2_doctext"
PREC = {1: "1 exact point", 2: "2 village/town", 3: "3 palika", 4: "4 district", 6: "6 province/region", 8: "8 national/central"}
STATUS = {1: "pipeline", 2: "implementation", 3: "finalisation", 4: "closed", 5: "cancelled", 6: "suspended"}
e = lambda s, n=None: html.escape(("" if (not isinstance(s, str) and pd.isna(s)) or str(s) in ("nan", "NaT") else str(s))[:n] if n else ("" if (not isinstance(s, str) and pd.isna(s)) or str(s) in ("nan", "NaT") else str(s)))
safe = lambda u: re.sub(r"[^A-Za-z0-9_-]", "_", u)


def objective(text, fallback):
    for pat in [r"(?:Project|Program|Programme) Development Objectives?\s*(?:\(PDO\))?\s*[:.]?\s*(?:The\s+)?(?:objective|PDO)?\s*(?:of the \w+)?\s*(?:is|are)\s+to\s+(.{40,450}?\.)",
                r"(?:objective|purpose) of the (?:project|program|programme) (?:is|are) to (.{40,450}?\.)", r"[Ii]mpact[:\s]+(.{40,300}?\.)"]:
        m = re.search(pat, text, re.S)
        if m:
            return "to " + re.sub(r"\s+", " ", m.group(1))
    return re.sub(r"\s+", " ", str(fallback))[:900]


def page(r):
    aid = r.uid[5:] if r.uid.startswith("IATI-") else None
    m = M.loc[aid] if aid in M.index else pd.Series(dtype=object)
    pl = PL.loc[r.uid] if r.uid in PL.index else pd.Series(dtype=object)
    doct = (DOCS / f"{safe(r.uid)}.txt")
    text = doct.read_text(encoding="utf-8", errors="ignore") if doct.exists() else ""
    pn = re.search(r"(P\d{6})", r.uid); adb = re.search(r"46004-(\d{5}-\d{3})", r.uid)
    F = []
    F.append(Paragraph(e(r.title, 160), S_T))
    F.append(Paragraph(f"{e(r.donor)} · {e(r.uid)} · {e(r.cat)} · DAC {e(r.sector_code)}", S_S))
    F.append(Spacer(1, 3))
    # key facts table
    stt = STATUS.get(int(m.get("status")), "") if pd.notna(m.get("status", None)) else ""
    facts = [["Budget (USD)", f"{r.usd:,.0f}" if pd.notna(r.usd) else "", "Status (IATI)", stt],
             ["Start (actual/planned)", e(str(pl.get("start_date", ""))[:10]), "End (planned/actual)", e(m.get("date_4", m.get("date_3", "")))],
             ["Approval / 1st commitment", e(str(pl.get("approval_date", ""))[:10]) or e(str(pl.get("commit_date", ""))[:10]), "IATI last updated", e(str(m.get("last_updated", ""))[:10])]]
    if adb and adb.group(1) in A:
        v = A[adb.group(1)]
        facts.append(["ADB concept clearance", e(v[0]), "ADB fact-finding / approval", f"{e(v[1])} / {e(v[2])}"])
    t = Table([[Paragraph(str(c), S_B if j % 2 else st(7.4, True)) for j, c in enumerate(row)] for row in facts], colWidths=[34 * mm, 55 * mm, 38 * mm, 55 * mm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#bbbbbb")), ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef3f8")),
                           ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#eef3f8")), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    F += [t, Spacer(1, 4)]
    F.append(Paragraph("기획 착수 시점 (machine code)", S_H))
    F.append(Paragraph(f"<b>{e(str(pl.get('planning_date', ''))[:10])}</b> (year {e(pl.get('planning_year'))}) · method <b>{e(pl.get('method'))}</b> · "
                       f"lag to start {e(pl.get('lag_years'))} y<br/>{e(pl.get('evidence'), 300)}", S_B))
    F.append(Paragraph("O1 ADB concept clearance · O1a ADB fact-finding adj. · O2 WB concept-stage PID · O3 IATI planning document (upper bound) · "
                       "S1 commitment minus median approval lag · E1 start minus median lag", S_S))
    F.append(Spacer(1, 3))
    F.append(Paragraph("목적 / 내용", S_H))
    F.append(Paragraph(e(objective(text, r.desc if pd.notna(r.desc) else ""), 900), S_B))
    F.append(Spacer(1, 3))
    F.append(Paragraph("위치 (machine code)", S_H))
    s = ST.loc[r.uid] if r.uid in ST.index else None
    g = G[G.uid == r.uid]
    if s is not None:
        F.append(Paragraph(f"Best precision: <b>{PREC.get(int(s.best_precision), s.best_precision)}</b> · districts: {int(s.n_districts)} · sources: {e(s.sources)}", S_B))
    if len(g):
        units = ", ".join(sorted({f"{d}" + (f" ({t})" if lv in ('palika', 'village', 'point') else "") for d, t, lv in zip(g.district, g.term, g.level)}))
        F.append(Paragraph("Units: " + e(units, 700), S_B))
        for sn in g.drop_duplicates("snippet").snippet.dropna().head(4):
            F.append(Paragraph("“" + e(sn, 220) + "”", S_S))
    if adb and adb.group(1) in A and A[adb.group(1)][3]:
        F.append(Paragraph("ADB PDS geographical location: " + e(A[adb.group(1)][3], 300), S_B))
    F.append(Spacer(1, 3))
    # documents / latest info
    F.append(Paragraph("문서 (최신순)", S_H))
    rows = []
    if pn:
        d = W[W.pnum == pn.group(1)].sort_values("docdt", ascending=False).head(7)
        rows = [f"{x.docdt} · {e(x.docty)} · {e(x.txturl, 110)}" for x in d.itertuples()]
    elif aid:
        d = L[L.aid == aid].drop_duplicates("url").sort_values("doc_date", ascending=False, na_position="last").head(7)
        rows = [f"{e(x.doc_date) or 'n.d.'} · [{e(x.category)}] {e(x.title, 60)} · {e(x.url, 100)}" for x in d.itertuples()]
    for x in rows or ["(no linked documents)"]:
        F.append(Paragraph(x, S_S))
    if pd.notna(m.get("participating", None)):
        F.append(Paragraph("Participating orgs: " + e(m.get("participating"), 300), S_S))
    F.append(Spacer(1, 5))
    chk = Table([[Paragraph("<b>Human coder</b>", S_B), Paragraph("위치 확인 [  ] 맞음 [  ] 수정: ____________________", S_B),
                  Paragraph("기획 시점 [  ] 맞음 [  ] 수정: __________", S_B), Paragraph("코더/날짜: ________", S_B)]],
                colWidths=[22 * mm, 70 * mm, 55 * mm, 35 * mm])
    chk.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#1f4e79")), ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff8e5"))]))
    F.append(chk)
    return [KeepInFrame(182 * mm, 272 * mm, F, mode="shrink"), PageBreak()]


class Doc(SimpleDocTemplate):
    def afterFlowable(self, f):
        if isinstance(f, KeepInFrame) and getattr(f, "_uid", None):
            self.canv.bookmarkPage(f._uid); self.canv.addOutlineEntry(f._label, f._uid, level=0)


story, idx = [], []
for n, r in enumerate(I.itertuples(), 1):
    fl = page(r)
    fl[0]._uid = safe(r.uid); fl[0]._label = f"{n:03d} {str(r.title)[:60]}"
    story += fl
    idx.append({"page": n, "uid": r.uid, "donor": r.donor, "title": r.title, "usd": r.usd})
doc = Doc(str(OUT / "np2_project_summaries.pdf"), pagesize=A4, leftMargin=14 * mm, rightMargin=14 * mm, topMargin=12 * mm, bottomMargin=12 * mm,
          title="Nepal development projects 2017-2025: one-page summaries")
doc.build(story)
pd.DataFrame(idx).to_csv(OUT / "np2_summary_index.csv", index=False, encoding="utf-8-sig")
print("pages:", len(idx))

