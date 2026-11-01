"""Audit text geoparsing: which terms trigger matches, how often, with context snippets (false-positive check)."""
import sys, re, collections
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gazetteer as gz  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INT, PROC = ROOT / "data" / "interim", ROOT / "data" / "processed"
PR = pd.read_csv(PROC / "oda_projects.csv")
AS = pd.read_csv(PROC / "oda_assignments.csv")
txt_uids = set(AS.loc[AS.method == "text", "uid"])
iati = pd.read_csv(INT / "iati_fulltext.csv").drop_duplicates("aid").set_index("aid").fillna("")
bb = pd.read_csv(INT / "bb5_rows.csv")


def text_of(r):
    if r.source == "IATI":
        a = r.uid[5:]
        return f"{r.title} {r.desc} {iati.location_names.get(a, '')} {iati.text.get(a, '')}"
    return f"{r.title} {r.desc}"


pats = [(pc, v, gz._pat(v)) for pc, vs in gz.POSTO.items() for v in vs] + [(mc, v, gz._pat(v)) for mc, vs in gz.MUN.items() for v in vs]
hits = collections.Counter(); ex = {}
for r in PR[PR.uid.isin(txt_uids)].itertuples():
    t = text_of(r)
    for code, v, p in pats:
        m = p.search(t)
        if m:
            hits[v] += 1
            if v not in ex:
                s = max(0, m.start() - 60)
                ex[v] = t[s:m.end() + 60].replace("\n", " ")
out = pd.DataFrame([{"term": k, "n": n, "example": ex[k]} for k, n in hits.most_common()])
out.to_csv(ROOT / "outputs" / "tables" / "geoparse_audit.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250, "display.max_colwidth", 140)
print(out.head(80).to_string())
