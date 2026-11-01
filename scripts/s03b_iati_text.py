"""Stream-parse the full IATI XML: per activity, collect all narrative text (descriptions, location names,
result/indicator titles, document titles, sub-national region names) for target-area extraction.
Output: data/interim/iati_fulltext.csv (aid, n_chars, text[:30000], location_names)
"""
import csv, sys
from lxml import etree
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "raw" / ((sys.argv[1] + "/iati") if len(sys.argv) > 1 else "iati") / ("dportal_full_NP.xml" if len(sys.argv) > 1 else "dportal_full_TL.xml")
OUT = ROOT / "data" / "interim" / ("iati_fulltext_NP.csv" if len(sys.argv) > 1 else "iati_fulltext.csv")
csv.field_size_limit(10 ** 8)

n = 0
with open(OUT, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["aid", "n_chars", "location_names", "text"])
    for _, el in etree.iterparse(str(SRC), events=("end",), tag="iati-activity", huge_tree=True, recover=True):
        aid = (el.findtext("iati-identifier") or "").strip()
        parts, locs = [], []
        # contact-info excluded: office addresses ("UN House, Caicoli Street, Dili") are not project locations
        for tag in ["title", "description", "location", "result", "document-link", "recipient-region", "policy-marker", "sector"]:
            for sub in el.iter(tag):
                for nar in sub.iter("narrative"):
                    if nar.text:
                        parts.append(nar.text.strip())
                        if tag == "location":
                            locs.append(nar.text.strip())
        txt = " | ".join(dict.fromkeys(p for p in parts if p))
        w.writerow([aid, len(txt), " ; ".join(dict.fromkeys(locs))[:3000], txt[:30000]])
        n += 1
        el.clear()
        while el.getprevious() is not None:
            del el.getparent()[0]
print("activities parsed:", n)
