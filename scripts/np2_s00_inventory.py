"""Nepal 2019-2025 inventory: projects starting 2017-2025 (IATI + World Bank), sector, size, existing location info,
document links in IATI records (for document-based geocoding). Output: data/interim/np2_inventory.csv
"""
import csv, re
import numpy as np, pandas as pd
from lxml import etree
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
csv.field_size_limit(10 ** 8)
PR = pd.read_csv(ROOT / "data/processed/nepal/oda_projects.csv")
AS = pd.read_csv(ROOT / "data/processed/nepal/oda_assignments.csv")
P = PR[PR.start_year.between(2017, 2025) & PR.source.isin(["IATI", "WorldBank-API"])].copy()
best = AS.groupby("uid").apply(lambda d: "coord" if (d.method == "coord").any() else ("text" if (d.method == "text").any() else d.method.iloc[0]),
                               include_groups=False)
P["loc_method"] = P.uid.map(best)

# document links from the full IATI XML (only for the activities in scope)
aids = set(P.loc[P.source == "IATI", "uid"].str[5:])
docs = {}
for _, el in etree.iterparse(str(ROOT / "data/raw/nepal/iati/dportal_full_NP.xml"), events=("end",), tag="iati-activity", huge_tree=True, recover=True):
    aid = (el.findtext("iati-identifier") or "").strip()
    if aid in aids:
        links = []
        for d in el.iter("document-link"):
            url = d.get("url") or ""
            title = " ".join(n.text or "" for n in d.iter("narrative"))
            cat = ",".join(c.get("code") or "" for c in d.iter("category"))
            if url:
                links.append((url, title[:120], cat))
        docs[aid] = links
    el.clear()
    while el.getprevious() is not None:
        del el.getparent()[0]
P["n_docs"] = P.uid.str[5:].map(lambda a: len(docs.get(a, [])))
P["pdf_docs"] = P.uid.str[5:].map(lambda a: sum(1 for u, t, c in docs.get(a, []) if u.lower().endswith(".pdf") or "pdf" in u.lower()))
P["doc_urls"] = P.uid.str[5:].map(lambda a: " | ".join(u for u, t, c in docs.get(a, [])[:15]))
DEV = {"SOC_education", "SOC_health", "SOC_water", "SOC_other", "GOV", "PROD", "ENV_MULTI"}
P["dev"] = P["cat"].isin(DEV)
P.to_csv(ROOT / "data/interim/np2_inventory.csv", index=False, encoding="utf-8-sig")
d = P[P.dev]
print("projects starting 2017-2025 (IATI+WB):", len(P), "| development (non-infra):", len(d))
print("by start year:", d.start_year.value_counts().sort_index().to_dict())
print("target type:", d.target_type.value_counts().to_dict())
print("location method:", d.loc_method.value_counts().to_dict())
print("with >=1 document link:", (d.n_docs > 0).mean().round(2), "| with pdf:", (d.pdf_docs > 0).mean().round(2))
print("top reporters:", d.donor.value_counts().head(12).to_dict())
d2 = d.sort_values("usd", ascending=False)
cum = d2.usd.fillna(0).cumsum() / d2.usd.fillna(0).sum()
print("projects covering 80% / 90% of USD:", int((cum <= 0.8).sum()) + 1, int((cum <= 0.9).sum()) + 1)
