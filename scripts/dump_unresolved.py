"""Dump the largest unresolved projects (with full text snippet) for manual target-area coding."""
import sys
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC, INT = ROOT / "data" / "processed", ROOT / "data" / "interim"
PR = pd.read_csv(PROC / "oda_projects.csv")
iati = pd.read_csv(INT / "iati_fulltext.csv").drop_duplicates("aid").set_index("aid").fillna("")
U = PR[(PR.target_type == "unresolved") & (PR.start_year.between(2000, 2026))].copy()
U = U[~U.cat.isin(["OTHER"])].sort_values("usd", ascending=False).head(110).reset_index(drop=True)


def txt(r):
    if r.source == "IATI":
        return iati.text.get(r.uid[5:], "")
    return str(r.desc)


lo, hi = int(sys.argv[1]), int(sys.argv[2])
for i, r in U.iloc[lo:hi].iterrows():  # r["cat"]: .cat is a pandas accessor
    c = r["cat"]
    print(f"#{i} {r.uid} | {r.donor_group} | {c} | ${r.usd/1e6:.1f}M | {r.start_year:.0f}\n   T: {str(r.title)[:120]}\n   D: {txt(r)[:520]}")
U[["uid", "donor_group", "cat", "usd", "start_year", "title"]].to_csv(INT / "unresolved_top110.csv", index=False, encoding="utf-8-sig")
