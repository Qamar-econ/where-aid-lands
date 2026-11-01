"""Largest unresolved Nepal projects (with text snippet) for manual target-area coding."""
import sys
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = pd.read_csv(ROOT / "data/processed/nepal/oda_projects.csv")
ft = pd.read_csv(ROOT / "data/interim/iati_fulltext_NP.csv").drop_duplicates("aid").set_index("aid").text.fillna("")
U = P[P.start_year.between(2000, 2026) & (P.target_type == "unresolved")].sort_values("usd", ascending=False)
top = U.head(int(sys.argv[1]) if len(sys.argv) > 1 else 60).reset_index(drop=True)
print(f"top {len(top)} = {top.usd.sum() / U.usd.sum():.1%} of unresolved USD")
for i, r in top.iterrows():
    t = ft.get(r.uid[5:], r.desc) if r.uid.startswith("IATI-") else r.desc
    print(f"#{i} {r.uid} | {r.donor_group} | {r['cat']} | ${(r.usd or 0) / 1e6:.0f}M | {r.start_year:.0f} | {str(r.title)[:90]}\n    {str(t)[:300]}")
top[["uid", "donor_group", "cat", "usd", "start_year", "title"]].to_csv(ROOT / "data/interim/nepal_unresolved_top.csv", index=False, encoding="utf-8-sig")
