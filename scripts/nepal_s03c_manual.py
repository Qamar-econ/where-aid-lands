"""Apply Nepal manual target coding (top unresolved by USD). Run after nepal_s03_oda.py and nepal_dump_unresolved.py."""
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gazetteer_nepal import UPPER_DISTRICTS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "nepal"
PR, AS = pd.read_csv(OUT / "oda_projects.csv"), pd.read_csv(OUT / "oda_assignments.csv")
top = pd.read_csv(ROOT / "data/interim/nepal_unresolved_top.csv").reset_index().rename(columns={"index": "rank"})
codes = pd.read_csv(ROOT / "data/manual/nepal_manual_target_codes.csv").merge(top[["rank", "uid", "title"]], on="rank")
new = []
for r in codes.itertuples():
    if r.code == "located":
        ds = []
        for tok in r.districts.split(";"):
            ds += UPPER_DISTRICTS[tok] if tok.startswith("P:") else [tok]
        AS = AS[AS.uid != r.uid]
        new += [{"uid": r.uid, "level": "district", "unit": d, "weight": 1 / len(ds), "method": "manual",
                 "tier": "B" if r.confidence == "high" else "C"} for d in ds]
        PR.loc[PR.uid == r.uid, "target_type"] = "located"
    elif r.code in ("central", "nationwide", "regional"):
        PR.loc[PR.uid == r.uid, "target_type"] = {"central": "central_institution", "regional": "regional_multicountry"}.get(r.code, r.code)
AS = pd.concat([AS, pd.DataFrame(new)], ignore_index=True)
PR["manual_coded"] = PR.uid.isin(codes.uid)
PR.to_csv(OUT / "oda_projects.csv", index=False, encoding="utf-8-sig"); AS.to_csv(OUT / "oda_assignments.csv", index=False, encoding="utf-8-sig")
codes.to_csv(ROOT / "outputs/tables/nepal_manual_codes_with_titles.csv", index=False, encoding="utf-8-sig")
P2 = PR[PR.start_year.between(2000, 2026)]
print(codes.code.value_counts().to_dict())
print("USD share by target type:", (P2.groupby("target_type").usd.sum() / P2.usd.sum()).round(3).to_dict())
