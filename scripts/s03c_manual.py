"""Apply manual target-area coding of the 110 largest unresolved projects (data/manual/manual_target_codes.csv).

located -> municipality-level assignments (method 'manual'); central / nationwide / regional -> target_type;
drop -> removed (aggregate rows); unknown -> stays unresolved.
Run after s03_oda.py.
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC, INT, MAN = ROOT / "data" / "processed", ROOT / "data" / "interim", ROOT / "data" / "manual"
PR = pd.read_csv(PROC / "oda_projects.csv")
AS = pd.read_csv(PROC / "oda_assignments.csv")
top = pd.read_csv(INT / "unresolved_top110.csv").reset_index().rename(columns={"index": "rank"})
codes = pd.read_csv(MAN / "manual_target_codes.csv").merge(top[["rank", "uid", "title"]], on="rank")
MUNI = {"Aileu": "TL02", "Ainaro": "TL01", "Baucau": "TL03", "Bobonaro": "TL04", "Covalima": "TL05", "Dili": "TL06",
        "Ermera": "TL07", "Liquiçá": "TL08", "Lautém": "TL09", "Manufahi": "TL10", "Manatuto": "TL11", "Oecussi": "TL12", "Viqueque": "TL13"}
new_rows = []
for r in codes.itertuples():
    if r.code == "located":
        ms = [MUNI[m] for m in r.municipalities.split(";")]
        AS = AS[AS.uid != r.uid]
        new_rows += [{"uid": r.uid, "level": "municipio", "unit": m, "weight": 1 / len(ms), "method": "manual",
                      "tier": "B" if r.confidence == "high" else "C"} for m in ms]
        PR.loc[PR.uid == r.uid, "target_type"] = "located"
    elif r.code in ("central", "nationwide", "regional"):
        PR.loc[PR.uid == r.uid, "target_type"] = {"central": "central_institution", "regional": "regional_multicountry"}.get(r.code, r.code)
    elif r.code == "drop":
        PR = PR[PR.uid != r.uid]; AS = AS[AS.uid != r.uid]
AS = pd.concat([AS, pd.DataFrame(new_rows)], ignore_index=True)
PR["manual_coded"] = PR.uid.isin(codes.uid)
PR.to_csv(PROC / "oda_projects.csv", index=False, encoding="utf-8-sig")
AS.to_csv(PROC / "oda_assignments.csv", index=False, encoding="utf-8-sig")
codes.to_csv(ROOT / "outputs" / "tables" / "manual_target_codes_with_titles.csv", index=False, encoding="utf-8-sig")
print(codes.code.value_counts())
print("target types after manual coding:\n", PR.target_type.value_counts())
P2 = PR[PR.start_year.between(2000, 2026)]
w = P2.groupby("target_type").usd.sum() / P2.usd.sum() * 100
print("share of USD by target type (%):\n", w.round(1))
