"""Score human validation (data/manual/validation_sheet.xlsx, sheets coder1/coder2) against automated codes.

Reports, once humans have filled the sheet:
  * inter-coder agreement: Cohen's kappa (target type, sector, ICT), Jaccard of municipality sets
  * automated vs consensus (coder1 where coders agree): accuracy/kappa for target type, sector, ICT;
    municipality-level precision / recall / F1 of geocoding (overall, excl. Dili, Dili only)
Output: outputs/tables/validation_scores.csv
"""
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
f = ROOT / "data" / "manual" / "validation_sheet.xlsx"
c1, c2 = pd.read_excel(f, "coder1"), pd.read_excel(f, "coder2")
done1, done2 = c1.h_target_type.notna(), c2.h_target_type.notna()
if done1.sum() == 0:
    raise SystemExit("validation sheet not yet coded (h_target_type empty)")

MUN_OF = {}  # auto_units like "Aileu[C]; Same[B]" -> municipality names via COD
import geopandas as gpd
a2 = gpd.read_file(ROOT / "data/raw/cod_ab/tls_admin_boundaries.shp/tls_admin2.shp")
p2m = dict(zip(a2.adm2_name, a2.adm1_name)); mset = set(a2.adm1_name)


def auto_muns(s):
    out = set()
    for tok in str(s).split(";"):
        nm = tok.split("[")[0].strip()
        if nm in mset: out.add(nm)
        elif nm in p2m: out.add(p2m[nm])
    return out


def hum_muns(s):
    return {x.strip() for x in str(s).split(";") if x.strip() and x.strip() != "nan"}


rows = []
both = done1 & done2
if both.sum():
    for col in ["h_target_type", "h_sector", "h_ict"]:
        rows.append({"metric": f"inter-coder kappa {col}", "value": cohen_kappa_score(c1.loc[both, col].astype(str), c2.loc[both, col].astype(str)), "n": int(both.sum())})
    jac = [len(hum_muns(a) & hum_muns(b)) / max(len(hum_muns(a) | hum_muns(b)), 1) for a, b in zip(c1.loc[both, "h_municipalities"], c2.loc[both, "h_municipalities"])]
    rows.append({"metric": "inter-coder mean Jaccard municipalities", "value": float(np.mean(jac)), "n": int(both.sum())})
ref = c1[done1].copy()
for col, auto in [("h_target_type", "target_type"), ("h_sector", "cat"), ("h_ict", "ict")]:
    a, h = ref[auto].astype(str).str.replace("True", "1").str.replace("False", "0"), ref[col].astype(str)
    rows.append({"metric": f"auto vs human accuracy {auto}", "value": float((a == h).mean()), "n": len(ref)})
    rows.append({"metric": f"auto vs human kappa {auto}", "value": cohen_kappa_score(a, h), "n": len(ref)})
for scope, keep in [("all", lambda m: True), ("excl. Dili", lambda m: m != "Dili"), ("Dili only", lambda m: m == "Dili")]:
    tp = fp = fn = 0
    for a, h in zip(ref.auto_units, ref.h_municipalities):
        A = {m for m in auto_muns(a) if keep(m)}; H = {m for m in hum_muns(h) if keep(m)}
        tp += len(A & H); fp += len(A - H); fn += len(H - A)
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    rows += [{"metric": f"geocoding precision ({scope})", "value": p, "n": tp + fp},
             {"metric": f"geocoding recall ({scope})", "value": r, "n": tp + fn},
             {"metric": f"geocoding F1 ({scope})", "value": 2 * p * r / max(p + r, 1e-9), "n": tp + fp + fn}]
out = pd.DataFrame(rows)
out.to_csv(ROOT / "outputs" / "tables" / "validation_scores.csv", index=False)
print(out.round(3).to_string(index=False))
