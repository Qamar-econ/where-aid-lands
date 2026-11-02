"""Poverty (-RWI) alone with the full inference rule of the main estimates (kh_s04.full: clustered, Conley 50/100 km,
Freedman-Lane randomisation inference with 2,000 draws; most conservative p). kh_s10 reported the clustered p only.
Output: outputs/tables/kh_need_conservative.csv
"""
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "scripts/kh_s04_models.py").read_text(encoding="utf-8")
exec(src[:src.index("\nbase = A[A.level")])
rw = pd.read_csv(ROOT / "data/processed/cambodia/kh_rwi.csv").set_index("unit").rwi
d = outcome(A[A.level == "adm2"])
d["need"] = -d.unit.map(rw); d["need"] = (d.need - d.need.mean()) / d.need.std()
d = d[d.need.notna()]
R = pd.DataFrame([full(d, "need")]); R.to_csv(T / "kh_need_conservative.csv", index=False)
print(R[["var", "pct", "p_cluster", "p_conley50", "p_conley100", "p_ri", "p_conservative"]].round(4).to_string(index=False))
