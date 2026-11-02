"""Timing check for digital access (exploratory): later project windows, and 2019-only Ookla speed so that every test
precedes the 2020 starts. Randomisation inference is not run here; p_max = larger of clustered and Conley 50/100 km.
"""
import sys, math, numpy as np, pandas as pd, geopandas as gpd
from pathlib import Path
from scipy.stats import norm
ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "scripts/kh_s04_models.py").read_text(encoding="utf-8")
g = {"__file__": str(ROOT / "scripts/kh_s04_models.py")}; exec(src[:src.index("\nbase = A[A.level")], g)
A, outcome, ppml, conley_se = g["A"], g["outcome"], g["ppml"], g["conley_se"]
# Ookla 2019 Q1-Q4 only (all before 2020 starts)
def xy2ll(x, y, z=16):
    n = 2 ** z; return (x + .5) / n * 360 - 180, math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + .5) / n))))
U = gpd.read_file(ROOT / "data/processed/cambodia/kh_units.gpkg").rename(columns={"GEOLEVEL2": "unit"})
O = pd.concat([pd.read_csv(f) for f in sorted((ROOT / "data/raw/cambodia/ookla").glob("ookla_mobile_2019Q*_KH.csv"))])
ll = np.array([xy2ll(a, b) for a, b in zip(O.tx, O.ty)])
O = gpd.sjoin(gpd.GeoDataFrame(O, geometry=gpd.points_from_xy(ll[:, 0], ll[:, 1]), crs=4326), U[["unit", "geometry"]], predicate="within")
sp = O.groupby("unit").apply(lambda d: np.log(np.average(d.avg_d_kbps, weights=d.tests)), include_groups=False); sp.index = sp.index.astype(int)
zs = lambda s: (s - s.mean()) / s.std()
for lab, years in [("starts 2017-2020 (main)", range(2017, 2021)), ("starts 2019-2020", [2019, 2020]), ("starts 2020 only", [2020])]:
    a = A[(A.level == "adm2") & A.startyear.isin(years)]
    d = outcome(a); d = d.assign(sp19=zs(d.unit.astype(int).map(sp)))
    for v in ["phys", "conn", "sp19"]:
        r = ppml(f"N ~ {v} + lnpop + C(prov)", d); b = r.params[v]
        ps = [r.pvalues[v]] + [2 * norm.sf(abs(b / conley_se(r, d, v, c))) for c in (50, 100)]
        print(f"{lab:<26s} {v:<5s} {100*(np.exp(b)-1):+6.0f}%  p_max {max(ps):.3f}  projects {a.project_id.nunique()}")
