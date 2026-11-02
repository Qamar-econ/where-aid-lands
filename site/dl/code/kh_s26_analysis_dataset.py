"""One table with everything the main models use, one row per district (the 141 analysis units).

Columns
  unit, name, prov                district code (IPUMS GEO2_KH), name, province code
  lat, lon, x_km, y_km            district centroid (WGS84; UTM 48N km, used for Conley distances)
  pop_2019, lnpop                 WorldPop 2019 population and its log
  N, N_social, N_infra            new aid activities starting 2017-2020, district-located, ICT excluded (all; social; transport/energy)
  M                               their commitments, US$ million (2014 prices)
  access_pc1, phys                physical-access index (raw PC1 from kh_access.csv) and standardised over the 141 districts
  mob_mbps_shrunk, conn           mobile download speed (Mbps, shrunk) and standardised log
  ntl_pc_2015, lights             night lights per person 2015 and standardised log(1 + 1000 x)
  rwi, need                       Meta Relative Wealth Index (district mean) and poverty = standardised -RWI
  prior_infra                     1 if the district had a transport/energy project starting 2010-2016
Output: data/processed/cambodia/kh_analysis.csv
"""
import numpy as np, pandas as pd, geopandas as gpd
from pathlib import Path

SRC = Path(__file__).with_name("kh_s06_ladder.py")
ns = {"__file__": str(SRC)}
exec(SRC.read_text(encoding="utf-8").split("LADDER = [")[0], ns)
DF, cur, outcome, ROOT = ns["DF"], ns["cur"], ns["outcome"], ns["ROOT"]
PROC = ROOT / "data/processed/cambodia"
z = lambda s: (s - s.mean()) / s.std()

d = outcome(cur)
for fam, col in [("social", "N_social"), ("infra", "N_infra")]:
    d[col] = d.unit.map(outcome(cur[cur.family == fam]).set_index("unit").N).fillna(0)
U = gpd.read_file(PROC / "kh_units.gpkg"); U["unit"] = U.GEOLEVEL2.astype("int64")
c_ll = U.geometry.representative_point(); c_utm = U.to_crs(32648).geometry.centroid
G = pd.DataFrame({"unit": U.unit, "lat": c_ll.y.values, "lon": c_ll.x.values, "x_km": c_utm.x.values / 1000, "y_km": c_utm.y.values / 1000})
d = d.drop(columns=[c for c in ["x_km", "y_km", "lat", "lon"] if c in d]).merge(G, on="unit", how="left")
rw = pd.read_csv(PROC / "kh_rwi.csv").set_index("unit").rwi
d["rwi"] = d.unit.map(rw); d["need"] = z(-d.rwi)
d["phys"] = d["access"]
out = d[["unit", "ADMIN_NAME", "prov", "lat", "lon", "x_km", "y_km", "pop_2019", "lnpop", "N", "N_social", "N_infra", "M",
         "access_pc1", "phys", "mob_mbps_shrunk", "conn", "ntl_pc_2015", "lights", "rwi", "need", "prior_infra"]].rename(columns={"ADMIN_NAME": "name"})
out.to_csv(PROC / "kh_analysis.csv", index=False, encoding="utf-8")
print(out.shape, "| projects", round(out.N.sum(), 1)); print(out.describe().T[["mean", "min", "max"]].round(2).to_string())

# the published numbers, as a separate file, so kh_replicate.py can set an independent re-estimate beside them
T = ROOT / "outputs/tables"
mn = pd.read_csv(T / "kh_main.csv").set_index("var"); nd = pd.read_csv(T / "kh_need_conservative.csv").iloc[0]
jt = pd.read_csv(T / "kh_joint_test.csv").iloc[0]; pca = pd.read_csv(T / "kh_access_pca.csv")
pub = [{"check": f"single:{v}", "effect_pct": mn.loc[v, "pct"], "p_cluster": mn.loc[v, "p_cluster"], "p_conley50": mn.loc[v, "p_conley50"],
        "p_conley100": mn.loc[v, "p_conley100"], "p_ri": mn.loc[v, "p_ri"], "p_reported": mn.loc[v, "p_conservative"]} for v in ["phys", "conn", "ntl"]]
pub[-1]["check"] = "single:lights"
pub.append({"check": "single:need", "effect_pct": nd.pct, "p_cluster": nd.p_cluster, "p_conley50": nd.p_conley50, "p_conley100": nd.p_conley100,
            "p_ri": nd.p_ri, "p_reported": nd.p_conservative})
pub.append({"check": "joint:four_factors", "effect_pct": None, "p_cluster": jt.p_wald_cluster, "p_conley100": jt.p_wald_conley100, "p_ri": jt.p_ri,
            "p_reported": jt.p_conservative, "statistic": jt.wald_cluster})
pub.append({"check": "index:pc1_share", "statistic": pca.pc1_var_share.iloc[0]})
pd.DataFrame(pub).to_csv(PROC / "kh_published.csv", index=False)
print(pd.DataFrame(pub).round(4).to_string(index=False))
