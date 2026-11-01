"""Assign GODAD Cambodia project locations to IPUMS GEO2_KH harmonized districts (143 units).

Sample: projects starting 2017-2020 (planning ~ 2016-2019, aligned with the 2019 census and Ookla 2019).
Precision (GODAD codebook 5b): ADM2-reliable = precision_crs == 3 (EU/US, CRS-based) or precision_code <= 3 (World Bank, China,
India). ADM1-only locations (precision_crs == 4 / precision_code == 4) are kept for a robustness check, split equally over the
province's units. Each location point is assigned to the unit containing it; a project's weight is split equally over its
distinct units (as GODAD's *_loc_evensplit).
Outputs: data/processed/cambodia/kh_projects.csv (project level), kh_assign.csv (project x unit, weights)
"""
import numpy as np, pandas as pd, geopandas as gpd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/cambodia"
K = pd.read_csv(ROOT / "data/raw/cambodia/godad_khm.csv", low_memory=False)
U = gpd.read_file(OUT / "kh_units.gpkg")[["GEOLEVEL2", "ADMIN_NAME", "PARENT", "geometry"]].rename(columns={"GEOLEVEL2": "unit"})
K["dgroup"] = np.select([K.donor.eq("World Bank"), K.donor.str.contains("China", na=False), K.donor.str.contains("India", na=False)],
                        ["World Bank", "China", "India"], "EU/US")
K["prec"] = np.where(K.dgroup == "EU/US", np.select([K.precision_crs == 3, K.precision_crs == 4], ["adm2", "adm1"], "coarse"),
                     np.select([K.precision_code <= 3, K.precision_code == 4], ["adm2", "adm1"], "coarse"))
K = K[K.startyear.between(2017, 2020)].copy()
g = gpd.GeoDataFrame(K, geometry=gpd.points_from_xy(K.longitude, K.latitude), crs=4326)
g = gpd.sjoin(g, U[["unit", "PARENT", "geometry"]], predicate="within", how="left")
# points just outside a polygon (coast / border): nearest unit within 5 km
miss = g.unit.isna() & g.longitude.notna()
if miss.any():
    nn = gpd.sjoin_nearest(g[miss].drop(columns=["unit", "PARENT", "index_right"]).to_crs(32648), U.to_crs(32648)[["unit", "PARENT", "geometry"]],
                           max_distance=5000, how="left")
    g.loc[miss, ["unit", "PARENT"]] = nn[["unit", "PARENT"]].values
rows = []
for pid, d in g.groupby("project_id"):
    a2 = d[(d.prec == "adm2") & d.unit.notna()]
    units = sorted(set(a2.unit))
    for u in units:
        rows.append({"project_id": pid, "unit": u, "w": 1 / len(units), "level": "adm2"})
    a1 = d[(d.prec == "adm1") & d.PARENT.notna()]
    if a1.shape[0]:  # ADM1-only locations: spread over the province's units (robustness only)
        provs = sorted(set(a1.PARENT)); us = U[U.PARENT.isin(provs)].unit.tolist()
        for u in us:
            rows.append({"project_id": pid, "unit": u, "w": 1 / len(us), "level": "adm1_spread"})
A = pd.DataFrame(rows)
P = K.groupby("project_id").agg(donor=("donor", "first"), dgroup=("dgroup", "first"), startyear=("startyear", "first"), title=("title", "first"),
                                 sector=("sector_categories", "first"), sector_code=("sector_codes", "first"), comm=("comm", "first"),
                                 flow=("flow_class", "first"), n_loc=("location_count", "first"),
                                 best=("prec", lambda s: "adm2" if (s == "adm2").any() else ("adm1" if (s == "adm1").any() else "coarse"))).reset_index()
P["n_units"] = P.project_id.map(A[A.level == "adm2"].groupby("project_id").unit.nunique()).fillna(0).astype(int)
P.to_csv(OUT / "kh_projects.csv", index=False, encoding="utf-8-sig"); A.to_csv(OUT / "kh_assign.csv", index=False)
print("projects starting 2017-2020:", len(P), "| by donor group:", P.dgroup.value_counts().to_dict())
print("best precision:", P.best.value_counts().to_dict(), "| by start year (adm2):", P[P.best == "adm2"].startyear.value_counts().sort_index().to_dict())
a2 = A[A.level == "adm2"]
print("ADM2-located projects:", a2.project_id.nunique(), "| units reached:", a2.unit.nunique(), "of", len(U),
      "| projects in >5 units:", (P.n_units > 5).sum())
cnt = a2.groupby("unit").w.sum().reindex(U.unit).fillna(0)
print("projects per unit: mean", round(cnt.mean(), 2), "| zero units", int((cnt == 0).sum()), "| top-5 units share", round(cnt.nlargest(5).sum() / cnt.sum(), 3))
print("top units:", U.set_index("unit").ADMIN_NAME.reindex(cnt.nlargest(5).index).tolist())
