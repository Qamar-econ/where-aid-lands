"""Assign GODAD Cambodia project locations to IPUMS GEO2_KH harmonized districts (143 units).

Unit of observation (spec 9-5, 9-7): the activity. EU/US CRS records of the same donor and title are one activity, dated by
its first commitment year (kh_common.activityize). Sample: activities starting 2017-2020 (planning ~ 2016-2019).
Precision (GODAD codebook 5b): ADM2-reliable = precision_crs == 3 (EU/US, CRS-based) or precision_code <= 3 (World Bank, China,
India). ADM1-only locations (precision_crs == 4 / precision_code == 4) are kept for a robustness check, split equally over the
province's units. Each location point is assigned to the unit containing it (points up to 10 km outside snap to the nearest
unit); an activity's weight is split equally over its distinct units (as GODAD's *_loc_evensplit).
Outputs: data/processed/cambodia/kh_projects.csv (activity level), kh_assign.csv (activity x unit, weights);
         kh_projects_records.csv / kh_assign_records.csv: the same built on CRS records, for comparison (spec 9-5)
"""
import sys
import numpy as np, pandas as pd, geopandas as gpd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kh_common import activityize, assign_units, donor_group

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/cambodia"
RAWK = pd.read_csv(ROOT / "data/raw/cambodia/godad_khm.csv", low_memory=False)
U = gpd.read_file(OUT / "kh_units.gpkg")[["GEOLEVEL2", "ADMIN_NAME", "PARENT", "geometry"]].rename(columns={"GEOLEVEL2": "unit"})


def build(K, suffix):
    K = K.copy()
    if "dgroup" not in K:
        K["dgroup"] = donor_group(K)
    K["prec"] = np.where(K.dgroup == "EU/US", np.select([K.precision_crs == 3, K.precision_crs == 4], ["adm2", "adm1"], "coarse"),
                         np.select([K.precision_code <= 3, K.precision_code == 4], ["adm2", "adm1"], "coarse"))
    K = K[K.startyear.between(2017, 2020)].copy()
    g = assign_units(K, U[["unit", "PARENT", "geometry"]])
    rows = []
    for pid, d in g.groupby("project_id"):
        units = sorted(set(d[(d.prec == "adm2") & d.unit.notna()].unit))
        for u in units:
            rows.append({"project_id": pid, "unit": u, "w": 1 / len(units), "level": "adm2"})
        a1 = d[(d.prec == "adm1") & d.PARENT.notna()]
        if a1.shape[0] and not units:  # ADM1-only activities: spread over the province's units (robustness only)
            us = U[U.PARENT.isin(sorted(set(a1.PARENT)))].unit.tolist()
            for u in us:
                rows.append({"project_id": pid, "unit": u, "w": 1 / len(us), "level": "adm1_spread"})
    A = pd.DataFrame(rows)
    P = K.groupby("project_id").agg(donor=("donor", "first"), dgroup=("dgroup", "first"), startyear=("startyear", "first"), title=("title", "first"),
                                     sector=("sector_categories", "first"), sector_code=("sector_codes", "first"), comm=("comm", "first"),
                                     flow=("flow_class", "first"), n_loc=("location_count", "first"),
                                     n_records=("record_id", "nunique") if "record_id" in K else ("project_id", "size"),
                                     best=("prec", lambda s: "adm2" if (s == "adm2").any() else ("adm1" if (s == "adm1").any() else "coarse"))).reset_index()
    P["n_units"] = P.project_id.map(A[A.level == "adm2"].groupby("project_id").unit.nunique()).fillna(0).astype(int)
    P.to_csv(OUT / f"kh_projects{suffix}.csv", index=False, encoding="utf-8-sig"); A.to_csv(OUT / f"kh_assign{suffix}.csv", index=False)
    a2 = A[A.level == "adm2"]
    cnt = a2.groupby("unit").w.sum().reindex(U.unit).fillna(0)
    print(f"[{suffix or 'activities'}] starting 2017-2020: {len(P)} | district-located: {a2.project_id.nunique()} | by donor group:",
          P[P.project_id.isin(a2.project_id)].dgroup.value_counts().to_dict(), "| zero units", int((cnt == 0).sum()),
          "| top-5 units share", round(cnt.nlargest(5).sum() / cnt.sum(), 3))
    return P, A


build(activityize(RAWK), "")
build(RAWK, "_records")
