"""What is a 'project' in GODAD? For EU/US donors the project_id is a CRS record (CrsID x donor x recipient), and the CRS
reports one activity as several records: one per year and per purpose code, including disbursement-only rows (comm = 0).
Counting records therefore weights multi-year, multi-sector activities several times. This script rebuilds the outcome
with activities as the unit and reruns the main models.

  D0 records      current definition (kh_projects.csv / kh_assign.csv)
  D1 activities   EU/US records grouped by donor + normalised title; World Bank and China keep their project_id.
                  Locations = union of the activity's district-reliable units in 2017-2020; weight 1/K over that union.
  D2 new only     D1 restricted to activities whose first record in all GODAD years starts in 2017-2020
                  (drops continuations of activities committed before 2017)
  D1 money        commitments summed over the activity's 2017-2020 records, split over its units
Outputs: outputs/tables/kh_dedup_main.csv, kh_dedup_summary.csv, data/processed/cambodia/kh_activities.csv
"""
import re, unicodedata
import numpy as np, pandas as pd, geopandas as gpd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "scripts/kh_s04_models.py").read_text(encoding="utf-8")
exec(src[:src.index("base = A[A.level")])          # data, outcome(), full(), quick(), theta() exactly as in the main script

OUT = ROOT / "data/processed/cambodia"
K = pd.read_csv(ROOT / "data/raw/cambodia/godad_khm.csv", low_memory=False)
UG = gpd.read_file(OUT / "kh_units.gpkg")[["GEOLEVEL2", "geometry"]].rename(columns={"GEOLEVEL2": "unit"})
UG["unit"] = UG.unit.astype("int64")
K["dgroup"] = np.select([K.donor.eq("World Bank"), K.donor.str.contains("China", na=False), K.donor.str.contains("India", na=False)],
                        ["World Bank", "China", "India"], "EU/US")
K["prec"] = np.where(K.dgroup == "EU/US", np.where(K.precision_crs == 3, "adm2", "other"), np.where(K.precision_code <= 3, "adm2", "other"))


def norm_title(t):
    t = unicodedata.normalize("NFKD", str(t)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", t)).strip()


K["key"] = np.where(K.dgroup == "EU/US", K.donor + " | " + K.title.map(norm_title), K.dgroup + " | " + K.project_id.astype(str))
first = K.groupby("key").startyear.min()

W = K[K.startyear.between(2017, 2020)].copy()
g = gpd.GeoDataFrame(W, geometry=gpd.points_from_xy(W.longitude, W.latitude), crs=4326)
g = gpd.sjoin(g, UG, predicate="within", how="left")
miss = g.unit.isna() & g.longitude.notna()
if miss.any():
    nn = gpd.sjoin_nearest(g[miss].drop(columns=["unit", "index_right"]).to_crs(32648), UG.to_crs(32648), max_distance=5000, how="left")
    g.loc[miss, "unit"] = nn["unit"].values
g = g[(g.prec == "adm2") & g.unit.notna()]

rec = W.drop_duplicates("project_id")                 # one row per record
acts = rec.groupby("key").agg(dgroup=("dgroup", "first"), donor=("donor", "first"), title=("title", "first"), n_records=("project_id", "size"),
                              start_window=("startyear", "min"), comm=("comm", lambda s: s.fillna(0).clip(lower=0).sum()),
                              comm_missing=("comm", lambda s: s.isna().all()), sector=("sector_categories", "first")).reset_index()
acts["first_ever"] = acts.key.map(first)
units = g.groupby("key").unit.apply(lambda s: sorted(set(s.astype("int64"))))
acts["units"] = acts.key.map(units)
acts = acts[acts.units.notna()].copy()
acts["n_units"] = acts.units.str.len()
AA = acts.explode("units").rename(columns={"units": "unit"})
AA["unit"] = AA.unit.astype("int64"); AA["w"] = 1 / AA.n_units
acts.drop(columns="units").to_csv(OUT / "kh_activities.csv", index=False, encoding="utf-8-sig")

base = A[A.level == "adm2"]
defs = {"D0 records (current)": outcome(base),
        "D1 activities": outcome(AA),
        "D2 activities new in 2017-20": outcome(AA[AA.first_ever >= 2017]),
        "D1 activities, binary (1 per district)": outcome(AA.assign(one=1.0), "one")}
money = {"D0 records, US$m": outcome(base.assign(um=base.comm.fillna(0).clip(lower=0) * base.w / 1e6), "um"),
         "D1 activities, US$m": outcome(AA.assign(um=AA.comm * AA.w / 1e6), "um")}

summ = [{"definition": "D0 records (current)", "units_counted": base.project_id.nunique(), "total_N": round(defs["D0 records (current)"].N.sum(), 1)},
        {"definition": "D1 activities", "units_counted": acts.shape[0], "total_N": round(defs["D1 activities"].N.sum(), 1)},
        {"definition": "D2 activities new in 2017-20", "units_counted": int((acts.first_ever >= 2017).sum()), "total_N": round(defs["D2 activities new in 2017-20"].N.sum(), 1)}]
print(pd.DataFrame(summ).to_string(index=False))
print("records per activity (EU/US):", acts[acts.dgroup == "EU/US"].n_records.describe().round(2).to_dict())
print("top activities by records:\n", acts.nlargest(8, "n_records")[["donor", "title", "n_records", "first_ever", "n_units"]].to_string(index=False))
print("activities started before 2017 (continuations):", int((acts.first_ever < 2017).sum()), "| by donor group:",
      acts[acts.first_ever < 2017].dgroup.value_counts().to_dict())
d0, d1 = defs["D0 records (current)"], defs["D1 activities"]
print("district N correlation D0 vs D1:", round(d0.N.corr(d1.N), 3), "| spearman", round(d0.N.corr(d1.N, method="spearman"), 3))

rows = []
for name, d in defs.items():
    for v in ["combined", "phys", "conn", "ntl"]:
        r = full(d, v, ri=name.startswith("D1 activities") and "binary" not in name)
        rows.append({"definition": name, **r})
    rows.append({"definition": name, "var": "theta", **{k: v for k, v in theta(d).items()}})
for name, d in money.items():
    q = quick(d)
    rows.append({"definition": name, "var": "money_quick", **q})
R = pd.DataFrame(rows)
R.to_csv(T / "kh_dedup_main.csv", index=False); pd.DataFrame(summ).to_csv(T / "kh_dedup_summary.csv", index=False)
cols = ["definition", "var", "pct", "p_cluster", "p_conley100", "p_ri", "p_conservative", "ci_lo_pct", "ci_hi_pct", "theta"]
print(R[[c for c in cols if c in R]].round(3).to_string(index=False))
print(R[R["var"] == "money_quick"][["definition", "combined_pct", "combined_p", "phys_pct", "phys_p", "conn_pct", "conn_p", "ntl_pct", "ntl_p"]].round(3).to_string(index=False))
