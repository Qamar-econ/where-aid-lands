"""Shared GODAD conventions for the Cambodia pipeline (spec 9-5, 9-7).

activityize(K): GODAD's EU/US project_id is a CRS record (one per year and purpose code). Records of the same donor with the
same normalised title are one activity. World Bank and China ids are already projects. After the call, every location row
carries the activity's id (project_id), its first commitment year (startyear), its total commitment over all records (comm)
and the sector codes of its earliest record (sector_codes). The original record id is kept as record_id.
assign_units(K, U): point-in-polygon district assignment, with points up to 10 km outside every unit (coast, border, offshore
geocodes) snapped to the nearest unit.
"""
import re, unicodedata
import numpy as np, pandas as pd, geopandas as gpd

SNAP_KM = 10


def norm_title(t):
    t = unicodedata.normalize("NFKD", str(t)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", t)).strip()


def donor_group(K):
    return np.select([K.donor.eq("World Bank"), K.donor.str.contains("China", na=False), K.donor.str.contains("India", na=False)],
                     ["World Bank", "China", "India"], "EU/US")


def activityize(K):
    K = K.copy()
    K["dgroup"] = donor_group(K)
    K["record_id"] = K.project_id
    K["akey"] = np.where(K.dgroup == "EU/US", K.donor + " | " + K.title.map(norm_title), K.dgroup + " | " + K.project_id.astype(str))
    rec = K.drop_duplicates("record_id").sort_values(["startyear", "record_id"])
    act = rec.groupby("akey").agg(startyear=("startyear", "min"), comm=("comm", lambda s: s.clip(lower=0).sum(min_count=1)),
                                  sector_codes=("sector_codes", "first"), n_records=("record_id", "size"))
    act["project_id"] = np.arange(1, len(act) + 1) * 10 + 7   # stable integer ids, distinct from GODAD record ids
    for c in ["project_id", "startyear", "comm", "sector_codes", "n_records"]:
        K[c] = K.akey.map(act[c])
    return K


def assign_units(K, U):
    """K: rows with longitude/latitude; U: GeoDataFrame with 'unit' (and optionally 'PARENT'). Returns K with unit columns."""
    cols = [c for c in ["unit", "PARENT"] if c in U]
    g = gpd.GeoDataFrame(K, geometry=gpd.points_from_xy(K.longitude, K.latitude), crs=4326)
    g = gpd.sjoin(g, U[cols + ["geometry"]], predicate="within", how="left").drop(columns="index_right")
    miss = g.unit.isna() & g.longitude.notna()
    if miss.any():
        nn = gpd.sjoin_nearest(g[miss].drop(columns=cols).to_crs(32648), U.to_crs(32648)[cols + ["geometry"]],
                               max_distance=SNAP_KM * 1000, how="left")
        nn = nn[~nn.index.duplicated()]
        g.loc[miss, cols] = nn[cols].values
    return pd.DataFrame(g.drop(columns="geometry"))


def reclassify_geocodes(A, units, proc, move=("PROVINCE", "UNCLEAR"), drop=("NATIONAL",), how="pop"):
    """Spec 9-26: EU/US activities geocoded to a province-capital town or a province name, hand-classified from their text
    (kh_geocode_classified.csv). PROVINCE/UNCLEAR weight at the flagged district is spread over the province's districts by
    population (Phnom Penh: its four khans); NATIONAL weight is dropped; TOWN stays. A: rows with project_id, unit, w (other
    columns are carried). units: unit, prov, pop_2019, access, capital. Set KH_GEOCODE=original to skip."""
    import os
    if os.environ.get("KH_GEOCODE", "main") == "original":
        return A
    cl = pd.read_csv(proc / "kh_geocode_classified.csv")[["project_id", "class"]]
    fu = pd.read_csv(proc / "kh_geocode_flagged_units.csv").merge(cl, on="project_id")
    key = fu.set_index(["project_id", "unit"])["class"]
    cls = pd.Series([key.get((p, u)) for p, u in zip(A.project_id, A.unit)], index=A.index)
    out = [A[~cls.isin(list(move) + list(drop))]]
    cap = set(units[units.capital].unit); prov = units.set_index("unit").prov
    for _, r in A[cls.isin(move)].iterrows():
        pool = units[units.unit.isin(cap)] if r.unit in cap else units[units.prov == prov.get(r.unit)]
        sh = (pool.pop_2019 / pool.pop_2019.sum()).values if how == "pop" else \
             (pool.access == pool.access.min()).astype(float).values if how == "worst" else np.full(len(pool), 1 / len(pool))
        rows = pd.DataFrame([r] * len(pool)).reset_index(drop=True)
        rows["unit"] = pool.unit.values; rows["w"] = r.w * sh
        out.append(rows[rows.w > 0])
    return pd.concat(out, ignore_index=True)
