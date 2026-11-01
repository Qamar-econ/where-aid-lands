"""Country scan: how rich and how spatially even is Ookla open data (mobile + fixed, 2023 Q1) in candidate aid-recipient countries?
Metrics per country: tests per 1,000 people; share of ADM2 units with >= 1 / >= 30 / >= 100 tests in the quarter; share of tests in the
top-5% ADM2 units (concentration). ADM2 boundaries: geoBoundaries gbOpen (simplified).
Output: outputs/tables/country_scan_ookla.csv
"""
import io, math, os, requests
import numpy as np, pandas as pd, geopandas as gpd, pyarrow.parquet as pq
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TMP = Path(os.environ["TEMP"]) / "tloda" / "ookla_tmp"; TMP.mkdir(parents=True, exist_ok=True)
POP = {"KEN": 55, "PHL": 115, "COL": 52, "GHA": 34, "RWA": 14, "BGD": 172, "UGA": 48, "KHM": 17, "PER": 34, "HND": 10.6, "NPL": 30.5,
       "TLS": 1.4, "VNM": 100, "BOL": 12.4, "GTM": 18, "SEN": 18, "ZMB": 20.5, "TZA": 67, "MWI": 20.9, "LKA": 22, "MNG": 3.4, "LAO": 7.6}
G = {}
for iso in POP:
    try:
        meta = requests.get(f"https://www.geoboundaries.org/api/current/gbOpen/{iso}/ADM2/", timeout=120).json()
        G[iso] = gpd.read_file(io.BytesIO(requests.get(meta["simplifiedGeometryGeoJSON"], timeout=300).content))[["geometry"]].assign(iso=iso)
        print(iso, "ADM2 units", len(G[iso]), flush=True)
    except Exception as e:
        print(iso, "boundary failed", e, flush=True)
B = pd.concat(G.values(), ignore_index=True).reset_index().rename(columns={"index": "uid"})
B = gpd.GeoDataFrame(B, geometry="geometry", crs=4326)
bbox = B.total_bounds


def qk_xy(qk):
    x = y = 0; z = len(qk)
    for i in range(z):
        m = 1 << (z - i - 1); d = int(qk[i])
        if d & 1: x |= m
        if d & 2: y |= m
    n = 2 ** z
    return (x + .5) / n * 360 - 180, math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + .5) / n))))


rows = []
for typ in ["mobile", "fixed"]:
    key = f"parquet/performance/type={typ}/year=2023/quarter=1/2023-01-01_performance_{typ}_tiles.parquet"
    f = TMP / Path(key).name
    if not f.exists():
        with requests.get("https://ookla-open-data.s3.amazonaws.com/" + key, stream=True, timeout=900) as r:
            r.raise_for_status()
            with open(f, "wb") as fh:
                for ch in r.iter_content(1 << 22):
                    fh.write(ch)
    pf = pq.ParquetFile(f)
    parts = []
    for i in range(pf.num_row_groups):
        d = pf.read_row_group(i, columns=["quadkey", "tests", "avg_d_kbps"]).to_pandas()
        ll = np.array([qk_xy(q) for q in d.quadkey])
        d["lon"], d["lat"] = ll[:, 0], ll[:, 1]
        parts.append(d[(d.lon.between(bbox[0], bbox[2])) & (d.lat.between(bbox[1], bbox[3]))])
    pf.close()
    T = pd.concat(parts)
    T = gpd.sjoin(gpd.GeoDataFrame(T, geometry=gpd.points_from_xy(T.lon, T.lat), crs=4326), B, predicate="within")
    agg = T.groupby("uid").agg(tests=("tests", "sum"))
    U = B[["uid", "iso"]].merge(agg, on="uid", how="left").fillna({"tests": 0})
    for iso, d in U.groupby("iso"):
        s = d.tests.sort_values(ascending=False)
        rows.append({"iso": iso, "type": typ, "adm2_units": len(d), "tests": int(s.sum()), "tests_per_1k": round(s.sum() / (POP[iso] * 1000), 2),
                     "share_units_ge1": round((s >= 1).mean(), 2), "share_units_ge30": round((s >= 30).mean(), 2), "share_units_ge100": round((s >= 100).mean(), 2),
                     "top5pct_units_share_of_tests": round(s.head(max(1, int(len(s) * 0.05))).sum() / max(1, s.sum()), 2)})
    try:
        f.unlink()
    except Exception:
        pass
R = pd.DataFrame(rows)
R.to_csv(ROOT / "outputs/tables/country_scan_ookla.csv", index=False)
print(R.pivot_table(index="iso", columns="type", values=["tests_per_1k", "share_units_ge30", "top5pct_units_share_of_tests"]).round(2).to_string())
