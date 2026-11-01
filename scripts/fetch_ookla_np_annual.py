"""Annual Ookla Q1 tiles for Nepal (mobile 2020-2024; fixed 2019-2025): keep the Nepal bbox, delete the global file."""
import os, requests
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "nepal" / "ookla"
OUT.mkdir(parents=True, exist_ok=True)
TMP = Path(os.environ["TEMP"]) / "tloda" / "ookla_tmp"
TMP.mkdir(parents=True, exist_ok=True)
BBOX = (80.0, 26.3, 88.3, 30.5)


def tile_xy_bounds(z=16):
    import math
    def lon2x(lon): return int((lon + 180) / 360 * 2**z)
    def lat2y(lat):
        r = math.radians(lat)
        return int((1 - math.log(math.tan(r) + 1 / math.cos(r)) / math.pi) / 2 * 2**z)
    return lon2x(BBOX[0]), lon2x(BBOX[2]), lat2y(BBOX[3]), lat2y(BBOX[1])


def quadkey_to_xy(qk):
    x = y = 0
    z = len(qk)
    for i in range(z):
        mask = 1 << (z - i - 1)
        d = int(qk[i])
        if d & 1: x |= mask
        if d & 2: y |= mask
    return x, y


x0, x1, y0, y1 = tile_xy_bounds()
jobs = [("mobile", y) for y in ["2020", "2021", "2022", "2023", "2024"]] + [("fixed", y) for y in ["2019", "2020", "2021", "2022", "2023", "2024", "2025"]]
for typ, y in jobs:
    out = OUT / f"ookla_{typ}_{y}Q1_NP.csv"
    if out.exists():
        continue
    key = f"parquet/performance/type={typ}/year={y}/quarter=1/{y}-01-01_performance_{typ}_tiles.parquet"
    f = TMP / Path(key).name
    try:
        if not f.exists():
            with requests.get("https://ookla-open-data.s3.amazonaws.com/" + key, stream=True, timeout=900) as r:
                r.raise_for_status()
                with open(f, "wb") as fh:
                    for chunk in r.iter_content(1 << 22):
                        fh.write(chunk)
        pf = pq.ParquetFile(f)
        keep = []
        for i in range(pf.num_row_groups):
            df = pf.read_row_group(i, columns=["quadkey", "avg_d_kbps", "avg_u_kbps", "avg_lat_ms", "tests", "devices"]).to_pandas()
            xy = df["quadkey"].map(quadkey_to_xy)
            df["tx"] = [a for a, b in xy]; df["ty"] = [b for a, b in xy]
            keep.append(df[(df.tx >= x0) & (df.tx <= x1) & (df.ty >= y0) & (df.ty <= y1)])
        res = pd.concat(keep)
        res.to_csv(out, index=False)
        print(typ, y, "NP tiles:", len(res), "tests:", res.tests.sum(), flush=True)
        pf.close(); del pf
    except Exception as e:
        print(typ, y, "FAILED", e, flush=True)
    try:
        f.unlink()
    except Exception:
        pass
