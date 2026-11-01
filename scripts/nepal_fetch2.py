"""Nepal large inputs: OSM PBF, Copernicus DEM tiles, Ookla mobile (clip), IATI d-portal JSON, World Bank API,
Meta RWI, WorldPop 1 km 2010/2020."""
import json, os, requests, zipfile
import pandas as pd, pyarrow.parquet as pq
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "nepal"
H = {"User-Agent": "research/0.1"}
BBOX = (80.0, 26.3, 88.3, 30.5)


def dl(sub, name, url, timeout=1800):
    d = RAW / sub; d.mkdir(parents=True, exist_ok=True); f = d / name
    if f.exists() and f.stat().st_size > 0:
        return f
    try:
        with requests.get(url, headers=H, stream=True, timeout=timeout) as r:
            if r.status_code != 200:
                print("FAIL", r.status_code, name, flush=True); return None
            with open(f, "wb") as fh:
                for ch in r.iter_content(1 << 22):
                    fh.write(ch)
        print("saved", sub, name, f.stat().st_size, flush=True); return f
    except Exception as e:
        print("ERR", name, e, flush=True)


# IATI (d-portal JSON) and World Bank
d = RAW / "iati"; d.mkdir(parents=True, exist_ok=True)
for frm, fn in [("act,country", "act"), ("act,location,country", "loc"), ("act,sector,country", "sector")]:
    f = d / f"dportal_{fn}_NP.json"
    if not f.exists():
        rows = requests.get(f"https://d-portal.org/q.json?from={frm}&country_code=NP&limit=-1", headers=H, timeout=1800).json()["rows"]
        f.write_text(json.dumps(rows), encoding="utf-8"); print("IATI", fn, len(rows), flush=True)
f = RAW / "worldbank" / "wb_projects_NP.json"
if not f.exists():
    f.parent.mkdir(parents=True, exist_ok=True)
    wb = requests.get("https://search.worldbank.org/api/v2/projects?format=json&countrycode_exact=NP&rows=500&fl=*", headers=H, timeout=600).json()
    f.write_text(json.dumps(wb), encoding="utf-8"); print("WB", len(wb["projects"]), flush=True)

# RWI, WorldPop
p = requests.get("https://data.humdata.org/api/3/action/package_show?id=relative-wealth-index", headers=H, timeout=60).json()["result"]
for r in p["resources"]:
    if "nepal" in r["name"].lower() or "npl" in r["name"].lower():
        dl("rwi", "npl_relative_wealth_index.csv", r["url"])
for y in [2010, 2020]:
    dl("worldpop", f"npl_ppp_{y}_1km_UNadj.tif", f"https://data.worldpop.org/GIS/Population/Global_2000_2020_1km_UNadj/{y}/NPL/npl_ppp_{y}_1km_Aggregated_UNadj.tif")

# DEM tiles (Copernicus GLO-90): N26..N30 x E080..E088
for lat in range(26, 31):
    for lon in range(80, 89):
        n = f"Copernicus_DSM_COG_30_N{lat:02d}_00_E{lon:03d}_00_DEM"
        dl("dem", n + ".tif", f"https://copernicus-dem-90m.s3.amazonaws.com/{n}/{n}.tif")

# OSM
dl("osm", "nepal-latest.osm.pbf", "https://download.geofabrik.de/asia/nepal-latest.osm.pbf", timeout=3600)


# Ookla mobile 2019Q1 / 2025Q1 clipped to Nepal
def qk_xy(qk):
    x = y = 0; z = len(qk)
    for i in range(z):
        m = 1 << (z - i - 1); c = int(qk[i])
        if c & 1: x |= m
        if c & 2: y |= m
    return x, y


import math
def lon2x(lon): return int((lon + 180) / 360 * 2 ** 16)
def lat2y(lat):
    r = math.radians(lat); return int((1 - math.log(math.tan(r) + 1 / math.cos(r)) / math.pi) / 2 * 2 ** 16)
x0, x1, y0, y1 = lon2x(BBOX[0]), lon2x(BBOX[2]), lat2y(BBOX[3]), lat2y(BBOX[1])
TMP = Path(os.environ["TEMP"]) / "tloda" / "ookla_tmp"; TMP.mkdir(parents=True, exist_ok=True)
for yy in ["2019", "2025"]:
    out = RAW / "ookla" / f"ookla_mobile_{yy}Q1_NP.csv"
    if out.exists():
        continue
    key = f"parquet/performance/type=mobile/year={yy}/quarter=1/{yy}-01-01_performance_mobile_tiles.parquet"
    g = TMP / Path(key).name
    if not g.exists():
        with requests.get("https://ookla-open-data.s3.amazonaws.com/" + key, stream=True, timeout=1800) as r:
            with open(g, "wb") as fh:
                for ch in r.iter_content(1 << 22):
                    fh.write(ch)
    pf = pq.ParquetFile(g); keep = []
    for i in range(pf.num_row_groups):
        df = pf.read_row_group(i, columns=["quadkey", "avg_d_kbps", "avg_lat_ms", "tests", "devices"]).to_pandas()
        xy = df.quadkey.map(qk_xy); df["tx"] = [a for a, b in xy]; df["ty"] = [b for a, b in xy]
        keep.append(df[(df.tx >= x0) & (df.tx <= x1) & (df.ty >= y0) & (df.ty <= y1)])
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(keep).to_csv(out, index=False); pf.close(); del pf
    try:
        g.unlink()
    except PermissionError:
        pass
    print("ookla", yy, "done", flush=True)
print("ALL DONE", flush=True)
