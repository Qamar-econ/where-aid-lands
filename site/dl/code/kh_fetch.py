"""Cambodia raw inputs: WorldPop 1 km (2015, 2019, 2020), harmonized DMSP-VIIRS night lights (Li et al.; 2008, 2015, 2019),
Copernicus GLO-90 DEM tiles, OSM PBF (Geofabrik), GeoNames KH, Ookla 2019 Q1-Q4 + 2020 Q1 tiles (mobile, fixed; Cambodia bbox).
"""
import io, os, zipfile, requests
import pandas as pd, pyarrow.parquet as pq, rasterio
from rasterio.windows import from_bounds
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/cambodia"
TMP = Path(os.environ["TEMP"]) / "tloda" / "kh_tmp"; TMP.mkdir(parents=True, exist_ok=True)
BBOX = (102.3, 10.3, 107.7, 14.8)


def dl(sub, name, url, timeout=1800):
    p = RAW / sub / name; p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        return p
    with requests.get(url, stream=True, timeout=timeout) as r:
        if r.status_code != 200:
            print("FAILED", url, r.status_code, flush=True); return None
        with open(p, "wb") as fh:
            for ch in r.iter_content(1 << 22):
                fh.write(ch)
    print("ok", name, flush=True)
    return p


for y in [2008, 2015, 2019, 2020]:
    dl("worldpop", f"khm_ppp_{y}_1km.tif", f"https://data.worldpop.org/GIS/Population/Global_2000_2020_1km_UNadj/{y}/KHM/khm_ppp_{y}_1km_Aggregated_UNadj.tif")
for lat in range(10, 15):
    for lon in range(102, 108):
        n = f"Copernicus_DSM_COG_30_N{lat:02d}_00_E{lon:03d}_00_DEM"
        dl("dem", n + ".tif", f"https://copernicus-dem-90m.s3.amazonaws.com/{n}/{n}.tif")
dl("osm", "cambodia-latest.osm.pbf", "https://download.geofabrik.de/asia/cambodia-latest.osm.pbf")
dl("osm", "cambodia-170101.osm.pbf", "https://download.geofabrik.de/asia/cambodia-170101.osm.pbf")   # main network (spec 9-6)
# Ministry of Health facility list (2010, HDX), for the hospital check (kh_s28, spec 9-17)
z = dl("moh_health", "health_facility.zip", "https://data.humdata.org/dataset/1803994d-6218-4b98-ac3a-30c7f85c6dbc/resource/f30b0f4b-1c40-45f3-986d-2820375ea8dd/download/health_facility.zip")
if z and not (RAW / "moh_health/Health").exists():
    zipfile.ZipFile(z).extractall(RAW / "moh_health")
# Malaria Atlas Project accessibility maps clipped to Cambodia, for external validation (kh_s29, spec 9-18)
for cov in ["Accessibility__201501_Global_Travel_Time_to_Cities", "Accessibility__202001_Global_Motorized_Travel_Time_to_Healthcare"]:
    dl("map_access", cov + ".tif", "https://data.malariaatlas.org/geoserver/Accessibility/wcs?service=WCS&version=2.0.1&request=GetCoverage"
       f"&coverageId={cov}&subset=Lat(10.3,14.8)&subset=Long(102.3,107.7)&format=image/geotiff")
g = RAW / "geonames" / "KH.txt"
if not g.exists():
    g.parent.mkdir(parents=True, exist_ok=True)
    g.write_bytes(zipfile.ZipFile(io.BytesIO(requests.get("https://download.geonames.org/export/dump/KH.zip", timeout=300).content)).read("KH.txt"))
# night lights
files = {f["name"]: f["download_url"] for f in requests.get("https://api.figshare.com/v2/articles/9828827", timeout=60).json()["files"]}
for y in [2008, 2015, 2019]:
    out = RAW / "ntl" / f"ntl_{y}_KHM.tif"
    if out.exists():
        continue
    name = next((n for n in files if n.startswith(f"Harmonized_DN_NTL_{y}_")), None)
    if not name:
        print("NTL year missing", y); continue
    tmp = TMP / name
    if not tmp.exists():
        with requests.get(files[name], stream=True, timeout=1800) as r:
            with open(tmp, "wb") as fh:
                for ch in r.iter_content(1 << 22):
                    fh.write(ch)
    with rasterio.open(tmp) as s:
        w = from_bounds(*BBOX, s.transform); a = s.read(1, window=w); prof = s.profile
        prof.update(height=a.shape[0], width=a.shape[1], transform=s.window_transform(w))
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", **prof) as d:
        d.write(a, 1)
    tmp.unlink(); print("ok NTL", y, flush=True)


# Ookla
def qk_xy(qk, z=16):
    x = y = 0
    for i in range(len(qk)):
        m = 1 << (len(qk) - i - 1); d = int(qk[i])
        if d & 1: x |= m
        if d & 2: y |= m
    return x, y


import math
n16 = 2 ** 16
x0, x1 = int((BBOX[0] + 180) / 360 * n16), int((BBOX[2] + 180) / 360 * n16)
lat2y = lambda lat: int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n16)
y0, y1 = lat2y(BBOX[3]), lat2y(BBOX[1])
for typ in ["mobile", "fixed"]:
    for y, q in [(2019, 1), (2019, 2), (2019, 3), (2019, 4), (2020, 1)]:
        out = RAW / "ookla" / f"ookla_{typ}_{y}Q{q}_KH.csv"
        if out.exists():
            continue
        m = {1: "01", 2: "04", 3: "07", 4: "10"}[q]
        key = f"parquet/performance/type={typ}/year={y}/quarter={q}/{y}-{m}-01_performance_{typ}_tiles.parquet"
        f = TMP / Path(key).name
        try:
            if not f.exists():
                with requests.get("https://ookla-open-data.s3.amazonaws.com/" + key, stream=True, timeout=1800) as r:
                    r.raise_for_status()
                    with open(f, "wb") as fh:
                        for ch in r.iter_content(1 << 22):
                            fh.write(ch)
            pf = pq.ParquetFile(f); keep = []
            for i in range(pf.num_row_groups):
                d = pf.read_row_group(i, columns=["quadkey", "avg_d_kbps", "avg_u_kbps", "avg_lat_ms", "tests", "devices"]).to_pandas()
                xy = d.quadkey.map(qk_xy); d["tx"] = [a for a, b in xy]; d["ty"] = [b for a, b in xy]
                keep.append(d[d.tx.between(x0, x1) & d.ty.between(y0, y1)])
            pf.close(); out.parent.mkdir(parents=True, exist_ok=True)
            pd.concat(keep).to_csv(out, index=False); print("ok Ookla", typ, y, q, flush=True)
        except Exception as e:
            print("Ookla failed", typ, y, q, e, flush=True)
        try:
            f.unlink()
        except Exception:
            pass
print("done")
