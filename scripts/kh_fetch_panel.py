"""Annual panels for the infrastructure-to-development check (kh_s30): harmonised night lights (Li et al., figshare 9828827)
and WorldPop 1 km UN-adjusted population, 2003-2021 / 2003-2020, clipped to Cambodia. Global night-light files are
downloaded to a temporary folder, clipped and deleted.
"""
import os, requests, rasterio
from rasterio.windows import from_bounds
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/cambodia"
TMP = Path(os.environ["TEMP"]) / "tloda" / "kh_tmp"; TMP.mkdir(parents=True, exist_ok=True)
BBOX = (102.3, 10.3, 107.7, 14.8)

files = {f["name"]: f["download_url"] for f in requests.get("https://api.figshare.com/v2/articles/9828827", timeout=60).json()["files"]}
for y in range(2003, 2022):
    out = RAW / "ntl" / f"ntl_{y}_KHM.tif"
    if out.exists():
        continue
    name = next((n for n in files if n.startswith(f"Harmonized_DN_NTL_{y}_")), None)
    if not name:
        print("NTL year missing", y); continue
    tmp = TMP / name
    with requests.get(files[name], stream=True, timeout=1800) as r, open(tmp, "wb") as fh:
        for ch in r.iter_content(1 << 22):
            fh.write(ch)
    with rasterio.open(tmp) as s:
        w = from_bounds(*BBOX, s.transform); a = s.read(1, window=w); prof = s.profile
        prof.update(height=a.shape[0], width=a.shape[1], transform=s.window_transform(w))
    with rasterio.open(out, "w", **prof) as d:
        d.write(a, 1)
    tmp.unlink(); print("ok NTL", y, flush=True)

for y in range(2003, 2021):
    out = RAW / "worldpop" / f"khm_ppp_{y}_1km.tif"
    if out.exists():
        continue
    url = f"https://data.worldpop.org/GIS/Population/Global_2000_2020_1km_UNadj/{y}/KHM/khm_ppp_{y}_1km_Aggregated_UNadj.tif"
    r = requests.get(url, timeout=600)
    if r.status_code == 200:
        out.write_bytes(r.content); print("ok pop", y, flush=True)
    else:
        print("FAILED pop", y, r.status_code)
