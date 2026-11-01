"""Harmonized DMSP-VIIRS night lights (Li et al., figshare 9828827): clip selected years to Timor-Leste and Nepal."""
import os, requests, rasterio
from rasterio.windows import from_bounds
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TMP = Path(os.environ["TEMP"]) / "tloda" / "ntl_tmp"; TMP.mkdir(parents=True, exist_ok=True)
BOXES = {"TLS": (124.0, -9.6, 127.5, -8.1), "NPL": (80.0, 26.3, 88.3, 30.5)}
files = {f["name"]: f["download_url"] for f in requests.get("https://api.figshare.com/v2/articles/9828827", timeout=60).json()["files"]}
for y in [2000, 2005, 2010, 2013, 2015, 2020]:
    name = next(n for n in files if n.startswith(f"Harmonized_DN_NTL_{y}_"))
    outs = {c: ROOT / "data" / "raw" / ("ntl" if c == "TLS" else "nepal/ntl") / f"ntl_{y}_{c}.tif" for c in BOXES}
    if all(o.exists() for o in outs.values()):
        continue
    g = TMP / name
    if not g.exists():
        with requests.get(files[name], stream=True, timeout=900) as r:
            r.raise_for_status()
            with open(g, "wb") as fh:
                for ch in r.iter_content(1 << 22):
                    fh.write(ch)
    with rasterio.open(g) as src:
        for c, b in BOXES.items():
            w = from_bounds(*b, transform=src.transform)
            arr = src.read(1, window=w)
            meta = src.meta.copy(); meta.update(height=arr.shape[0], width=arr.shape[1], transform=src.window_transform(w), compress="deflate")
            outs[c].parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(outs[c], "w", **meta) as dst:
                dst.write(arr, 1)
    g.unlink()
    print("clipped", y, flush=True)
