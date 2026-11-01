"""Cambodia analysis units: IPUMS-International spatially harmonized districts (GEO2_KH, 1998-2019), which match the
census microdata; fallback = geoBoundaries ADM2 (2019-era, 197 districts). Output: data/processed/cambodia/kh_units.gpkg
"""
import io, re, zipfile, requests
import geopandas as gpd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data/raw/cambodia", ROOT / "data/processed/cambodia"
RAW.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
H = {"User-Agent": "Mozilla/5.0 (research)"}
src = None
page = requests.get("https://international.ipums.org/international/gis_harmonized_2nd.shtml", headers=H, timeout=60)
links = [l for l in re.findall(r'href="([^"]+)"', page.text) if "kh" in l.lower() and l.lower().endswith(".zip")]
print("IPUMS page status", page.status_code, "| Cambodia links:", links[:5])
for l in links:
    url = l if l.startswith("http") else ("https://international.ipums.org" + l if l.startswith("/") else "https://international.ipums.org/international/" + l)
    r = requests.get(url, headers=H, timeout=300)
    if r.status_code == 200 and r.content[:2] == b"PK":
        d = RAW / "ipums_geo2_kh"; zipfile.ZipFile(io.BytesIO(r.content)).extractall(d)
        shp = next(d.rglob("*.shp")); src = gpd.read_file(shp).to_crs(4326); print("IPUMS GEO2_KH:", url, len(src), src.columns.tolist())
        break
if src is None:
    meta = requests.get("https://www.geoboundaries.org/api/current/gbOpen/KHM/ADM2/", timeout=120).json()
    src = gpd.read_file(io.BytesIO(requests.get(meta["gjDownloadURL"], timeout=300).content)).to_crs(4326)
    print("fallback geoBoundaries ADM2:", len(src))
src.to_file(OUT / "kh_units.gpkg")
