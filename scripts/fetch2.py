"""Fetch COD boundaries/population, RWI, AidData WB geocoded; unzip archives."""
import requests, zipfile
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
H = {"User-Agent": "research/0.1"}

FILES = {
    "cod_ab": [("tls_admin_boundaries.shp.zip", "https://data.humdata.org/dataset/ee4a485c-9180-43db-9672-55e18dac5f96/resource/a3d43a39-2ddc-4f66-8e53-6fcbec8bd665/download/tls_admin_boundaries.shp.zip")],
    "cod_ps": [("tls_admpop_2022.xlsx", "https://data.humdata.org/dataset/0a27693a-c453-4320-8d89-6b75025e1ac2/resource/c4b86157-f600-457b-acb7-db9ebdff58fa/download/tls_admpop_2022.xlsx"),
               ("TLS_AdminBoundaries_TabularData.xlsx", "https://data.humdata.org/dataset/0a27693a-c453-4320-8d89-6b75025e1ac2/resource/dd647746-344d-409c-8d40-32c885de0d3f/download/tls_adminboundaries_tabulardata.xlsx")],
    "rwi": [("tls_relative_wealth_index.csv", "https://data.humdata.org/dataset/76f2a2ea-ba50-40f5-b79c-db95d668b843/resource/f642c669-51db-4ed3-befe-703cc913122e/download/tls_relative_wealth_index.csv")],
    "aiddata_wb": [("WorldBank_GeocodedResearchRelease_Level1_v1.4.2.zip", "https://raw.githubusercontent.com/AidData-WM/public_datasets/master/geocoded/WorldBank_GeocodedResearchRelease_Level1_v1.4.2.zip")],
}

for sub, items in FILES.items():
    d = RAW / sub
    d.mkdir(parents=True, exist_ok=True)
    for name, url in items:
        r = requests.get(url, headers=H, timeout=300)
        r.raise_for_status()
        (d / name).write_bytes(r.content)
        print("saved", sub, name, len(r.content))

for z in list(RAW.rglob("*.zip")):
    out = z.with_suffix("")
    if not out.exists():
        with zipfile.ZipFile(z) as zf:
            zf.extractall(out)
        print("unzipped", z.name)
