"""Nepal core data: AidData Nepal AIMS + earthquake geocoded, OCHA COD-AB / COD-PS, HDX census 2011 VDC dataset."""
import json, zipfile, requests
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "nepal"
H = {"User-Agent": "research/0.1"}


def save(sub, name, url):
    d = RAW / sub; d.mkdir(parents=True, exist_ok=True)
    f = d / name
    if not f.exists():
        r = requests.get(url, headers=H, timeout=900); r.raise_for_status(); f.write_bytes(r.content)
        print("saved", sub, name, len(r.content), flush=True)
    if f.suffix == ".zip" and not f.with_suffix("").exists():
        with zipfile.ZipFile(f) as z:
            z.extractall(f.with_suffix(""))
    return f


base = "https://raw.githubusercontent.com/AidData-WM/public_datasets/master/geocoded/"
save("aiddata", "NepalAIMS_GeocodedResearchRelease_Level1_v1.4.1.zip", base + "NepalAIMS_GeocodedResearchRelease_Level1_v1.4.1.zip")
save("aiddata", "NepalEarthquake_GeocodedResearchRelease_Level1_v1.0.zip", base + "NepalEarthquake_GeocodedResearchRelease_Level1_v1.0.zip")
for pid, keep in [("cod-ab-npl", ["shp.zip", ".xlsx"]), ("cod-ps-npl", ["admpop_2023.xlsx", "admgz.xlsx"]),
                  ("national-population-and-household-census-2011-vdc-municipality-level", [""])]:
    p = requests.get(f"https://data.humdata.org/api/3/action/package_show?id={pid}", headers=H, timeout=60).json()["result"]
    for r in p["resources"]:
        if any(k in r["name"] for k in keep):
            save(pid, r["name"].replace("/", "_"), r["url"])
print("done")
