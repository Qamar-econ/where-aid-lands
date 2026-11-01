"""Fetch boundaries, AidData TL, WB projects, IATI (d-portal) activities+locations, HDX metadata."""
import json, os, requests, zipfile, io
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
H = {"User-Agent": "research/0.1"}


def get(u, **kw):
    r = requests.get(u, headers=H, timeout=180, **kw)
    r.raise_for_status()
    return r


def save(sub, name, content):
    d = RAW / sub
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_bytes(content)
    print("saved", sub, name, len(content))


# 1. geoBoundaries ADM1/ADM2
for lvl in ["ADM1", "ADM2"]:
    meta = get(f"https://www.geoboundaries.org/api/current/gbOpen/TLS/{lvl}/").json()
    save("geoboundaries", f"TLS_{lvl}.geojson", get(meta["gjDownloadURL"]).content)

# 2. HDX COD-AB resources list
cod = get("https://data.humdata.org/api/3/action/package_show?id=cod-ab-tls").json()["result"]
print("COD-AB resources:")
for r in cod["resources"]:
    print("  ", r["name"], r["url"])
for pid in ["cod-ps-tls", "relative-wealth-index"]:
    try:
        p = get(f"https://data.humdata.org/api/3/action/package_show?id={pid}").json()["result"]
        print(pid, "resources:")
        for r in p["resources"]:
            if pid != "relative-wealth-index" or "tls" in r["name"].lower() or "timor" in r["name"].lower():
                print("  ", r["name"], r["url"])
    except Exception as e:
        print(pid, "ERR", e)

# 3. AidData geocoded TL
lst = get("https://api.github.com/repos/AidData-WM/public_datasets/contents/geocoded").json()
for f in lst:
    if "timor" in f["name"].lower() or "worldbank" in f["name"].lower().replace("_", ""):
        print("aiddata file:", f["name"], f["size"])
for f in lst:
    if "timor" in f["name"].lower():
        save("aiddata_tl", f["name"], get(f["download_url"]).content)

# 4. World Bank projects (all TL)
wb = get("https://search.worldbank.org/api/v2/projects?format=json&countrycode_exact=TP&rows=500&fl=*").json()
save("worldbank", "wb_projects_TL.json", json.dumps(wb).encode())
print("WB projects:", len(wb["projects"]))

# 5. d-portal IATI activities + locations
acts = get("https://d-portal.org/q.json?from=act,country&country_code=TL&limit=-1").json()["rows"]
save("iati", "dportal_act_TL.json", json.dumps(acts).encode())
print("IATI activities:", len(acts))
locs = get("https://d-portal.org/q.json?from=act,location,country&country_code=TL&limit=-1").json()["rows"]
save("iati", "dportal_loc_TL.json", json.dumps(locs).encode())
print("IATI activity-locations:", len(locs))
