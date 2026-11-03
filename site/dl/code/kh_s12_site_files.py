"""Lay out the archive's downloadable files as individually served files (the page zips them in the browser).
site/dl/data/*.csv|geojson, site/dl/tables/*.csv, site/dl/code/*.py (served as text/plain), 
Writes site/data/manifest.json (bundle -> file list) and prints the Artifact `files` map.
"""
import json, shutil
import geopandas as gpd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"; DL = SITE / "dl"
for old in DL.glob("*"):
    if old.is_file():
        old.unlink()
proc, raw, tab = ROOT / "data/processed/cambodia", ROOT / "data/raw/cambodia", ROOT / "outputs/tables"
bundles = {"data": [], "tables": [], "code": []}
def put(src, sub, name=None):
    dst = DL / sub / (name or src.name); dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, dst)
    bundles[sub].append(f"dl/{sub}/{dst.name}")
for f in ["kh_projects.csv", "kh_assign.csv", "kh_covariates.csv", "kh_access.csv", "kh_rwi.csv", "kh_analysis.csv", "kh_published.csv", "kh_geocode_classified.csv", "kh_geocode_flagged_units.csv", "kh_weiss_district.csv"]:
    put(proc / f, "data")
put(raw / "godad_khm.csv", "data")
g = DL / "data/godad_khm.csv"  # the GODAD source text carries a few U+FFFD replacement characters; mark them as '?'
g.write_text(g.read_text(encoding="utf-8", errors="replace").replace(chr(0xFFFD), "?"), encoding="utf-8")
put(tab / "kh_access_pca.csv", "data")
gpd.read_file(proc / "kh_units.gpkg").to_file(DL / "data/kh_units.geojson", driver="GeoJSON"); bundles["data"].append("dl/data/kh_units.geojson")
for f in sorted(tab.glob("kh_*.csv")):
    put(f, "tables")
for f in sorted((ROOT / "scripts").glob("kh_*.py")):
    put(f, "code")
put(ROOT / "scripts/README_pipeline.md", "code")
# main-analysis figures (kh_figures.py), English and Korean, shown on the page
for f in sorted((ROOT / "outputs/figures/cambodia").glob("F[1-9]_*.png")):
    shutil.copyfile(f, SITE / "img" / f.name)
# working documents (Korean) stay in the repository and are not offered on the public site
(SITE / "data/manifest.json").write_text(json.dumps(bundles, ensure_ascii=False, indent=1), encoding="utf-8")
files = {"data/districts.geojson": "research/site/data/districts.geojson", "data/projects.json": "research/site/data/projects.json",
         "data/results.json": "research/site/data/results.json", "data/manifest.json": "research/site/data/manifest.json"}
for sub, lst in bundles.items():
    for p in lst:
        src = "research/site/" + p
        files[p] = {"from": src, "contentType": "text/plain"} if p.endswith(".py") else src
print(json.dumps(files, ensure_ascii=False))
print({k: len(v) for k, v in bundles.items()}, "| total MB", round(sum(f.stat().st_size for f in DL.rglob("*") if f.is_file()) / 1e6, 1))
