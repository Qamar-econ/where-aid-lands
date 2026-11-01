"""Nepal gazetteer v2 for village / palika / district geocoding (AidData-UCDP style precision).

Entries:  palika (COD admin3, 753 local levels; with 'Rural Municipality'/'Municipality'/'Gaunpalika'/'Nagarpalika' variants)
          village / town (GeoNames NP feature class P, >= 5 characters) -> palika and district by point-in-polygon
Each entry carries its district key so that the co-mention rule can be applied (a village or an ambiguous palika name
counts only if its district is also named in the same text).
Output: data/interim/np2_gazetteer.csv
"""
import io, re, zipfile, requests
import pandas as pd, geopandas as gpd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/nepal"
gz = RAW / "geonames" / "NP.txt"
if not gz.exists():
    gz.parent.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(io.BytesIO(requests.get("https://download.geonames.org/export/dump/NP.zip", timeout=300).content))
    gz.write_bytes(z.read("NP.txt"))
cols = ["geonameid", "name", "asciiname", "alternatenames", "lat", "lon", "fclass", "fcode", "cc", "cc2", "a1", "a2", "a3", "a4", "pop", "elev", "dem", "tz", "mod"]
G = pd.read_csv(gz, sep="\t", names=cols, dtype=str, quoting=3)
G = G[G.fclass == "P"].copy()
G["lat"], G["lon"] = G.lat.astype(float), G.lon.astype(float)

P3 = gpd.read_file(RAW / "cod-ab-npl/npl_admin_boundaries.shp/npl_admin3.shp")[["adm3_pcode", "adm3_name", "adm2_name", "geometry"]]
D = gpd.read_file(ROOT / "data/processed/nepal/districts.gpkg")[["k", "district", "geometry"]]
cent = P3.copy(); cent["geometry"] = P3.representative_point()
P3 = P3.merge(gpd.sjoin(cent, D, predicate="within")[["adm3_pcode", "k"]], on="adm3_pcode")
STOP = {"nepal", "school", "health", "water", "bazar", "bazaar", "tole", "chowk", "pani", "danda", "khola", "gaun", "village", "mandir", "temple",
        "airport", "camp", "hospital", "office", "market", "central", "western", "eastern", "north", "south", "upper", "lower", "middle", "valley"}


def clean(n):
    return re.sub(r"\s+", " ", re.sub(r"[^A-Za-z \-]", " ", str(n))).strip()


rows = []
# palikas
for r in P3.itertuples():
    base = clean(re.sub(r"\b(Rural Municipality|Sub-Metropolitan City|Metropolitan City|Municipality|Gaunpalika|Nagarpalika|RM|Mun)\b", "", r.adm3_name))
    if len(base) < 4 or base.lower() in STOP:
        continue
    for v in {base, base + " Rural Municipality", base + " Municipality", base + " Gaunpalika", base + " Nagarpalika"}:
        rows.append({"name": v, "level": "palika", "pcode": r.adm3_pcode, "k": r.k, "lat": None, "lon": None})
# villages / towns from GeoNames (point -> palika)
gg = gpd.GeoDataFrame(G, geometry=gpd.points_from_xy(G.lon, G.lat), crs=4326)
gg = gpd.sjoin(gg, P3[["adm3_pcode", "k", "geometry"]], predicate="within")
for r in gg.itertuples():
    names = {clean(r.name), clean(r.asciiname)}
    for n in names:
        if len(n) >= 5 and n.lower() not in STOP and not n.lower().endswith((" school", " khola", " danda")):
            rows.append({"name": n, "level": "village", "pcode": r.adm3_pcode, "k": r.k, "lat": r.lat, "lon": r.lon})
Z = pd.DataFrame(rows).drop_duplicates(["name", "level", "pcode"])
# ambiguity: how many districts share a name (within level)
amb = Z.groupby(["name"]).k.nunique()
Z["n_districts"] = Z.name.map(amb)
Z.to_csv(ROOT / "data/interim/np2_gazetteer.csv", index=False, encoding="utf-8-sig")
print("gazetteer entries:", len(Z), Z.level.value_counts().to_dict(), "| names unique to one district:", (Z.n_districts == 1).mean().round(2))
