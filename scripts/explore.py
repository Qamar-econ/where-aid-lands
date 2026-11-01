import json
import pandas as pd, geopandas as gpd
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 60)

a2 = gpd.read_file(RAW / "cod_ab/tls_admin_boundaries.shp/tls_admin2.shp")
print("ADMIN2", a2.shape, list(a2.columns))
print(a2.drop(columns="geometry").head(3).T)
a1 = gpd.read_file(RAW / "cod_ab/tls_admin_boundaries.shp/tls_admin1.shp")
print("ADMIN1", a1[[c for c in a1.columns if "name" in c.lower() or "pcode" in c.lower()]].head(20))

x = pd.ExcelFile(RAW / "cod_ps/tls_admpop_2022.xlsx")
print("POP sheets", x.sheet_names)
for s in x.sheet_names:
    df = x.parse(s)
    print(s, df.shape); print(df.head(3).T.head(30))

ad = RAW / "aiddata_tl/TimorLesteAIMS_GeocodedResearchRelease_Level1_v1.4.1/TimorLesteAIMS_GeocodedResearchRelease_Level1_v1.4.1/data"
for f in ["projects.csv", "locations.csv", "transactions.csv", "projects_ancillary.csv"]:
    df = pd.read_csv(ad / f, encoding="latin1")
    print("AIDDATA", f, df.shape); print(df.head(2).T)

locs = json.loads((RAW / "iati/dportal_loc_TL.json").read_text(encoding="utf-8"))
print("IATI loc keys", list(locs[0].keys()))
print(json.dumps(locs[0], ensure_ascii=False)[:1500])
acts = json.loads((RAW / "iati/dportal_act_TL.json").read_text(encoding="utf-8"))
print("IATI act keys", list(acts[0].keys()))
print(pd.Series([a["reporting"] for a in acts]).value_counts().head(40))

wb = json.loads((RAW / "worldbank/wb_projects_TL.json").read_text(encoding="utf-8"))["projects"]
p = next(iter(wb.values()))
print("WB keys", list(p.keys()))
print("WB locations sample", json.dumps(p.get("locations"), ensure_ascii=False)[:800])
print(sum(1 for v in wb.values() if v.get("locations")), "WB projects with locations")

rwi = pd.read_csv(RAW / "rwi/tls_relative_wealth_index.csv")
print("RWI", rwi.shape, rwi.head())
