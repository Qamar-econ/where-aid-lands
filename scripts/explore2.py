import json
import pandas as pd
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 80)
ad = RAW / "aiddata_tl/TimorLesteAIMS_GeocodedResearchRelease_Level1_v1.4.1/TimorLesteAIMS_GeocodedResearchRelease_Level1_v1.4.1/data"
P = pd.read_csv(ad / "projects.csv", encoding="latin1")
L = pd.read_csv(ad / "locations.csv", encoding="latin1")
A = pd.read_csv(ad / "projects_ancillary.csv", encoding="latin1")
print(L.groupby(["precision_code", "location_type_code"]).size())
print(P.is_geocoded.value_counts(), P.transactions_start_year.describe())
print(P.ad_sector_names.str.split("|").explode().value_counts().head(40))
print(P.donors.value_counts().head(40))
print(A["Implementation Level"].value_counts(dropna=False))
print("ancillary district non-empty", A["District"].notna().sum(), "subdistrict", A["SubDistrict"].notna().sum())
print(A[A["SubDistrict"].notna()][["District", "SubDistrict"]].head(10))
print("start year hist", pd.to_datetime(P.start_actual_isodate, errors="coerce").dt.year.value_counts().sort_index().to_dict())

acts = pd.DataFrame(json.loads((RAW / "iati/dportal_act_TL.json").read_text(encoding="utf-8")))
locs = pd.DataFrame(json.loads((RAW / "iati/dportal_loc_TL.json").read_text(encoding="utf-8")))
acts["y0"] = pd.to_datetime(acts.day_start, unit="D", errors="coerce").dt.year
print("IATI start years", acts.y0.value_counts().sort_index().to_dict())
print("IATI precision", locs.location_precision.value_counts(dropna=False).to_dict())
locs["xy"] = locs.location_longitude.round(3).astype(str) + "," + locs.location_latitude.round(3).astype(str)
print("most repeated coords:\n", locs.groupby("xy").agg(n=("aid", "nunique"), name=("location_name", "first"), rep=("reporting", "first")).sort_values("n", ascending=False).head(15))
print("IATI acts with any loc", locs.aid.nunique(), "of", len(acts))
print(locs[locs.reporting != "AidData"].groupby("reporting").aid.nunique().sort_values(ascending=False).head(25))
print("commitment>0 share", (acts.commitment > 0).mean(), "spend>0", (acts.spend > 0).mean())
sec = pd.DataFrame(json.loads((RAW / "iati/dportal_sector_TL.json").read_text(encoding="utf-8")))
print(sec[["sector_group", "sector_code"]].head(), sec.sector_group.value_counts().head(30))

wbl = pd.read_csv(RAW / "aiddata_wb/WorldBank_GeocodedResearchRelease_Level1_v1.4.2/WorldBank_GeocodedResearchRelease_Level1_v1.4.2/data/locations.csv", encoding="latin1")
wbp = pd.read_csv(RAW / "aiddata_wb/WorldBank_GeocodedResearchRelease_Level1_v1.4.2/WorldBank_GeocodedResearchRelease_Level1_v1.4.2/data/projects.csv", encoding="latin1")
t = wbp[wbp.recipients.astype(str).str.contains("Timor")]
print("WB geocoded TL projects", len(t), "locations", wbl.project_id.isin(t.project_id).sum())
print(wbl[wbl.project_id.isin(t.project_id)].precision_code.value_counts())
