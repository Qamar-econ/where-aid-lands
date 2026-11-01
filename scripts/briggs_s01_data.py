"""Briggs (2017, IO) style dataset: Sub-Saharan Africa, DHS first-level regions.

For each country: one DHS standard survey 2003-2011 with region-level
  HC_WIXQ_P_LOW (population in lowest wealth quintile; Briggs's poverty measure),
  HC_HEFF_H_MPH (households possessing a mobile telephone; observed network access),
  HC_ELEC_H_ELC (households with electricity).
Region polygons from the DHS geometry API. Aid: AidData World Bank geocoded v1.4.2 (precision 1-2 locations),
projects approved after the survey year (through 2014). Population: WorldPop 1 km (2010). Night lights 2010 (harmonized).
Output: data/processed/briggs/regions.gpkg
"""
import os, json, time, requests
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.mask import mask
from shapely import wkt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw" / "briggs", ROOT / "data" / "processed" / "briggs"
RAW.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
B = "https://api.dhsprogram.com/rest/dhs"
SSA = "AO BJ BF BU CM CF TD KM CG CD CI ET GA GM GH GN KE LS LB MD MW ML MZ NM NI NG RW SN SL ZA SZ TZ TG UG ZM ZW ER ST".split()
IND = ["HC_WIXQ_P_LOW", "HC_HEFF_H_MPH", "HC_ELEC_H_ELC"]


def get(path, **p):
    for a in range(4):
        try:
            r = requests.get(B + path, params={"f": "json", "perpage": 5000, **p}, timeout=180)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(2 + 3 * a)
    raise RuntimeError(path)


f = RAW / "dhs_subnational.json"
if not f.exists():
    rows = []
    for c in SSA:
        rows += get("/data", indicatorIds=",".join(IND), countryIds=c, breakdown="subnational",
                    returnFields="SurveyId,SurveyYear,SurveyType,CountryName,DHS_CountryCode,IndicatorId,CharacteristicLabel,RegionId,Value,ByVariableLabel")["Data"]
    f.write_text(json.dumps(rows))
d = pd.DataFrame(json.loads(f.read_text()))
d = d[(d.SurveyType == "DHS") & d.SurveyYear.between(2003, 2011) & (d.ByVariableLabel.fillna("") == "")]
w = d.pivot_table(index=["SurveyId", "SurveyYear", "DHS_CountryCode", "CountryName", "RegionId", "CharacteristicLabel"],
                  columns="IndicatorId", values="Value", aggfunc="first").reset_index().dropna(subset=IND)
# one survey per country: the latest 2003-2011 survey with all three indicators for >= 3 regions
ok = w.groupby("SurveyId").RegionId.count().loc[lambda s: s >= 3].index
w = w[w.SurveyId.isin(ok)]
pick = w.groupby("DHS_CountryCode").SurveyYear.max()
w = w[w.SurveyYear == w.DHS_CountryCode.map(pick)]
print("countries", w.DHS_CountryCode.nunique(), "regions", len(w), w.groupby("CountryName").SurveyYear.first().to_dict())

g = RAW / "dhs_geometry.json"
if not g.exists():
    geo = []
    for s in w.SurveyId.unique():
        geo += get("/geometry", surveyIds=s, returnFields="SurveyId,RegionID,Coordinates,LevelRank")["Data"]
    g.write_text(json.dumps(geo))
G = pd.DataFrame(json.loads(g.read_text()))
G = G[G.RegionID.isin(w.RegionId)].drop_duplicates("RegionID")
G["geometry"] = G.Coordinates.map(wkt.loads)
R = gpd.GeoDataFrame(w.merge(G[["RegionID", "geometry"]], left_on="RegionId", right_on="RegionID"), geometry="geometry", crs=4326)
R["geometry"] = R.geometry.buffer(0)
print("regions with geometry", len(R))

# ---- World Bank geocoded projects (AidData v1.4.2) ----
wb = ROOT / "data/raw/aiddata_wb/WorldBank_GeocodedResearchRelease_Level1_v1.4.2/WorldBank_GeocodedResearchRelease_Level1_v1.4.2/data"
P = pd.read_csv(wb / "projects.csv", encoding="latin1"); L = pd.read_csv(wb / "locations.csv", encoding="latin1")
P["year"] = pd.to_datetime(P.start_actual_isodate, errors="coerce").dt.year
P["year"] = P.year.fillna(pd.to_numeric(P.transactions_start_year.astype(str).str[:4], errors="coerce"))
L = L[L.precision_code.isin([1, 2])].merge(P[["project_id", "year", "total_commitments", "ad_sector_codes"]], on="project_id")
L = gpd.GeoDataFrame(L, geometry=gpd.points_from_xy(L.longitude, L.latitude), crs=4326)
J = gpd.sjoin(L, R[["RegionId", "SurveyYear", "geometry"]], predicate="within")
J = J[(J.year > J.SurveyYear) & (J.year <= 2014)]                      # approved after the survey
nloc = L.groupby("project_id").size()
J["usd_w"] = J.total_commitments / J.project_id.map(nloc)
agg = J.groupby("RegionId").agg(n_proj=("project_id", "nunique"), n_loc=("project_id", "size"), usd=("usd_w", "sum"))
R = R.merge(agg, on="RegionId", how="left").fillna({"n_proj": 0, "n_loc": 0, "usd": 0})
R["any_aid"] = (R.n_proj > 0).astype(int)
R["years_obs"] = 2014 - R.SurveyYear

# ---- population (WorldPop 2010, 1 km) and night lights (harmonized 2010) ----
ISO = {"AO": "AGO", "BJ": "BEN", "BF": "BFA", "BU": "BDI", "CM": "CMR", "CF": "CAF", "TD": "TCD", "KM": "COM", "CG": "COG", "CD": "COD", "CI": "CIV",
       "ET": "ETH", "GA": "GAB", "GM": "GMB", "GH": "GHA", "GN": "GIN", "KE": "KEN", "LS": "LSO", "LB": "LBR", "MD": "MDG", "MW": "MWI", "ML": "MLI",
       "MZ": "MOZ", "NM": "NAM", "NI": "NER", "NG": "NGA", "RW": "RWA", "SN": "SEN", "SL": "SLE", "ZA": "ZAF", "SZ": "SWZ", "TZ": "TZA", "TG": "TGO",
       "UG": "UGA", "ZM": "ZMB", "ZW": "ZWE", "ER": "ERI", "ST": "STP"}
pops = []
for c, grp in R.groupby("DHS_CountryCode"):
    iso = ISO[c]; fp = RAW / f"wp_{iso}_2010.tif"
    if not fp.exists():
        r = requests.get(f"https://data.worldpop.org/GIS/Population/Global_2000_2020_1km_UNadj/2010/{iso}/{iso.lower()}_ppp_2010_1km_Aggregated_UNadj.tif", timeout=600)
        fp.write_bytes(r.content)
    with rasterio.open(fp) as s:
        for i, geom in zip(grp.index, grp.geometry):
            a, _ = mask(s, [geom], crop=True, filled=True, nodata=0); a = a[0].astype(float); a[a < 0] = 0
            pops.append((i, a.sum()))
R.loc[[i for i, _ in pops], "pop2010"] = [p for _, p in pops]
ntl_global = Path(os.environ["TEMP"]) / "tloda" / "ntl_tmp" / "Harmonized_DN_NTL_2010_calDMSP.tif"
if not ntl_global.exists():
    url = {f["name"]: f["download_url"] for f in requests.get("https://api.figshare.com/v2/articles/9828827", timeout=60).json()["files"]}["Harmonized_DN_NTL_2010_calDMSP.tif"]
    ntl_global.write_bytes(requests.get(url, timeout=900).content)
with rasterio.open(ntl_global) as s:
    R["ntl2010"] = [mask(s, [gm], crop=True, filled=True, nodata=0)[0][0].astype(float).sum() for gm in R.geometry]
R["area_km2"] = R.to_crs(6933).area / 1e6
R = R.drop(columns=["RegionID"]).rename(columns={"HC_WIXQ_P_LOW": "poor_q1", "HC_HEFF_H_MPH": "mobile", "HC_ELEC_H_ELC": "electricity"})
R.to_file(OUT / "regions.gpkg", driver="GPKG")
print(R[["CountryName", "SurveyYear"]].drop_duplicates().to_string())
print(R[["poor_q1", "mobile", "electricity", "n_proj", "pop2010", "ntl2010"]].describe().round(1).to_string())
