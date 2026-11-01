"""Time-aligned outcomes (research plan v2 §3): counts of non-infrastructure development projects per unit and window,
'active' (implementation period overlaps the window) and 'new' (started inside the window), with aligned covariates.

Windows   Timor-Leste: W1 2008-2012 (DHS 2009-10), W2 2014-2018 (DHS 2016), W3 2019-2025 (Ookla 2019/22/25)
          Nepal:       W1 2009-2013 (census 2011), W2 2019-2023 (census 2021; network NOT opened before pre-registration)
Period rules  Budget Book projects: active from first to last edition listed + 1 year (each edition lists next-year plans).
              Missing end year: start + median duration of the same source (flag 'end_imputed').
Outputs: data/processed/windows_TL.csv, data/processed/nepal/windows_NP.csv
"""
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.mask import mask
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEV = {"SOC_education", "SOC_health", "SOC_water", "SOC_other", "GOV", "PROD", "ENV_MULTI"}
SECT = {"SOC_education": "edu", "SOC_health": "health", "SOC_water": "water", "GOV": "gov", "PROD": "prod"}


def periods(PR):
    PR = PR.copy()
    bb = PR.source.eq("BudgetBook5")
    PR.loc[bb, "end_year"] = PR.loc[bb, "end_year"].fillna(PR.loc[bb, "start_year"]) + 1
    med = PR.assign(d=PR.end_year - PR.start_year).query("d >= 0 and d <= 30").groupby("source").d.median()
    PR["end_imputed"] = PR.end_year.isna() | (PR.end_year < PR.start_year)
    PR.loc[PR.end_imputed, "end_year"] = PR.loc[PR.end_imputed, "start_year"] + PR.loc[PR.end_imputed, "source"].map(med).fillna(4)
    return PR


def count(links, PR, units, windows, ucol):
    """links: uid, unit, cnt (1 or fractional), precise (0/1)."""
    L = links.merge(PR[["uid", "cat", "start_year", "end_year", "end_imputed", "ict"]], on="uid")
    L = L[L.cat.isin(DEV)]
    out = []
    for w, (a, b) in windows.items():
        act = L[(L.start_year <= b) & (L.end_year >= a)]
        new = L[L.start_year.between(a, b)]
        g = pd.DataFrame({ucol: units})
        for nm, df in [("active", act), ("new", new)]:
            g[f"{nm}_all"] = g[ucol].map(df.groupby("unit").cnt.sum()).fillna(0)
            p = df[df.precise == 1]
            g[f"{nm}"] = g[ucol].map(p.groupby("unit").uid.nunique()).fillna(0)                      # primary: precise (tier A/B)
            g[f"{nm}_noimp"] = g[ucol].map(p[~p.end_imputed].groupby("unit").uid.nunique()).fillna(0)
            for c, s in SECT.items():
                g[f"{nm}_{s}"] = g[ucol].map(p[p.cat == c].groupby("unit").uid.nunique()).fillna(0)
            g[f"{nm}_ict"] = g[ucol].map(p[p.ict].groupby("unit").uid.nunique()).fillna(0)
        g["window"] = w
        out.append(g)
    return pd.concat(out)


# ================= Timor-Leste =================
PR = periods(pd.read_csv(ROOT / "data/processed/oda_projects.csv"))
AS = pd.read_csv(ROOT / "data/processed/oda_assignments.csv")
X = pd.read_csv(ROOT / "data/processed/unit_project_links.csv")[["uid", "unit", "cnt", "precise"]]
tier = AS[AS.level == "posto"].groupby("uid").tier.min()
X.loc[X.uid.map(tier).isna() & (X.precise == 1), "precise"] = 0
U = gpd.read_file(ROOT / "data/processed/units.gpkg")
W_TL = {"W1": (2008, 2012), "W2": (2014, 2018), "W3": (2019, 2025)}
T = count(X, PR, U.unit, W_TL, "unit")
cov = U[["unit", "unit_name", "mun_pcode", "municipio", "pop2010", "pop2015", "pop2020", "cpop2010", "cpop2015", "access_phys", "access_net",
         "deprivation", "rwi", "tt_wet_min", "ntl2010", "ntl2013", "ntl2020", "area_km2"]].copy()
with rasterio.open(ROOT / "data/raw/ntl/ntl_2015_TLS.tif") as s:
    cov["ntl2015"] = [mask(s, [g], crop=True, filled=True, nodata=0)[0][0].astype(float).sum() for g in U.geometry]
import requests
dhs = pd.DataFrame(requests.get("https://api.dhsprogram.com/rest/dhs/data", params={"f": "json", "indicatorIds": "HC_HEFF_H_MPH", "countryIds": "TL",
                   "breakdown": "subnational", "perpage": 200}, timeout=120).json()["Data"])
dhs["municipio"] = dhs.CharacteristicLabel.replace({"Cova Lima": "Covalima", "Liquica": "Liquiçá", "Lautem": "Lautém"})
for sv, col in [("TL2009DHS", "dhs_mobile_2009"), ("TL2016DHS", "dhs_mobile_2016")]:
    cov = cov.merge(dhs[dhs.SurveyId == sv][["municipio", "Value"]].rename(columns={"Value": col}), on="municipio", how="left")
T = T.merge(cov, on="unit")
# aligned covariates per window
T["pop_w"] = np.select([T.window == "W1", T.window == "W2"], [T.cpop2010, T.cpop2015], T.cpop2015)
T["ntl_w"] = np.select([T.window == "W1", T.window == "W2"], [T.ntl2010, T.ntl2015], T.ntl2020)
T["net_w"] = np.select([T.window == "W1", T.window == "W2"], [T.dhs_mobile_2009, T.dhs_mobile_2016], T.access_net)
T["net_source"] = np.select([T.window == "W1", T.window == "W2"], ["DHS 2009-10 (municipality)", "DHS 2016 (municipality)"], "Ookla 2019-25 (posto)")
T.to_csv(ROOT / "data/processed/windows_TL.csv", index=False, encoding="utf-8-sig")
print("TL windows:", T.groupby("window")[["active", "new", "active_all"]].sum().round(0).to_dict("index"))
print("TL end imputed share:", PR.end_imputed.mean().round(3), PR.groupby("source").end_imputed.mean().round(2).to_dict())

# ================= Nepal =================
PRn = periods(pd.read_csv(ROOT / "data/processed/nepal/oda_projects.csv"))
ASn = pd.read_csv(ROOT / "data/processed/nepal/oda_assignments.csv")
ASn = ASn[ASn.level != "national"]
Ln = pd.DataFrame({"uid": ASn.uid, "unit": ASn.unit, "cnt": np.where(ASn.level == "district", 1.0, ASn.weight),
                   "precise": (ASn.tier.isin(["A", "B"]) & (ASn.level == "district")).astype(int)})
Ln = Ln.groupby(["uid", "unit"], as_index=False).agg(cnt=("cnt", "max"), precise=("precise", "max"))
Dn = pd.read_csv(ROOT / "data/processed/nepal/district_panel.csv")
W_NP = {"W1": (2009, 2013), "W2": (2019, 2023)}
N = count(Ln, PRn, Dn.k, W_NP, "k").merge(Dn[["k", "district", "province", "capital", "pop2011", "pop2023", "net_census", "mobile_pct", "internet_pct",
                                                 "phys_idx", "deprivation", "rwi", "ntl2010", "tt_wet_min", "area_km2", "ook_idx"]], on="k")
N["pop_w"] = np.where(N.window == "W1", N.pop2011, N.pop2023)
N["net_w"] = np.where(N.window == "W1", N.net_census, np.nan)          # W2 network (census 2021) intentionally not opened yet
N.to_csv(ROOT / "data/processed/nepal/windows_NP.csv", index=False, encoding="utf-8-sig")
print("NP windows:", N.groupby("window")[["active", "new", "active_all"]].sum().round(0).to_dict("index"))
print("NP end imputed share:", PRn.end_imputed.mean().round(3))
