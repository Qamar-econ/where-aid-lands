"""Build the analysis dataset (62 units = 61 postos + 'Dili (urban)' merging the 4 Dili city postos).

Researcher decisions (confirmed by the PI):
  * Dili city postos Dom Aleixo, Nain Feto, Vera Cruz, Cristo Rei merged into one unit 'Dili (urban)'.
  * Municipality-level projects split equally across the municipality's units; precise-only kept as robustness.
  * Main outcome = number of non-infrastructure development projects (social, governance, productive,
    environment/multisector). Humanitarian, 'other/unallocated' and access infrastructure are counted separately.
  * Accessibility = PCA composite of physical + network components; components reported separately.
Outputs: data/processed/units.gpkg, units.csv, unit_year_panel.csv
"""
import numpy as np, pandas as pd, geopandas as gpd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
DILI_URBAN_NAMES = {"Dom Aleixo", "Nain Feto", "Vera Cruz", "Cristo Rei"}
DEV = {"SOC_education", "SOC_health", "SOC_water", "SOC_other", "GOV", "PROD", "ENV_MULTI"}

cov = gpd.read_file(PROC / "posto_covariates.gpkg")
acc = pd.read_csv(PROC / "posto_access.csv")
pan = pd.read_csv(PROC / "posto_access_panel.csv")
cov = cov.merge(acc, on="pcode", how="left")
dili_u = cov.posto.isin(DILI_URBAN_NAMES)
assert dili_u.sum() == 4
cov["unit"] = np.where(dili_u, "TL06U", cov.pcode)
cov["unit_name"] = np.where(dili_u, "Dili (urban)", cov.posto)

w = cov["pop2022"]
WMEAN = ["rwi", "elev_mean", "slope_mean", "steep_share", "wet_rain_mm", "tt_dry_min", "tt_wet_min", "walk_min",
         "share_pop_2km_road", "wet_penalty_min", "urban_share"]
SUM = ["area_km2", "pop2005", "pop2010", "pop2015", "pop2020", "pop2022", "road_km", "paved_km"] + \
      [c for c in cov.columns if c.startswith("ook_") and (c.endswith("_tiles") or c.endswith("_tests") or c.endswith("_all"))]
g = cov.groupby("unit")
U = g[SUM].sum()
for c in WMEAN:
    U[c] = g.apply(lambda d: np.average(d[c], weights=d.pop2022), include_groups=False)
for c in ["ook_mobile_2025_dl_mbps", "ook_mobile_2019_dl_mbps"]:
    tcol = c.replace("_dl_mbps", "_tests")
    U[c] = g.apply(lambda d: np.average(d[c].fillna(0), weights=d[tcol]) if d[tcol].sum() > 0 else np.nan, include_groups=False)
meta = g[["unit_name", "mun_pcode", "municipio"]].first()
U = meta.join(U)
U["geometry"] = g.geometry.apply(lambda s: s.union_all())
U = gpd.GeoDataFrame(U.reset_index(), geometry="geometry", crs=4326)
U["density"] = U.pop2022 / U.area_km2
U["road_density"] = U.road_km / U.area_km2
U["paved_share"] = U.paved_km / U.road_km
U["net_tiles_per10k"] = U.ook_mobile_tiles_all / U.pop2022 * 1e4
U["net_tests_per1k"] = U.ook_mobile_tests_all / U.pop2022 * 1e3
U["log_pop"] = np.log(U.pop2022)
U["log_tt_wet"] = np.log(U.tt_wet_min)
U["oecusse"] = (U.mun_pcode == "TL12").astype(int)
U["island"] = (U.unit_name == "Atauro").astype(int)

# ---------- composite accessibility (PCA, first component) ----------
comp = pd.DataFrame({
    "neg_log_tt_wet": -np.log(U.tt_wet_min),
    "paved_share": U.paved_share,
    "log_road_density": np.log(U.road_density),
    "neg_steep_share": -U.steep_share,
    "neg_wet_penalty_share": -(U.wet_penalty_min / U.tt_wet_min),
    "log_net_tiles": np.log1p(U.net_tiles_per10k),
    "log_net_tests": np.log1p(U.net_tests_per1k),
})
Z = (comp - comp.mean()) / comp.std()
vals, vecs = np.linalg.eigh(np.cov(Z.T))
order = np.argsort(vals)[::-1]
pc1 = vecs[:, order[0]]
if pc1[0] < 0:
    pc1 = -pc1
U["access_pc1"] = Z.values @ pc1
U["access_pc1"] = (U.access_pc1 - U.access_pc1.mean()) / U.access_pc1.std()
# physical-only and network-only sub-indices
U["access_phys"] = Z[["neg_log_tt_wet", "paved_share", "log_road_density", "neg_steep_share", "neg_wet_penalty_share"]].mean(axis=1)
U["access_net"] = Z[["log_net_tiles", "log_net_tests"]].mean(axis=1)
load = pd.Series(pc1, index=comp.columns)
expl = vals[order[0]] / vals.sum()
pd.DataFrame({"loading": load, "explained_var_pc1": expl}).to_csv(PROC / "access_pca_loadings.csv")
print("PCA loadings (PC1, %.0f%% var):\n" % (expl * 100), load.round(3))

# ---------- outcomes ----------
PR = pd.read_csv(PROC / "oda_projects.csv")
AS = pd.read_csv(PROC / "oda_assignments.csv")
PR = PR[(PR.start_year >= 2000) & (PR.start_year <= 2026)]
AS = AS[AS.uid.isin(PR.uid)].merge(PR[["uid", "cat", "start_year", "usd", "donor_group", "source"]], on="uid")
p2u = cov.set_index("pcode").unit.to_dict()
units_in_mun = U.groupby("mun_pcode").unit.apply(list).to_dict()
units_in_mun["TL06"] = ["TL06U"]   # 'Dili' in project records means the city -> Dili (urban), not Atauro/Metinaro

rows = []
for r in AS.itertuples():
    if r.level == "posto":
        rows.append((r.uid, p2u[r.unit], 1.0, 1, r.weight))
    elif r.level == "municipio":
        us = units_in_mun.get(r.unit, [])
        for u in us:
            rows.append((r.uid, u, 1.0 / len(us), 0, r.weight / len(us)))
X = pd.DataFrame(rows, columns=["uid", "unit", "cnt", "precise", "usd_w"])
# a project touching several postos of the same unit counts once there
X = X.groupby(["uid", "unit"], as_index=False).agg(cnt=("cnt", "max"), precise=("precise", "max"), usd_w=("usd_w", "sum"))
dv = PR[PR.cat.isin(DEV) & (PR.usd > 0)]
q1, q2 = dv.usd.quantile([1 / 3, 2 / 3])
PR["size"] = np.select([PR.usd.isna() | (PR.usd <= 0), PR.usd <= q1, PR.usd <= q2], ["unknown", "small", "medium"], "large")
PR["dur"] = np.select([PR.duration.isna(), PR.duration <= 2, PR.duration <= 5], ["unknown", "short", "medium"], "long")
print(f"size tercile cut-offs (dev projects): small <= ${q1/1e6:.2f}m < medium <= ${q2/1e6:.2f}m < large")
# projects whose only geographic evidence is the word "Dili" in text (may denote HQ rather than target) -> robustness
g = AS.groupby("uid").agg(meth=("method", lambda s: set(s)), units=("unit", lambda s: set(s)))
PR["dili_text_only"] = PR.uid.map(lambda u: u in g.index and g.at[u, "meth"] == {"text"} and g.at[u, "units"] == {"TL06"})
X = X.merge(PR[["uid", "cat", "start_year", "end_year", "usd", "donor_group", "source", "ict", "size", "dur", "dili_text_only"]], on="uid")
X["usd_alloc"] = X.usd.fillna(0) * X.usd_w
X["grp"] = np.select([X.cat.isin(DEV), X.cat.str.startswith("ACCESS"), X.cat == "HUMAN"], ["dev", "access", "human"], "other")
X["era"] = np.where(X.start_year <= 2015, "e1", "e2")
X.to_csv(PROC / "unit_project_links.csv", index=False)

BILAT = {"Korea", "Japan", "Australia", "USA", "EU", "Portugal", "New Zealand", "Germany", "China", "Other bilateral/other"}
MULTI = {"World Bank", "ADB", "UN agencies", "Global funds"}


def agg(df, suffix):
    o = pd.DataFrame(index=U.unit)
    o[f"n_{suffix}"] = df.groupby("unit").cnt.sum()
    o[f"np_{suffix}"] = df[df.precise == 1].groupby("unit").cnt.sum()
    o[f"usd_{suffix}"] = df.groupby("unit").usd_alloc.sum()
    return o.fillna(0)


parts = [agg(X[X.grp == "dev"], "dev"), agg(X[X.grp == "access"], "access"), agg(X[X.grp == "human"], "human"),
         agg(X[(X.grp == "dev") & (X.era == "e1")], "dev_e1"), agg(X[(X.grp == "dev") & (X.era == "e2")], "dev_e2"),
         agg(X[(X.grp == "dev") & X.donor_group.isin(BILAT)], "dev_bilat"),
         agg(X[(X.grp == "dev") & X.donor_group.isin(MULTI)], "dev_multi"),
         agg(X[(X.grp == "dev") & X.donor_group.isin({"Korea", "Japan"})], "dev_kj")]
for c in ["SOC_education", "SOC_health", "SOC_water", "SOC_other", "GOV", "PROD", "ENV_MULTI",
          "ACCESS_transport", "ACCESS_energy", "ACCESS_comms"]:
    parts.append(agg(X[X.cat == c], c))
dvx = X[X.grp == "dev"]
parts += [agg(dvx[dvx.ict], "dev_ict"), agg(dvx[~dvx.ict], "dev_nonict"), agg(dvx[~dvx.dili_text_only], "dev_strict"),
          agg(dvx[dvx.start_year <= 2010], "dev_0610"), agg(dvx[dvx.start_year.between(2016, 2025)], "dev_1625")]
for s in ["small", "medium", "large"]:
    parts.append(agg(dvx[dvx["size"] == s], f"dev_{s}"))
for s in ["short", "medium", "long"]:
    parts.append(agg(dvx[dvx["dur"] == s], f"dur_{s}"))
O = pd.concat(parts, axis=1).loc[:, lambda d: ~d.columns.duplicated()].reset_index()
U = U.merge(O, on="unit", how="left")

# ---------- census (2010 wall chart need indicators, 2010/2015 population) ----------
cen = pd.read_csv(PROC / "posto_census.csv")
cen["unit"] = cen.pcode.map(p2u)
NEED = ["water_improved", "sanit_improved", "literacy_15_24", "cook_clean", "ger_secondary", "lfpr"]
cu = cen.groupby("unit").apply(lambda d: pd.Series({"cpop2010": d.pop2010.sum(), "cpop2015": d.pop2015.sum(),
                                                     "cpop2022": d.pop2022c.sum() if d.pop2022c.notna().all() else np.nan,
                                                     **{c: np.average(d[c], weights=d.pop2010) for c in NEED}}), include_groups=False).reset_index()
cu["g_10_15"] = (cu.cpop2015 / cu.cpop2010) ** 0.2 - 1
cu["g_15_22"] = (cu.cpop2022 / cu.cpop2015) ** (1 / 7) - 1
dep = pd.DataFrame({"no_water": 100 - cu.water_improved, "no_sanit": 100 - cu.sanit_improved, "illiterate": 100 - cu.literacy_15_24,
                    "dirty_fuel": 100 - cu.cook_clean, "no_secondary": -cu.ger_secondary})
Zd = (dep - dep.mean()) / dep.std()
cu["deprivation"] = Zd.mean(axis=1) / Zd.mean(axis=1).std()       # census deprivation index (2010), 1 SD units
U = U.merge(cu, on="unit", how="left")
# ---------- night lights (harmonized DMSP-VIIRS): pre-period 'hub-ness' independent of later network ----------
import rasterio
from rasterio.mask import mask as rmask
for y in [2000, 2005, 2010, 2013, 2020]:
    with rasterio.open(ROOT / "data" / "raw" / "ntl" / f"ntl_{y}_TLS.tif") as src:
        vals, lit = [], []
        for geom in U.geometry:
            a, _ = rmask(src, [geom], crop=True, filled=True, nodata=0)
            a = a[0].astype(float)
            vals.append(a.sum()); lit.append((a > 7).sum() / max((a >= 0).sum(), 1))
    U[f"ntl{y}"] = vals; U[f"lit{y}"] = lit
U["log_ntl2010"] = np.log1p(U.ntl2010)
U["log_ntl2000"] = np.log1p(U.ntl2000)

CAPITALS = {"Aileu Vila", "Ainaro", "Baucau", "Maliana", "Suai", "Dili (urban)", "Ermera", "Lospalos", "Liquiçá", "Same",
            "Manatuto", "Pante Macassar", "Viqueque"}
U["mun_capital"] = U.unit_name.isin(CAPITALS).astype(int)
assert U.mun_capital.sum() == 13
U["dev_per10k"] = U.n_dev / U.pop2022 * 1e4
U["usd_dev_pc"] = U.usd_dev / U.pop2022
U.to_file(PROC / "units.gpkg", driver="GPKG")
U.drop(columns="geometry").to_csv(PROC / "units.csv", index=False, encoding="utf-8-sig")

# ---------- unit-year panel (new project starts, 2006-2025) ----------
pan["unit"] = pan.pcode.map(p2u)
popw = cov.set_index("pcode").pop2022
pan["w"] = pan.pcode.map(popw)
P = pan.groupby(["unit", "year"]).apply(lambda d: pd.Series({
    "tt_wet_min": np.average(d.tt_wet_min, weights=d.w), "tt_dry_min": np.average(d.tt_dry_min, weights=d.w)}),
    include_groups=False).reset_index()
cnt = X[X.grp == "dev"].groupby(["unit", "start_year"]).cnt.sum().rename("n_dev_new").reset_index().rename(columns={"start_year": "year"})
cnp = X[(X.grp == "dev") & (X.precise == 1)].groupby(["unit", "start_year"]).cnt.sum().rename("np_dev_new").reset_index().rename(columns={"start_year": "year"})
cna = X[X.grp == "access"].groupby(["unit", "start_year"]).cnt.sum().rename("n_access_new").reset_index().rename(columns={"start_year": "year"})
P = P.merge(cnt, how="left").merge(cnp, how="left").merge(cna, how="left").fillna({"n_dev_new": 0, "np_dev_new": 0, "n_access_new": 0})
P = P[(P.year >= 2006) & (P.year <= 2025)]
base = P[P.year == 2006].set_index("unit").tt_wet_min
P["gain_min"] = P.unit.map(base) - P.tt_wet_min
# treatment timing: first year with >= 60 min wet-season gain
first = P[P.gain_min >= 60].groupby("unit").year.min()
P["treat_year"] = P.unit.map(first)
P.to_csv(PROC / "unit_year_panel.csv", index=False)
print("units:", len(U), " treated units (>=60 min gain):", first.size, first.value_counts().sort_index().to_dict())
print(U[["unit_name", "pop2022", "tt_wet_min", "access_pc1", "rwi", "n_dev", "np_dev", "dev_per10k", "n_access"]]
      .sort_values("access_pc1", ascending=False).round(2).to_string())
